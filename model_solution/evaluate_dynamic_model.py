#!/usr/bin/env python3
"""
Скрипт пакетной и индивидуальной проверки динамической модели движения трамвая (model_solution).
Сравнивает аналитическую модель (чисто по driver_cmd и карте path) с эталоном GNSS
на выборке валидных прогонов.

Запуск:
    .venv/bin/python model_solution/evaluate_dynamic_model.py [--bag <path>] [--mass 23000] [--plot]
"""

import sys
import argparse
from pathlib import Path

# Добавляем пути в sys.path
_current_dir = Path(__file__).resolve().parent
_repo_dir = _current_dir.parent
for p in [str(_repo_dir), str(_current_dir), str(_repo_dir / "scripts"), str(_repo_dir / "solution")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import csv
import numpy as np
import matplotlib.pyplot as plt
from rosbags.highlevel import AnyReader
from batch_evaluate_filtered import setup_typestore

from tram_dynamic_model import TramParameters
from dynamic_odometry_node import DynamicOdometryNode


def process_bag(
    bag_path: Path,
    route_name: str,
    path_file: Path,
    typestore,
    mass_kg: float = 23000.0,
    save_plot_path: Path = None
):
    """
    Обработка одного bag-файла с оценкой динамической модели против GNSS и тележек.
    """
    params = TramParameters(mass_kg=mass_kg)
    node = DynamicOdometryNode(
        path_geometry_file_or_data=path_file,
        tram_params=params
    )

    gnss_vel_records = []
    bogie_vel_records = []
    front_v_last = 0.0
    rear_v_last = 0.0

    target_topics = {
        "/vehicle/driver_position_cmd",
        "/vehicle/front_bogie_velocity",
        "/vehicle/rear_bogie_velocity",
        "/sensing/gnss/master/vel"
    }

    try:
        with AnyReader([bag_path], default_typestore=typestore) as reader:
            conns = [c for c in reader.connections if c.topic in target_topics]
            if not conns:
                return None

            for conn, timestamp, raw in reader.messages(connections=conns):
                msg = reader.deserialize(raw, conn.msgtype)
                t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                if t <= 0:
                    t = timestamp * 1e-9

                if conn.topic == "/vehicle/driver_position_cmd":
                    node.update_driver_cmd(t, msg.position)
                elif conn.topic == "/vehicle/front_bogie_velocity":
                    front_v_last = float(msg.velocity) / 3.6
                    bogie_vel_records.append((t, 0.5 * (front_v_last + rear_v_last)))
                elif conn.topic == "/vehicle/rear_bogie_velocity":
                    rear_v_last = float(msg.velocity) / 3.6
                    bogie_vel_records.append((t, 0.5 * (front_v_last + rear_v_last)))
                elif conn.topic == "/sensing/gnss/master/vel":
                    vx = msg.twist.linear.x
                    vy = msg.twist.linear.y
                    v_gnss = float(np.hypot(vx, vy))
                    gnss_vel_records.append((t, v_gnss))
    except Exception as e:
        print(f"Ошибка чтения {bag_path.name}: {e}")
        return None

    # Получаем расчетную историю от матмодели
    data_model = node.get_arrays()
    if not data_model or len(data_model["timestamp"]) < 10:
        return None

    t_m = data_model["timestamp"]
    v_m = data_model["v_est"]
    s_m = data_model["distance"]
    cmd_m = data_model["driver_cmd"]
    slope_m = data_model["slope"]
    accel_m = data_model["acceleration"]

    duration_sec = float(t_m[-1] - t_m[0])
    if duration_sec < 10.0 or len(gnss_vel_records) < 20:
        return None

    # Интерполируем GNSS на временную сетку модели
    t_g = np.array([x[0] for x in gnss_vel_records])
    v_g = np.array([x[1] for x in gnss_vel_records])
    s_idx = np.argsort(t_g)
    t_g, v_g = t_g[s_idx], v_g[s_idx]

    v_gnss_interp = np.interp(t_m, t_g, v_g)
    dt_m = np.diff(t_m, prepend=t_m[0])
    dt_m[0] = 0.0
    r_gnss = np.cumsum(v_gnss_interp * dt_m)
    tot_gnss = float(r_gnss[-1])

    if tot_gnss < 150.0:  # Пропускаем чистые стоянки
        return None

    # Также рассчитываем эталонную дистанцию по тележкам
    if bogie_vel_records:
        t_b = np.array([x[0] for x in bogie_vel_records])
        v_b = np.array([x[1] for x in bogie_vel_records])
        s_idx_b = np.argsort(t_b)
        t_b, v_b = t_b[s_idx_b], v_b[s_idx_b]
        v_bogie_interp = np.interp(t_m, t_b, v_b)
        r_bogie = np.cumsum(v_bogie_interp * dt_m)
        tot_bogie = float(r_bogie[-1])
    else:
        v_bogie_interp = v_gnss_interp
        r_bogie = r_gnss
        tot_bogie = tot_gnss

    tot_model = float(s_m[-1])

    # Ошибки
    err_dist_m = tot_model - tot_gnss
    err_pct = (err_dist_m / tot_gnss) * 100.0
    abs_err_pct = abs(err_pct)

    # Ошибки скорости
    v_diff = v_m - v_gnss_interp
    v_rmse = float(np.sqrt(np.mean(v_diff ** 2)))
    v_mae = float(np.mean(np.abs(v_diff)))

    # Ошибка тележек для сравнения
    err_bogie_m = tot_bogie - tot_gnss
    pct_bogie = abs(err_bogie_m) / tot_gnss * 100.0

    # Построение графика при необходимости
    if save_plot_path is not None:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
        t_rel = t_m - t_m[0]

        # 1. Скорость
        ax1.plot(t_rel, v_gnss_interp * 3.6, label="GNSS (эталон)", color="black", linewidth=1.8, alpha=0.85)
        ax1.plot(t_rel, v_bogie_interp * 3.6, label="Тележки (тахометры)", color="#1f77b4", linestyle="--", linewidth=1.2, alpha=0.7)
        ax1.plot(t_rel, v_m * 3.6, label=f"Матмодель (m={mass_kg/1000:.0f}т)", color="#d62728", linewidth=1.6)
        ax1.set_ylabel("Скорость (км/ч)", fontsize=11)
        ax1.set_title(
            f"Анализ матмодели: {bag_path.name} ({route_name}) | "
            f"Ошибка пути: {err_dist_m:+.1f} м ({err_pct:+.1f}%) | "
            f"RMSE v: {v_rmse*3.6:.2f} км/ч",
            fontsize=13, fontweight="bold"
        )
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="upper right")

        # 2. Пройденный путь
        ax2.plot(t_rel, r_gnss, label=f"GNSS путь: {tot_gnss:.1f} м", color="black", linewidth=1.8)
        ax2.plot(t_rel, r_bogie, label=f"Тележки: {tot_bogie:.1f} м (ош: {pct_bogie:.1f}%)", color="#1f77b4", linestyle="--", linewidth=1.2)
        ax2.plot(t_rel, s_m, label=f"Матмодель: {tot_model:.1f} м (ош: {abs_err_pct:.1f}%)", color="#d62728", linewidth=1.6)
        ax2.set_ylabel("Пройденный путь (м)", fontsize=11)
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="upper left")

        # 3. Положение ручки водителя
        ax3.step(t_rel, cmd_m, label="Положение ручки (driver_cmd: -15..+15)", color="#2ca02c", linewidth=1.3)
        ax3.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax3.set_ylabel("Ручка водителя", fontsize=11)
        ax3.grid(True, linestyle=":", alpha=0.6)
        ax3.legend(loc="upper right")

        # 4. Уклон пути и ускорение
        ax4.plot(t_rel, slope_m, label="Уклон пути i(s) (‰)", color="#9467bd", linewidth=1.2)
        ax4.plot(t_rel, accel_m * 10.0, label="Расчетное ускорение a × 10 (м/с²)", color="#ff7f0e", linewidth=1.2, alpha=0.8)
        ax4.set_ylabel("Уклон / a*10", fontsize=11)
        ax4.set_xlabel("Время от старта (сек)", fontsize=11)
        ax4.grid(True, linestyle=":", alpha=0.6)
        ax4.legend(loc="upper right")

        plt.tight_layout()
        save_plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_plot_path, dpi=130)
        plt.close(fig)

    return {
        "bag_id": bag_path.name,
        "route": route_name,
        "duration_sec": duration_sec,
        "r_gnss_m": tot_gnss,
        "r_bogie_m": tot_bogie,
        "err_bogie_m": err_bogie_m,
        "pct_bogie": pct_bogie,
        "r_model_m": tot_model,
        "err_model_m": err_dist_m,
        "pct_model": abs_err_pct,
        "v_rmse_kmh": v_rmse * 3.6,
        "v_mae_kmh": v_mae * 3.6,
    }


def main():
    parser = argparse.ArgumentParser(description="Оценка динамической модели движения трамвая.")
    parser.add_argument("--bag", type=str, help="Путь к конкретному bag-файлу")
    parser.add_argument("--mass", type=float, default=23000.0, help="Масса вагона в кг (по умолчанию 23000)")
    parser.add_argument("--plot", action="store_true", help="Сохранять графики расхождения")
    parser.add_argument("--max_bags", type=int, default=15, help="Максимальное количество мешков для пакетного теста")
    args = parser.parse_args()

    ws_dir = _repo_dir
    typestore = setup_typestore(ws_dir)

    geom_files = {
        "щук-талл": ws_dir / "pathgrath" / "щукинская - таллинская.json",
        "талл-щук": ws_dir / "pathgrath" / "таллинская - щукинская.json",
    }

    results = []

    if args.bag:
        bag_path = Path(args.bag)
        route = "щук-талл" if "щук-талл" in str(bag_path) else "талл-щук"
        plot_path = ws_dir / "reports" / f"dynamic_model_{bag_path.name}.png" if args.plot else None
        res = process_bag(bag_path, route, geom_files[route], typestore, mass_kg=args.mass, save_plot_path=plot_path)
        if res:
            results.append(res)
    else:
        print(f"=== Запуск пакетного тестирования матмодели (Масса = {args.mass / 1000.0:.1f} т) ===")
        routes_to_test = [("щук-талл", ws_dir / "data" / "щук-талл"), ("талл-щук", ws_dir / "data" / "талл-щук")]
        
        bag_count = 0
        for r_name, r_dir in routes_to_test:
            if not r_dir.exists():
                continue
            bags = sorted([p for p in r_dir.iterdir() if p.is_dir()])
            for bag in bags:
                plot_file = ws_dir / "reports" / f"dynamic_model_{bag.name}.png" if (args.plot and bag_count < 3) else None
                res = process_bag(bag, r_name, geom_files[r_name], typestore, mass_kg=args.mass, save_plot_path=plot_file)
                if res is not None:
                    results.append(res)
                    bag_count += 1
                    print(
                        f"[{bag_count:02d}] {res['bag_id']} ({res['route']}): "
                        f"L_gnss={res['r_gnss_m']:.0f}м | "
                        f"L_model={res['r_model_m']:.0f}м | "
                        f"Ош_модели={res['err_model_m']:+.1f}м ({res['pct_model']:.1f}%) | "
                        f"RMSE_v={res['v_rmse_kmh']:.2f} км/ч | "
                        f"Ош_тележек={res['pct_bogie']:.1f}%"
                    )
                    if bag_count >= args.max_bags:
                        break
            if bag_count >= args.max_bags:
                break

    if not results:
        print("Нет валидных результатов для отображения.")
        return

    # Сохраняем сводный CSV отчет
    reports_dir = ws_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_file = reports_dir / f"dynamic_model_evaluation_m{int(args.mass/1000)}t.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    # Итоговая статистика
    pcts_model = [r["pct_model"] for r in results]
    pcts_bogie = [r["pct_bogie"] for r in results]
    v_rmses = [r["v_rmse_kmh"] for r in results]

    print("\n" + "=" * 70)
    print(f"ИТОГОВАЯ СВОДКА ТОЧНОСТИ МАТМОДЕЛИ (Выборка: {len(results)} прогонов, масса = {args.mass/1000:.1f} т):")
    print("=" * 70)
    print(f"Средняя ошибка пути модели:    {np.mean(pcts_model):.2f}% (медиана: {np.median(pcts_model):.2f}%)")
    print(f"Мин / Макс ошибка модели:      {np.min(pcts_model):.2f}% / {np.max(pcts_model):.2f}%")
    print(f"Средний RMSE скорости модели:  {np.mean(v_rmses):.2f} км/ч")
    print("-" * 70)
    print(f"Для сравнения (колеса/тахометры): Средняя ошибка {np.mean(pcts_bogie):.2f}%")
    print(f"Отчет сохранен в: {csv_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()

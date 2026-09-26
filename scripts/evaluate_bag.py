import sys
from pathlib import Path
_repo_dir = Path(__file__).resolve().parent.parent
for _p in [str(_repo_dir), str(_repo_dir / "solution"), str(_repo_dir / "scripts")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

#!/usr/bin/env python3
"""
Скрипт проверки и валидации решения резервной одометрии (без ROS, чистый Python):
R = sum(V_cp(t) * delta_t)

Сравнивает полученное расстояние и скорость с эталоном GNSS и формирует:
1. Таблицу погрешности по участкам маршрута (чекпоинты пути).
2. Сводную итоговую таблицу метрик качества (MAE, RMSE, финишная погрешность).
3. Графики скорости, пути и ошибки во времени.
"""

import sys
import os
import argparse
from pathlib import Path
import numpy as np

# Задаем рабочий каталог для кэша matplotlib
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

from odometry_node import DeadReckoningNode


def setup_typestore(ws_dir: Path):
    """Регистрирует кастомные типы сообщений tram_vehicle_msgs в rosbags."""
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    msg_dir = ws_dir / "tram_vehicle_msgs" / "msg"
    
    vel_msg_path = msg_dir / "VelocitySensor.msg"
    cmd_msg_path = msg_dir / "DriverControllerCommand.msg"
    
    if vel_msg_path.exists():
        typestore.register(get_types_from_msg(
            vel_msg_path.read_text(), "tram_vehicle_msgs/msg/VelocitySensor"
        ))
    if cmd_msg_path.exists():
        typestore.register(get_types_from_msg(
            cmd_msg_path.read_text(), "tram_vehicle_msgs/msg/DriverControllerCommand"
        ))
    return typestore


from typing import Optional

def evaluate_bag(bag_path: Path, ws_dir: Path, step_dist_m: float = 500.0, csv_path: Optional[Path] = None, save_plot: bool = True):
    """
    Прогон одометрии по bag-файлу и сравнение с GNSS.
    """
    if not bag_path.exists():
        raise FileNotFoundError(f"Каталог bag не найден: {bag_path}")

    typestore = setup_typestore(ws_dir)
    node = DeadReckoningNode(input_in_kmh=True, integration_method="trapezoidal")

    print(f"\n================================================================================")
    print(f"  Обработка прогона: {bag_path.name}")
    print(f"================================================================================")

    # Коллекторы данных
    gnss_records = []      # (t, v_gnss)
    gnss_fix_records = []  # (t, lat, lon)

    target_topics = {
        "/vehicle/front_bogie_velocity",
        "/vehicle/rear_bogie_velocity",
        "/sensing/gnss/master/vel",
        "/sensing/gnss/master/fix",
    }

    with AnyReader([bag_path], default_typestore=typestore) as reader:
        conns = [c for c in reader.connections if c.topic in target_topics]
        if not conns:
            print("Предупреждение: в bag-файле не найдены целевые топики телеметрии!")
            return

        for conn, timestamp, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            # Используем метку времени из заголовка сообщения
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            
            # Если stamp равен нулю, берем timestamp сохранения сообщения в bag
            if t <= 0:
                t = timestamp * 1e-9

            if conn.topic == "/vehicle/front_bogie_velocity":
                node.update_front(t, msg.velocity)
            elif conn.topic == "/vehicle/rear_bogie_velocity":
                node.update_rear(t, msg.velocity)
            elif conn.topic == "/sensing/gnss/master/vel":
                # Модуль горизонтальной скорости из TwistStamped
                vx = msg.twist.linear.x
                vy = msg.twist.linear.y
                v_gnss = float(np.hypot(vx, vy))
                gnss_records.append((t, v_gnss))
            elif conn.topic == "/sensing/gnss/master/fix":
                if hasattr(msg, "status") and msg.status.status >= 0:
                    gnss_fix_records.append((t, msg.latitude, msg.longitude))

    odom_data = node.get_arrays()
    if not odom_data or len(odom_data["timestamp"]) == 0:
        print("Ошибка: не удалось извлечь данные одометрии.")
        return

    t_odom = odom_data["timestamp"]
    v_odom = odom_data["v_avg"]
    r_odom = odom_data["distance"]

    has_gnss = len(gnss_records) > 10

    if not has_gnss:
        print("\n[Внимание] В данном bag-файле отсутствуют данные эталона GNSS.")
        print(f"Итоговое пройденное расстояние R: {r_odom[-1]:.2f} м")
        print(f"Длительность записи: {t_odom[-1] - t_odom[0]:.1f} с")
        return

    # Синхронизация GNSS и расчет эталонного пути
    t_gnss_raw = np.array([x[0] for x in gnss_records])
    v_gnss_raw = np.array([x[1] for x in gnss_records])

    # Сортируем GNSS по времени на случай мелких перескоков
    sort_idx = np.argsort(t_gnss_raw)
    t_gnss_raw = t_gnss_raw[sort_idx]
    v_gnss_raw = v_gnss_raw[sort_idx]

    # Интерполируем скорость GNSS на моменты времени одометрии
    v_gnss_interp = np.interp(t_odom, t_gnss_raw, v_gnss_raw)

    # Интегрируем GNSS скорость на той же сетке времени одометрии
    dt_odom = np.diff(t_odom, prepend=t_odom[0])
    dt_odom[0] = 0.0
    r_gnss = np.cumsum(v_gnss_interp * dt_odom)

    # Ошибка по расстоянию во времени: Delta R(t) = R_odom(t) - R_gnss(t)
    delta_r = r_odom - r_gnss
    abs_delta_r = np.abs(delta_r)

    # -------------------------------------------------------------------------
    # 1. ТАБЛИЦА ПОГРЕШНОСТИ ПО УЧАСТКАМ МАРШРУТА
    # -------------------------------------------------------------------------
    final_rg = r_gnss[-1]
    final_ro = r_odom[-1]

    print("\n" + "=" * 135)
    print(f"                      ТАБЛИЦА ПОГРЕШНОСТИ ПО УЧАСТКАМ МАРШРУТА (шаг ~{step_dist_m:.0f} м)")
    print(f"                      [Полная длина маршрута по GNSS: {final_rg:.2f} м | по Одометрии: {final_ro:.2f} м]")
    print("=" * 135)
    header = (
        f"{'Участок':^9} | {'Время':^9} | {'R одом':^10} | {'R GNSS':^10} | "
        f"{'ΔR након':^9} | {'% от тек':^9} | {'% от ВСЕГО':^11} | {'ΔR отрезка':^11} | {'% отрезка':^9} | "
        f"{'V_ср':^7} | {'V_GNSS':^7}"
    )
    print(header)
    print("-" * 135)

    total_r = r_odom[-1]
    checkpoints = np.arange(step_dist_m, total_r, step_dist_m)
    if len(checkpoints) == 0 or checkpoints[-1] < total_r - (step_dist_m * 0.2):
        checkpoints = np.append(checkpoints, total_r)

    t_start = t_odom[0]
    prev_ro = 0.0
    prev_rg = 0.0

    for idx, cp in enumerate(checkpoints, 1):
        # Находим ближайший индекс по пройденному расстоянию
        k = np.searchsorted(r_odom, cp)
        k = min(k, len(r_odom) - 1)

        t_cur = t_odom[k] - t_start
        ro = r_odom[k]
        rg = r_gnss[k]
        diff = ro - rg

        # Погрешность относительно текущего пройденного расстояния
        rel_err_curr = (abs(diff) / rg * 100.0) if rg > 1e-2 else 0.0

        # Погрешность относительно ВСЕГО пути
        rel_err_total = (abs(diff) / final_rg * 100.0) if final_rg > 1e-2 else 0.0

        # Локальная погрешность на данном конкретном отрезке (между чекпоинтами)
        seg_ro = ro - prev_ro
        seg_rg = rg - prev_rg
        seg_diff = seg_ro - seg_rg
        seg_rel_err = (abs(seg_diff) / seg_rg * 100.0) if seg_rg > 1e-2 else 0.0

        vo_kmh = v_odom[k] * 3.6
        vg_kmh = v_gnss_interp[k] * 3.6

        sign = "+" if diff >= 0 else ""
        seg_sign = "+" if seg_diff >= 0 else ""
        print(
            f"{f'#{idx}':^9} | {t_cur:7.1f} с | {ro:9.1f}м | {rg:9.1f}м | "
            f"{f'{sign}{diff:.2f}м':^9} | {rel_err_curr:8.2f}% | {rel_err_total:9.3f}%  | "
            f"{f'{seg_sign}{seg_diff:.2f}м':^11} | {seg_rel_err:8.2f}% | "
            f"{vo_kmh:6.1f} | {vg_kmh:6.1f}"
        )
        prev_ro = ro
        prev_rg = rg

    print("-" * 135)

    # -------------------------------------------------------------------------
    # 2. СВОДНАЯ ТАБЛИЦА ИТОГОВЫХ МЕТРИК
    # -------------------------------------------------------------------------
    final_diff = final_ro - final_rg
    final_rel_err = (abs(final_diff) / final_rg * 100.0) if final_rg > 1e-2 else 0.0

    mae_dist = float(np.mean(abs_delta_r))
    rmse_dist = float(np.sqrt(np.mean(delta_r ** 2)))
    max_dist_err = float(np.max(abs_delta_r))

    max_rel_to_total = (max_dist_err / final_rg * 100.0) if final_rg > 1e-2 else 0.0
    mae_rel_to_total = (mae_dist / final_rg * 100.0) if final_rg > 1e-2 else 0.0

    vel_err = v_odom - v_gnss_interp
    mae_vel_ms = float(np.mean(np.abs(vel_err)))
    rmse_vel_ms = float(np.sqrt(np.mean(vel_err ** 2)))

    duration_sec = t_odom[-1] - t_odom[0]

    print("\n" + "=" * 70)
    print("                 СВОДНЫЕ МЕТРИКИ РЕШЕНИЯ")
    print("=" * 70)
    print(f" {'Показатель':<42} | {'Значение':>23}")
    print("-" * 70)
    print(f" {'Длительность прогона':<42} | {f'{duration_sec:.1f} с ({duration_sec/60:.2f} мин)':>23}")
    print(f" {'Пройденный путь R (Одометрия)':<42} | {f'{final_ro:.2f} м':>23}")
    print(f" {'Пройденный путь R (GNSS эталон)':<42} | {f'{final_rg:.2f} м':>23}")
    print(f" {'Финальная ошибка ΔR':<42} | {f'{final_diff:+.2f} м':>23}")
    print(f" {'Финальная погр. относительно ВСЕГО пути':<42} | {f'{final_rel_err:.3f} %':>23}")
    print(f" {'Максимальный дрейф относительно ВСЕГО пути':<42} | {f'{max_rel_to_total:.3f} % ({max_dist_err:.2f} м)':>23}")
    print(f" {'Средний дрейф относительно ВСЕГО пути':<42} | {f'{mae_rel_to_total:.3f} % ({mae_dist:.2f} м)':>23}")
    print(f" {'Ср.-квадратичная ошибка пути (RMSE)':<42} | {f'{rmse_dist:.2f} м':>23}")
    print(f" {'MAE по скорости':<42} | {f'{mae_vel_ms:.3f} м/с ({mae_vel_ms*3.6:.2f} км/ч)':>23}")
    print(f" {'RMSE по скорости':<42} | {f'{rmse_vel_ms:.3f} м/с ({rmse_vel_ms*3.6:.2f} км/ч)':>23}")
    print("=" * 70 + "\n")

    # Экспорт в CSV при необходимости
    if csv_path:
        import csv
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "checkpoint", "time_sec", "r_odom_m", "r_gnss_m", "delta_r_m",
                "rel_err_curr_pct", "rel_err_total_pct", "seg_delta_r_m", "seg_rel_err_pct",
                "v_odom_kmh", "v_gnss_kmh"
            ])
            prev_ro, prev_rg = 0.0, 0.0
            for idx, cp in enumerate(checkpoints, 1):
                k = min(np.searchsorted(r_odom, cp), len(r_odom) - 1)
                t_cur = t_odom[k] - t_start
                ro, rg = r_odom[k], r_gnss[k]
                diff = ro - rg
                rel_curr = (abs(diff) / rg * 100.0) if rg > 1e-2 else 0.0
                rel_tot = (abs(diff) / final_rg * 100.0) if final_rg > 1e-2 else 0.0
                seg_diff = (ro - prev_ro) - (rg - prev_rg)
                seg_rg = rg - prev_rg
                seg_rel = (abs(seg_diff) / seg_rg * 100.0) if seg_rg > 1e-2 else 0.0
                writer.writerow([
                    f"#{idx}", f"{t_cur:.2f}", f"{ro:.2f}", f"{rg:.2f}", f"{diff:.2f}",
                    f"{rel_curr:.3f}", f"{rel_tot:.3f}", f"{seg_diff:.2f}", f"{seg_rel:.3f}",
                    f"{v_odom[k]*3.6:.2f}", f"{v_gnss_interp[k]*3.6:.2f}"
                ])
                prev_ro, prev_rg = ro, rg
        print(f"📄 Результаты участков экспортированы в CSV: {csv_path}")

    # -------------------------------------------------------------------------
    # 3. ПОСТРОЕНИЕ И СОХРАНЕНИЕ ГРАФИКОВ
    # -------------------------------------------------------------------------
    if save_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        t_rel = t_odom - t_start

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        # График скоростей
        ax1.plot(t_rel, v_odom * 3.6, label="V_ср (Одометрия)", color="#1f77b4", linewidth=1.5)
        ax1.plot(t_rel, v_gnss_interp * 3.6, label="V (GNSS эталон)", color="#ff7f0e", linestyle="--", linewidth=1.2, alpha=0.85)
        ax1.set_ylabel("Скорость (км/ч)", fontsize=11)
        ax1.set_title(f"Валидация решения R = sum(V_cp * dt) на прогоне {bag_path.name}", fontsize=13, fontweight="bold")
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="upper right", frameon=True)

        # График расстояния
        ax2.plot(t_rel, r_odom, label="R одометрии (м)", color="#2ca02c", linewidth=1.5)
        ax2.plot(t_rel, r_gnss, label="R GNSS эталон (м)", color="#d62728", linestyle="--", linewidth=1.2)
        ax2.set_ylabel("Расстояние R (м)", fontsize=11)
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="upper left", frameon=True)

        # График погрешности
        ax3.plot(t_rel, delta_r, label="Ошибка ΔR = R_одом - R_gnss (м)", color="#9467bd", linewidth=1.5)
        ax3.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
        ax3.set_xlabel("Время от начала прогона (сек)", fontsize=11)
        ax3.set_ylabel("Погрешность ΔR (м)", fontsize=11)
        ax3.grid(True, linestyle=":", alpha=0.6)
        ax3.legend(loc="upper left", frameon=True)

        img_dir = ws_dir / "img"
        img_dir.mkdir(exist_ok=True)
        plot_file = img_dir / f"evaluation_{bag_path.name}.png"
        fig.savefig(plot_file, dpi=150)
        plt.close(fig)
        print(f"📊 График валидации сохранен в файл: {plot_file.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Проверка решения резервной одометрии беспилотного трамвая (чистый Python, без ROS)."
    )
    parser.add_argument(
        "bag_path",
        type=str,
        nargs="?",
        default="data/30618_0e41eac3",
        help="Путь к каталогу распакованного bag (по умолчанию data/30618_0e41eac3)"
    )
    parser.add_argument(
        "--step",
        type=float,
        default=500.0,
        help="Шаг участков в метрах для таблицы погрешностей (по умолчанию 500 м)"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Путь к файлу для сохранения таблицы в формате CSV"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Отключить генерацию графиков PNG"
    )

    args = parser.parse_args()

    ws_dir = _repo_dir
    bag_path = Path(args.bag_path)
    if not bag_path.is_absolute():
        bag_path = ws_dir / bag_path

    csv_path = Path(args.csv) if args.csv else None
    if csv_path and not csv_path.is_absolute():
        csv_path = ws_dir / csv_path

    evaluate_bag(
        bag_path=bag_path,
        ws_dir=ws_dir,
        step_dist_m=args.step,
        csv_path=csv_path,
        save_plot=not args.no_plot
    )


if __name__ == "__main__":
    main()

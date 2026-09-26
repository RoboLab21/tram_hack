#!/usr/bin/env python3
"""
Полный прогон последнего решения резервной одометрии (PathCalibratedDeadReckoningNode)
по всем багам с валидным GNSS (56 заездов: 24 на щук-талл, 32 на талл-щук).
"""

import sys
from pathlib import Path
_repo_dir = Path(__file__).resolve().parent.parent
for _p in [str(_repo_dir), str(_repo_dir / "solution"), str(_repo_dir / "scripts")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import csv
import numpy as np
from rosbags.highlevel import AnyReader
from batch_evaluate_filtered import setup_typestore
from solution.odometry_node_path_calibrated import (
    PathCalibratedDeadReckoningNode,
    load_path_geometry
)

ws_dir = _repo_dir
typestore = setup_typestore(ws_dir)

# Предварительно загружаем геометрию обоих маршрутов
path_files = {
    "щук-талл": ws_dir / "pathgrath" / "щукинская - таллинская.json",
    "талл-щук": ws_dir / "pathgrath" / "таллинская - щукинская.json",
}

geom_by_route = {
    r: load_path_geometry(p) for r, p in path_files.items()
}

target_topics = {
    "/vehicle/driver_position_cmd",
    "/vehicle/front_bogie_velocity",
    "/vehicle/rear_bogie_velocity",
    "/sensing/gnss/master/vel"
}


def process_bag(bag_path: Path, route_name: str):
    straights, s_map, c_map = geom_by_route[route_name]
    
    # Последнее решение со всеми фильтрами:
    # 1. Фильтр сбоев dV^2
    # 2. Калибровка износа колес по карте пути
    # 3. Фильтр торможения при driver_cmd < 0
    # 4. Компенсация коротких / крутых поворотов
    node = PathCalibratedDeadReckoningNode(
        straight_sections=straights,
        path_geometry=(s_map, c_map),
        enable_brake_filter=True,
        brake_crawl_thresh_kmh=0.8,
        brake_decel_limit_ms2=2.5,
        enable_curve_compensation=True,
        curve_thresh_curv=0.008,
        curve_beta=0.20,
        input_in_kmh=True,
        integration_method="trapezoidal"
    )
    
    gnss_records = []

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
                    node.update_front(t, msg.velocity)
                elif conn.topic == "/vehicle/rear_bogie_velocity":
                    node.update_rear(t, msg.velocity)
                elif conn.topic == "/sensing/gnss/master/vel":
                    vx = msg.twist.linear.x
                    vy = msg.twist.linear.y
                    v_gnss = float(np.hypot(vx, vy))
                    gnss_records.append((t, v_gnss))
    except Exception as e:
        print(f"Ошибка чтения {bag_path.name}: {e}")
        return None

    data = node.get_arrays()
    if not data or len(data["timestamp"]) == 0 or len(gnss_records) <= 10:
        return None

    t_o = data["timestamp"]
    r_est = data["distance"][-1]
    duration_sec = float(t_o[-1] - t_o[0])
    calib_report = node.get_calibration_report()
    vehicle = bag_path.name.split("_")[0]

    # Интерполяция GNSS на временную сетку одометрии
    tg = np.array([x[0] for x in gnss_records])
    vg = np.array([x[1] for x in gnss_records])
    s_idx = np.argsort(tg); tg, vg = tg[s_idx], vg[s_idx]
    
    v_gnss_i = np.interp(t_o, tg, vg)
    dt = np.diff(t_o, prepend=t_o[0]); dt[0] = 0.0
    r_gnss = np.cumsum(v_gnss_i * dt)[-1]
    tot_gnss = float(r_gnss)

    if tot_gnss < 100.0:  # стояночные
        return None

    err_m = float(r_est - tot_gnss)
    pct = abs(err_m) / tot_gnss * 100.0
    max_drift_m = float(np.max(np.abs(data["distance"] - np.cumsum(v_gnss_i * dt))))
    max_drift_pct = max_drift_m / tot_gnss * 100.0

    return {
        "bag_id": bag_path.name,
        "route": route_name,
        "vehicle": vehicle,
        "duration_sec": duration_sec,
        "r_gnss_m": tot_gnss,
        "r_est_m": float(r_est),
        "err_m": err_m,
        "abs_err_m": abs(err_m),
        "pct": pct,
        "max_drift_m": max_drift_m,
        "max_drift_pct": max_drift_pct,
        "k_scale": calib_report["final_k_scale"],
        "wear_pct": calib_report["wear_pct"],
        "is_calibrated": calib_report["is_calibrated"],
        "sections_count": calib_report["measured_sections_count"]
    }


def main():
    routes = ["щук-талл", "талл-щук"]
    results = []
    
    print("=" * 100)
    print("ПРОГОН ПОСЛЕДНЕГО РЕШЕНИЯ РЕЗЕРВНОЙ ОДОМЕТРИИ ПО ВСЕМ БАГАМ С ВАЛИДНЫМ GNSS")
    print("=" * 100)

    for route_name in routes:
        folder = ws_dir / "data" / route_name
        bags = sorted([d for d in folder.iterdir() if d.is_dir() and (d / "metadata.yaml").exists()])
        print(f"Маршрут {route_name.upper()}: найдено {len(bags)} заездов...")
        for b in bags:
            res = process_bag(b, route_name)
            if res:
                results.append(res)
                print(f"  [{len(results):2d}/56] {res['bag_id']}: GNSS={res['r_gnss_m']:6.1f}м, Одом={res['r_est_m']:6.1f}м, Ошибка={res['err_m']:+6.2f}м ({res['pct']:.3f}%), k_scale={res['k_scale']:.5f}")

    print(f"\nВсего успешно обработано динамических заездов: {len(results)}")

    # Сохранение полного CSV отчета
    reports_dir = ws_dir / "reports"
    reports_dir.mkdir(exist_ok=True)
    csv_file = reports_dir / "valid_gnss_all_bags_detailed.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "index", "bag_id", "route", "vehicle", "duration_sec",
            "r_gnss_m", "r_est_m", "err_m", "abs_err_m", "pct",
            "max_drift_m", "max_drift_pct", "k_scale", "wear_pct", "is_calibrated"
        ])
        for idx, r in enumerate(results, 1):
            writer.writerow([
                idx, r["bag_id"], r["route"], r["vehicle"], f"{r['duration_sec']:.1f}",
                f"{r['r_gnss_m']:.2f}", f"{r['r_est_m']:.2f}", f"{r['err_m']:+.2f}",
                f"{r['abs_err_m']:.2f}", f"{r['pct']:.3f}",
                f"{r['max_drift_m']:.2f}", f"{r['max_drift_pct']:.3f}",
                f"{r['k_scale']:.5f}", f"{r['wear_pct']:+.3f}", "YES" if r["is_calibrated"] else "NO"
            ])
    print(f"Детальный отчет сохранен в: {csv_file}")

    # Вывод полной таблицы
    print("\n" + "=" * 105)
    print("ПОЛНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ ПО КАЖДОМУ ЗАЕЗДУ С ВАЛИДНЫМ GNSS (N = 56)")
    print("=" * 105)
    print(f"{'№':<3} | {'Bag ID':<16} | {'Маршрут':<9} | {'Вагон':<6} | {'GNSS (м)':<9} | {'Одом (м)':<9} | {'ΔR (м)':<9} | {'Ошибка (%)':<10} | {'k_scale':<8}")
    print("-" * 105)
    for idx, r in enumerate(results, 1):
        print(f"{idx:<3} | {r['bag_id']:<16} | {r['route']:<9} | {r['vehicle']:<6} | {r['r_gnss_m']:<9.1f} | {r['r_est_m']:<9.1f} | {r['err_m']:<+9.2f} | {r['pct']:<10.3f} | {r['k_scale']:<8.5f}")

    # Сводные таблицы
    def print_stat(title, subset):
        n = len(subset)
        if n == 0: return
        errs = np.array([r["abs_err_m"] for r in subset])
        pcts = np.array([r["pct"] for r in subset])
        tot_km = sum(r["r_gnss_m"] for r in subset) / 1000.0
        
        print(f"\n--- {title} (N = {n}, Пробег = {tot_km:.1f} км) ---")
        print(f"{'Метрика':<20} | {'Факт (метры)':<16} | {'Процент (%)':<16}")
        print("-" * 58)
        print(f"{'Минимальная ошибка':<20} | {np.min(errs):<16.2f} | {np.min(pcts):<16.3f}")
        print(f"{'Средняя ошибка':<20} | {np.mean(errs):<16.2f} | {np.mean(pcts):<16.3f}")
        print(f"{'Медианная ошибка':<20} | {np.median(errs):<16.2f} | {np.median(pcts):<16.3f}")
        print(f"{'Максимальная ошибка':<20} | {np.max(errs):<16.2f} | {np.max(pcts):<16.3f}")
        print(f"{'Стд. отклонение':<20} | {np.std(errs):<16.2f} | {np.std(pcts):<16.3f}")

    print("\n" + "=" * 105)
    print("ИТОГОВАЯ СТАТИСТИКА ТОЧНОСТИ ПОСЛЕДНЕГО РЕШЕНИЯ (МИН / СРЕД / МЕДИАНА / МАКС)")
    print("=" * 105)

    print_stat("ОБЩИЙ ИТОГ (ВСЕ 56 ВАЛИДНЫХ ЗАЕЗДОВ GNSS)", results)
    print_stat("МАРШРУТ ЩУКИНСКАЯ - ТАЛЛИНСКАЯ (24 заезда)", [r for r in results if r["route"] == "щук-талл"])
    print_stat("МАРШРУТ ТАЛЛИНСКАЯ - ЩУКИНСКАЯ (32 заезда)", [r for r in results if r["route"] == "талл-щук"])
    print_stat("ВАГОН 30618 (51 заезд)", [r for r in results if r["vehicle"] == "30618"])
    print_stat("ВАГОН 30639 (5 заездов со сбоями)", [r for r in results if r["vehicle"] == "30639"])


if __name__ == "__main__":
    main()

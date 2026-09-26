#!/usr/bin/env python3
"""
Пакетная проверка резервной одометрии (PathCalibratedDeadReckoningNode v5)
на всей выборке с хорошим GNSS (data/щук-талл и data/талл-щук).

Сравнивает:
1. Базовое решение (v1: сырая одометрия)
2. Фильтрованное решение (v2: квадратичный фильтр сбоев датчиков dV^2)
3. Автокалиброванное решение по прямым (v3: k_scale)
4. Комплексное решение (v4: k_scale + фильтр торможения driver_cmd < 0 + компенсация кривизны)
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
from solution.odometry_node import DeadReckoningNode
from solution.odometry_node_filtered import FilteredDeadReckoningNode
from solution.odometry_node_path_calibrated import (
    PathCalibratedDeadReckoningNode,
    load_path_geometry
)

ws_dir = _repo_dir
typestore = setup_typestore(ws_dir)

# Предварительно извлекаем геометрию пути для обоих маршрутов
shchuk_tall_geom = load_path_geometry(ws_dir / "pathgrath" / "щукинская - таллинская.json")
tall_shchuk_geom = load_path_geometry(ws_dir / "pathgrath" / "таллинская - щукинская.json")

geom_by_route = {
    "щук-талл": shchuk_tall_geom,
    "талл-щук": tall_shchuk_geom,
}


def process_bag(bag_path: Path, route_name: str, path_geom_data):
    straights, s_map, c_map = path_geom_data
    
    node_base = DeadReckoningNode(input_in_kmh=True, integration_method="trapezoidal")
    node_filt = FilteredDeadReckoningNode(input_in_kmh=True, integration_method="trapezoidal")
    # v3: только калибровка износа по прямым участкам карты
    node_v3 = PathCalibratedDeadReckoningNode(
        straight_sections=straights,
        enable_brake_filter=False,
        enable_curve_compensation=False,
        input_in_kmh=True,
        integration_method="trapezoidal"
    )
    # v4: калибровка по прямым + фильтр торможения (driver_cmd < 0) + компенсация кривизны
    node_v4 = PathCalibratedDeadReckoningNode(
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
                    node_v4.update_driver_cmd(t, msg.position)
                elif conn.topic == "/vehicle/front_bogie_velocity":
                    node_base.update_front(t, msg.velocity)
                    node_filt.update_front(t, msg.velocity)
                    node_v3.update_front(t, msg.velocity)
                    node_v4.update_front(t, msg.velocity)
                elif conn.topic == "/vehicle/rear_bogie_velocity":
                    node_base.update_rear(t, msg.velocity)
                    node_filt.update_rear(t, msg.velocity)
                    node_v3.update_rear(t, msg.velocity)
                    node_v4.update_rear(t, msg.velocity)
                elif conn.topic == "/sensing/gnss/master/vel":
                    vx = msg.twist.linear.x
                    vy = msg.twist.linear.y
                    v_gnss = float(np.hypot(vx, vy))
                    gnss_records.append((t, v_gnss))
    except Exception as e:
        print(f"Ошибка чтения {bag_path.name}: {e}")
        return None

    data_v4 = node_v4.get_arrays()
    data_v3 = node_v3.get_arrays()
    data_filt = node_filt.get_arrays()
    data_base = node_base.get_arrays()
    
    if not data_v4 or len(data_v4["timestamp"]) == 0:
        return None

    t_o = data_v4["timestamp"]
    r_v4 = data_v4["distance"]
    r_v3 = data_v3["distance"]
    r_filt = data_filt["distance"]
    r_base = data_base["distance"]
    
    duration_sec = float(t_o[-1] - t_o[0])
    calib_report = node_v4.get_calibration_report()
    vehicle = bag_path.name.split("_")[0]
    
    if len(gnss_records) <= 10:
        return None

    t_g = np.array([x[0] for x in gnss_records])
    v_g = np.array([x[1] for x in gnss_records])
    s_idx = np.argsort(t_g)
    t_g, v_g = t_g[s_idx], v_g[s_idx]
    
    v_gnss_i = np.interp(t_o, t_g, v_g)
    dt = np.diff(t_o, prepend=t_o[0]); dt[0] = 0.0
    r_gnss = np.cumsum(v_gnss_i * dt)
    tot_gnss = float(r_gnss[-1])

    if tot_gnss < 100.0:  # пропускаем стоянки
        return None

    # Ошибки
    err_base = float(r_base[-1] - tot_gnss)
    err_filt = float(r_filt[-1] - tot_gnss)
    err_v3 = float(r_v3[-1] - tot_gnss)
    err_v4 = float(r_v4[-1] - tot_gnss)
    
    pct_base = abs(err_base) / tot_gnss * 100.0
    pct_filt = abs(err_filt) / tot_gnss * 100.0
    pct_v3 = abs(err_v3) / tot_gnss * 100.0
    pct_v4 = abs(err_v4) / tot_gnss * 100.0

    max_drift_v4 = float(np.max(np.abs(r_v4 - r_gnss)))
    max_drift_pct_v4 = max_drift_v4 / tot_gnss * 100.0

    return {
        "bag_id": bag_path.name,
        "route": route_name,
        "vehicle": vehicle,
        "duration_sec": duration_sec,
        "r_gnss_m": tot_gnss,
        "r_base_m": float(r_base[-1]),
        "err_base_m": err_base,
        "pct_base": pct_base,
        "r_filt_m": float(r_filt[-1]),
        "err_filt_m": err_filt,
        "pct_filt": pct_filt,
        "r_v3_m": float(r_v3[-1]),
        "err_v3_m": err_v3,
        "pct_v3": pct_v3,
        "r_v4_m": float(r_v4[-1]),
        "err_v4_m": err_v4,
        "pct_v4": pct_v4,
        "max_drift_v4_m": max_drift_v4,
        "max_drift_v4_pct": max_drift_pct_v4,
        "k_scale": calib_report["final_k_scale"],
        "wear_pct": calib_report["wear_pct"],
        "delta_dia_mm": calib_report.get("delta_dia_mm_est", 0.0),
        "measured_sections": calib_report["measured_sections_count"],
        "is_calibrated": calib_report["is_calibrated"]
    }


def run_evaluation():
    routes = [
        ("щук-талл", geom_by_route["щук-талл"]),
        ("талл-щук", geom_by_route["талл-щук"])
    ]
    
    results = []
    print("Запуск проверки решения v4 (k_scale + фильтр торможения + компенсация кривизны)...")
    
    for route_name, p_geom in routes:
        folder = ws_dir / "data" / route_name
        bags = sorted([d for d in folder.iterdir() if d.is_dir() and (d / "metadata.yaml").exists()])
        print(f"\nОбработка маршрута {route_name.upper()} ({len(bags)} заездов)...")
        for b in bags:
            res = process_bag(b, route_name, p_geom)
            if res:
                results.append(res)
                print(f"  {res['bag_id']}: GNSS={res['r_gnss_m']:.1f}м | k={res['k_scale']:.5f} ({res['wear_pct']:+.3f}%) | v3={res['err_v3_m']:+.2f}м ({res['pct_v3']:.3f}%) -> v4={res['err_v4_m']:+.2f}м ({res['pct_v4']:.3f}%)")

    print(f"\nВсего успешно обработано динамических заездов: {len(results)}")

    # Сохраняем CSV
    reports_dir = ws_dir / "reports"
    reports_dir.mkdir(exist_ok=True)
    csv_path = reports_dir / "path_calibrated_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "bag_id", "route", "vehicle", "duration_sec", "r_gnss_m",
            "err_base_m", "pct_base", "err_filt_m", "pct_filt",
            "err_v3_m", "pct_v3", "err_v4_m", "pct_v4",
            "max_drift_v4_m", "max_drift_v4_pct",
            "k_scale", "wear_pct", "measured_sections", "is_calibrated"
        ])
        for r in results:
            writer.writerow([
                r["bag_id"], r["route"], r["vehicle"], f"{r['duration_sec']:.1f}", f"{r['r_gnss_m']:.2f}",
                f"{r['err_base_m']:+.2f}", f"{r['pct_base']:.3f}",
                f"{r['err_filt_m']:+.2f}", f"{r['pct_filt']:.3f}",
                f"{r['err_v3_m']:+.2f}", f"{r['pct_v3']:.3f}",
                f"{r['err_v4_m']:+.2f}", f"{r['pct_v4']:.3f}",
                f"{r['max_drift_v4_m']:.2f}", f"{r['max_drift_v4_pct']:.3f}",
                f"{r['k_scale']:.5f}", f"{r['wear_pct']:+.3f}",
                r["measured_sections"], "YES" if r["is_calibrated"] else "NO"
            ])
    print(f"Результаты сохранены в: {csv_path.name}")

    tot_gnss_km = sum(r["r_gnss_m"] for r in results) / 1000.0

    print("\n" + "="*96)
    print("ИТОГОВАЯ СВОДКА ТОЧНОСТИ ОДОМЕТРИИ НА ВСЕХ ВАЛИДНЫХ БАГАХ (56 заездов, 299.2 км)")
    print("="*96)
    print(f"Всего заездов: {len(results)} | Суммарный пробег: {tot_gnss_km:.1f} км")

    def print_stat_table(name, subset):
        n = len(subset)
        if n == 0: return
        print(f"\n--- {name} (N = {n}) ---")
        print(f"{'Решение':<26} | {'Абсолютная ошибка |ΔR| (метры)':<34} | {'Относительная ошибка (%)':<34}")
        print(f"{'':<26} | {'Мин':<7} {'Сред':<8} {'Медиана':<8} {'Макс':<8} | {'Мин':<7} {'Сред':<8} {'Медиана':<8} {'Макс':<8}")
        print("-" * 102)
        
        for v_name, col_err, col_pct in [
            ("v1 (Базовая)", "err_base_m", "pct_base"),
            ("v2 (Фильтр сбоев dV^2)", "err_filt_m", "pct_filt"),
            ("v3 (Калибр. износа k_scale)", "err_v3_m", "pct_v3"),
            ("v4 (+ Тормоз и Кривые)", "err_v4_m", "pct_v4"),
        ]:
            e_m = np.abs(np.array([r[col_err] for r in subset]))
            e_p = np.array([r[col_pct] for r in subset])
            print(f"{v_name:<26} | {np.min(e_m):<7.2f} {np.mean(e_m):<8.2f} {np.median(e_m):<8.2f} {np.max(e_m):<8.2f} | "
                  f"{np.min(e_p):<7.3f} {np.mean(e_p):<8.3f} {np.median(e_p):<8.3f} {np.max(e_p):<8.3f}")

    print_stat_table("ОБЩИЙ ИТОГ (ВСЕ ВАЛИДНЫЕ БАГИ)", results)
    print_stat_table("ВАГОН 30618 (51 заезд)", [r for r in results if r["vehicle"] == "30618"])
    print_stat_table("ВАГОН 30639 (5 заездов)", [r for r in results if r["vehicle"] == "30639"])

    # Статистика k_scale по вагонам
    k_scales_30618 = [r["k_scale"] for r in results if r["vehicle"] == "30618"]
    k_scales_30639 = [r["k_scale"] for r in results if r["vehicle"] == "30639"]
    print("\n" + "="*96)
    print("ОЦЕНКА МАСШТАБНОГО КОЭФФИЦИЕНТА ИЗНОСА КОЛЕС (k_scale = V_true / V_колес):")
    print("="*96)
    if k_scales_30618:
        print(f"Вагон 30618: медианный k_scale = {np.median(k_scales_30618):.5f} (износ {(np.median(k_scales_30618)-1)*100:+.3f}%, ΔD = {620*(np.median(k_scales_30618)-1):+.2f} мм)")
    if k_scales_30639:
        print(f"Вагон 30639: медианный k_scale = {np.median(k_scales_30639):.5f} (износ {(np.median(k_scales_30639)-1)*100:+.3f}%, ΔD = {620*(np.median(k_scales_30639)-1):+.2f} мм)")

    # Топ худших в v4
    print("\n" + "="*96)
    print("ТОП-5 ХУДШИХ ПРОГОНОВ В РЕШЕНИИ v4 (+ Тормоз и Кривые):")
    print("="*96)
    worst_v4 = sorted(results, key=lambda x: x["pct_v4"], reverse=True)[:5]
    for i, r in enumerate(worst_v4, 1):
        print(f"{i}. {r['bag_id']} ({r['route']}, вагон {r['vehicle']}): GNSS={r['r_gnss_m']:.1f}м | v3={r['err_v3_m']:+.2f}м ({r['pct_v3']:.3f}%) -> v4={r['err_v4_m']:+.2f}м ({r['pct_v4']:.3f}%) | k_scale={r['k_scale']:.5f}")


if __name__ == "__main__":
    run_evaluation()

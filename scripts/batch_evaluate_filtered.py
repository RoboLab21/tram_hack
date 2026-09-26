import sys
from pathlib import Path
_repo_dir = Path(__file__).resolve().parent.parent
for _p in [str(_repo_dir), str(_repo_dir / "solution"), str(_repo_dir / "scripts")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

#!/usr/bin/env python3
"""
Пакетный прогон фильтрованной резервной одометрии (v2) по всем bag-файлам датасета.

Оценивает:
- Ошибку интегрального пути R vs эталон GNSS
- Статистику: min, max, mean, median (в % и в метрах)
- Детектирование сбоев датчиков (выпадения в ноль, скачки скорости)
- Сравнение с базовым решением (v1)
- Выявление худших прогонов
"""

import sys
import os
import argparse
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

from odometry_node import DeadReckoningNode
from odometry_node_filtered import FilteredDeadReckoningNode


def setup_typestore(ws_dir: Path):
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


def process_single_bag(bag_path: Path, typestore, step_dist_m: float = 500.0) -> Optional[Dict[str, Any]]:
    node_base = DeadReckoningNode(input_in_kmh=True, integration_method="trapezoidal")
    node_filt = FilteredDeadReckoningNode(input_in_kmh=True, integration_method="trapezoidal")
    gnss_records = []

    target_topics = {
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

                if conn.topic == "/vehicle/front_bogie_velocity":
                    node_base.update_front(t, msg.velocity)
                    node_filt.update_front(t, msg.velocity)
                elif conn.topic == "/vehicle/rear_bogie_velocity":
                    node_base.update_rear(t, msg.velocity)
                    node_filt.update_rear(t, msg.velocity)
                elif conn.topic == "/sensing/gnss/master/vel":
                    vx = msg.twist.linear.x
                    vy = msg.twist.linear.y
                    v_gnss = float(np.hypot(vx, vy))
                    gnss_records.append((t, v_gnss))
    except Exception as e:
        print(f"Ошибка чтения {bag_path.name}: {e}")
        return None

    filt_data = node_filt.get_arrays()
    base_data = node_base.get_arrays()
    if not filt_data or len(filt_data["timestamp"]) == 0:
        return None

    t_odom = filt_data["timestamp"]
    v_filt = filt_data["v_est"]
    r_filt = filt_data["distance"]
    r_base = base_data["distance"]
    v_base = base_data["v_avg"]

    duration_sec = float(t_odom[-1] - t_odom[0])
    total_r_filt = float(r_filt[-1])
    total_r_base = float(r_base[-1])
    vehicle_id = bag_path.name.split("_")[0]

    # Подсчет аномалий
    fault_counts = {
        "ZERO_V1": 0,
        "ZERO_V2": 0,
        "SPIKE_V1": 0,
        "SPIKE_V2": 0,
        "EXCESSIVE_DIFF": 0
    }
    for s in node_filt.history:
        if s.fault_type in fault_counts:
            fault_counts[s.fault_type] += 1

    total_faults = sum(fault_counts.values())

    has_gnss = len(gnss_records) > 10

    res = {
        "bag_id": bag_path.name,
        "vehicle": vehicle_id,
        "duration_sec": duration_sec,
        "total_r_base": total_r_base,
        "total_r_filt": total_r_filt,
        "faults": fault_counts,
        "total_faults": total_faults,
        "has_gnss": has_gnss,
        "total_r_gnss": None,
        "base_final_diff_m": None,
        "base_final_rel_err_pct": None,
        "base_max_err_m": None,
        "base_max_rel_to_total_pct": None,
        "filt_final_diff_m": None,
        "filt_final_abs_diff_m": None,
        "filt_final_rel_err_pct": None,
        "filt_max_err_m": None,
        "filt_max_rel_to_total_pct": None,
        "filt_mae_dist_m": None,
        "filt_rmse_dist_m": None,
        "filt_mae_vel_kmh": None,
        "checkpoints": []
    }

    if not has_gnss:
        return res

    t_gnss_raw = np.array([x[0] for x in gnss_records])
    v_gnss_raw = np.array([x[1] for x in gnss_records])
    sort_idx = np.argsort(t_gnss_raw)
    t_gnss_raw = t_gnss_raw[sort_idx]
    v_gnss_raw = v_gnss_raw[sort_idx]

    # Интерполяция и расчет пути GNSS
    v_gnss_interp = np.interp(t_odom, t_gnss_raw, v_gnss_raw)
    dt_odom = np.diff(t_odom, prepend=t_odom[0])
    dt_odom[0] = 0.0
    r_gnss = np.cumsum(v_gnss_interp * dt_odom)

    total_rg = float(r_gnss[-1])

    # Ошибки базовой ноды
    base_delta_r = r_base - r_gnss
    base_abs_delta_r = np.abs(base_delta_r)
    base_final_diff = total_r_base - total_rg
    base_final_rel_err = (abs(base_final_diff) / total_rg * 100.0) if total_rg > 1e-2 else 0.0
    base_max_err_m = float(np.max(base_abs_delta_r))
    base_max_rel_total = (base_max_err_m / total_rg * 100.0) if total_rg > 1e-2 else 0.0

    # Ошибки фильтрованной ноды
    filt_delta_r = r_filt - r_gnss
    filt_abs_delta_r = np.abs(filt_delta_r)
    filt_final_diff = total_r_filt - total_rg
    filt_final_abs_diff = abs(filt_final_diff)
    filt_final_rel_err = (filt_final_abs_diff / total_rg * 100.0) if total_rg > 1e-2 else 0.0
    filt_max_err_m = float(np.max(filt_abs_delta_r))
    filt_max_rel_total = (filt_max_err_m / total_rg * 100.0) if total_rg > 1e-2 else 0.0

    filt_mae_dist = float(np.mean(filt_abs_delta_r))
    filt_rmse_dist = float(np.sqrt(np.mean(filt_delta_r ** 2)))

    filt_vel_err = np.abs(v_filt - v_gnss_interp)
    filt_mae_vel_kmh = float(np.mean(filt_vel_err) * 3.6)

    res.update({
        "total_r_gnss": total_rg,
        "base_final_diff_m": base_final_diff,
        "base_final_rel_err_pct": base_final_rel_err,
        "base_max_err_m": base_max_err_m,
        "base_max_rel_to_total_pct": base_max_rel_total,
        "filt_final_diff_m": filt_final_diff,
        "filt_final_abs_diff_m": filt_final_abs_diff,
        "filt_final_rel_err_pct": filt_final_rel_err,
        "filt_max_err_m": filt_max_err_m,
        "filt_max_rel_to_total_pct": filt_max_rel_total,
        "filt_mae_dist_m": filt_mae_dist,
        "filt_rmse_dist_m": filt_rmse_dist,
        "filt_mae_vel_kmh": filt_mae_vel_kmh
    })

    # Чекпоинты
    actual_step = step_dist_m if total_r_filt >= step_dist_m else max(50.0, total_r_filt / 4.0)
    checkpoints = np.arange(actual_step, total_r_filt, actual_step)
    if len(checkpoints) == 0 or checkpoints[-1] < total_r_filt - (actual_step * 0.2):
        checkpoints = np.append(checkpoints, total_r_filt)

    t_start = t_odom[0]
    prev_ro, prev_rg = 0.0, 0.0

    for idx, cp in enumerate(checkpoints, 1):
        k = min(np.searchsorted(r_filt, cp), len(r_filt) - 1)
        t_cur = float(t_odom[k] - t_start)
        ro = float(r_filt[k])
        rg = float(r_gnss[k])
        diff = ro - rg

        rel_curr = (abs(diff) / rg * 100.0) if rg > 1e-2 else 0.0
        rel_tot = (abs(diff) / total_rg * 100.0) if total_rg > 1e-2 else 0.0
        seg_diff = (ro - prev_ro) - (rg - prev_rg)
        seg_rg = rg - prev_rg
        seg_rel = (abs(seg_diff) / seg_rg * 100.0) if seg_rg > 1e-2 else 0.0

        res["checkpoints"].append({
            "num": idx,
            "time_sec": t_cur,
            "r_filt": ro,
            "r_gnss": rg,
            "diff_m": diff,
            "rel_curr_pct": rel_curr,
            "rel_total_pct": rel_tot,
            "seg_diff_m": seg_diff,
            "seg_rel_pct": seg_rel,
            "v_filt_kmh": float(v_filt[k] * 3.6),
            "v_gnss_kmh": float(v_gnss_interp[k] * 3.6)
        })
        prev_ro, prev_rg = ro, rg

    return res


def run_batch(data_dir: Path, ws_dir: Path):
    typestore = setup_typestore(ws_dir)
    bag_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and (d / "metadata.yaml").exists()])
    total_bags = len(bag_dirs)
    print(f"\nНайдено {total_bags} bag-директорий в {data_dir}. Запуск фильтрованной пакетной обработки...\n")

    results: List[Dict[str, Any]] = []
    for idx, bpath in enumerate(bag_dirs, 1):
        if idx % 10 == 0 or idx == total_bags:
            print(f"[{idx:3d}/{total_bags}] Обработка {bpath.name}...")
        res = process_single_bag(bpath, typestore)
        if res:
            results.append(res)

    print(f"\nУспешно обработано: {len(results)} из {total_bags} прогонов.\n")

    # CSV экспорт
    reports_dir = ws_dir / "reports"
    reports_dir.mkdir(exist_ok=True)
    csv_path = reports_dir / "all_bags_filtered_summary.csv"
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "bag_id", "vehicle", "duration_sec", "has_gnss",
            "r_filt_m", "r_gnss_m", "filt_diff_m", "filt_abs_diff_m", "filt_rel_err_pct",
            "filt_max_err_m", "filt_max_rel_total_pct", "filt_mae_dist_m", "filt_rmse_dist_m", "filt_mae_vel_kmh",
            "r_base_m", "base_diff_m", "base_rel_err_pct", "base_max_err_m", "base_max_rel_total_pct",
            "zero_v1_count", "zero_v2_count", "spike_v1_count", "spike_v2_count", "excessive_diff_count", "total_faults"
        ])
        for r in results:
            writer.writerow([
                r["bag_id"],
                r["vehicle"],
                f"{r['duration_sec']:.1f}",
                "YES" if r["has_gnss"] else "NO",
                f"{r['total_r_filt']:.2f}",
                f"{r['total_r_gnss']:.2f}" if r["has_gnss"] else "",
                f"{r['filt_final_diff_m']:+.2f}" if r["has_gnss"] else "",
                f"{r['filt_final_abs_diff_m']:.2f}" if r["has_gnss"] else "",
                f"{r['filt_final_rel_err_pct']:.3f}" if r["has_gnss"] else "",
                f"{r['filt_max_err_m']:.2f}" if r["has_gnss"] else "",
                f"{r['filt_max_rel_to_total_pct']:.3f}" if r["has_gnss"] else "",
                f"{r['filt_mae_dist_m']:.2f}" if r["has_gnss"] else "",
                f"{r['filt_rmse_dist_m']:.2f}" if r["has_gnss"] else "",
                f"{r['filt_mae_vel_kmh']:.2f}" if r["has_gnss"] else "",
                f"{r['total_r_base']:.2f}",
                f"{r['base_final_diff_m']:+.2f}" if r["has_gnss"] else "",
                f"{r['base_final_rel_err_pct']:.3f}" if r["has_gnss"] else "",
                f"{r['base_max_err_m']:.2f}" if r["has_gnss"] else "",
                f"{r['base_max_rel_to_total_pct']:.3f}" if r["has_gnss"] else "",
                r["faults"]["ZERO_V1"],
                r["faults"]["ZERO_V2"],
                r["faults"]["SPIKE_V1"],
                r["faults"]["SPIKE_V2"],
                r["faults"]["EXCESSIVE_DIFF"],
                r["total_faults"]
            ])
    print(f"Сводка сохранена в: {csv_path.name}")

    # Вычисляем агрегированную статистику
    gnss_res = [r for r in results if r["has_gnss"] and r["total_r_gnss"] > 10.0]
    print(f"Прогонов с GNSS: {len(gnss_res)}")

    filt_err_pct = np.array([r["filt_final_rel_err_pct"] for r in gnss_res])
    filt_diff_abs_m = np.array([r["filt_final_abs_diff_m"] for r in gnss_res])
    filt_diff_m = np.array([r["filt_final_diff_m"] for r in gnss_res])
    filt_max_drift_m = np.array([r["filt_max_err_m"] for r in gnss_res])
    filt_max_drift_pct = np.array([r["filt_max_rel_to_total_pct"] for r in gnss_res])
    filt_mae_m = np.array([r["filt_mae_dist_m"] for r in gnss_res])

    base_err_pct = np.array([r["base_final_rel_err_pct"] for r in gnss_res])
    base_diff_abs_m = np.array([abs(r["base_final_diff_m"]) for r in gnss_res])
    base_max_drift_m = np.array([r["base_max_err_m"] for r in gnss_res])
    base_max_drift_pct = np.array([r["base_max_rel_to_total_pct"] for r in gnss_res])

    tot_gnss_km = sum(r["total_r_gnss"] for r in gnss_res) / 1000.0
    tot_filt_km = sum(r["total_r_filt"] for r in gnss_res) / 1000.0
    tot_base_km = sum(r["total_r_base"] for r in gnss_res) / 1000.0

    print("\n" + "="*80)
    print("ИТОГОВАЯ СТАТИСТИКА: ФИЛЬТРОВАННАЯ НОДА vs БАЗОВАЯ")
    print("="*80)
    print(f"Дистанция GNSS: {tot_gnss_km:.2f} км | Одом фильтр: {tot_filt_km:.2f} км | Одом база: {tot_base_km:.2f} км")
    print(f"\n1. Финальная погрешность пути (%):")
    print(f"   ФИЛЬТР: Среднее = {np.mean(filt_err_pct):.3f}%, Медиана = {np.median(filt_err_pct):.3f}%, Мин = {np.min(filt_err_pct):.3f}%, Макс = {np.max(filt_err_pct):.3f}%")
    print(f"   БАЗА:   Среднее = {np.mean(base_err_pct):.3f}%, Медиана = {np.median(base_err_pct):.3f}%, Мин = {np.min(base_err_pct):.3f}%, Макс = {np.max(base_err_pct):.3f}%")

    print(f"\n2. Финальная погрешность пути в метрах (|ΔR|):")
    print(f"   ФИЛЬТР: Среднее = {np.mean(filt_diff_abs_m):.2f} м, Медиана = {np.median(filt_diff_abs_m):.2f} м, Мин = {np.min(filt_diff_abs_m):.2f} м, Макс = {np.max(filt_diff_abs_m):.2f} м")
    print(f"   БАЗА:   Среднее = {np.mean(base_diff_abs_m):.2f} м, Медиана = {np.median(base_diff_abs_m):.2f} м, Мин = {np.min(base_diff_abs_m):.2f} м, Макс = {np.max(base_diff_abs_m):.2f} м")

    print(f"\n3. Максимальный дрейф пути вдоль маршрута (% от общего пути):")
    print(f"   ФИЛЬТР: Среднее = {np.mean(filt_max_drift_pct):.3f}%, Медиана = {np.median(filt_max_drift_pct):.3f}%, Мин = {np.min(filt_max_drift_pct):.3f}%, Макс = {np.max(filt_max_drift_pct):.3f}%")
    print(f"   БАЗА:   Среднее = {np.mean(base_max_drift_pct):.3f}%, Медиана = {np.median(base_max_drift_pct):.3f}%, Мин = {np.min(base_max_drift_pct):.3f}%, Макс = {np.max(base_max_drift_pct):.3f}%")

    print(f"\n4. Максимальный дрейф пути в метрах (Max Drift m):")
    print(f"   ФИЛЬТР: Среднее = {np.mean(filt_max_drift_m):.2f} м, Медиана = {np.median(filt_max_drift_m):.2f} м, Мин = {np.min(filt_max_drift_m):.2f} м, Макс = {np.max(filt_max_drift_m):.2f} м")
    print(f"   БАЗА:   Среднее = {np.mean(base_max_drift_m):.2f} м, Медиана = {np.median(base_max_drift_m):.2f} м, Мин = {np.min(base_max_drift_m):.2f} м, Макс = {np.max(base_max_drift_m):.2f} м")

    print(f"\n5. MAE по пути:")
    print(f"   ФИЛЬТР: Среднее MAE = {np.mean(filt_mae_m):.2f} м")

    # Сортировка по худшим прогонам в фильтрованном решении
    print("\n" + "="*80)
    print("ТОП ХУДШИХ ПРОГОНОВ ПО ФИНАЛЬНОЙ ПОГРЕШНОСТИ (%):")
    print("="*80)
    worst_by_pct = sorted(gnss_res, key=lambda x: x["filt_final_rel_err_pct"], reverse=True)[:10]
    for i, r in enumerate(worst_by_pct, 1):
        print(f"{i:2d}. {r['bag_id']} (Трамвай {r['vehicle']}): GNSS={r['total_r_gnss']:.1f}м, Одом={r['total_r_filt']:.1f}м, Ошибка={r['filt_final_diff_m']:+.2f}м ({r['filt_final_rel_err_pct']:.3f}%), МаксДрейф={r['filt_max_err_m']:.2f}м, Сбоев={r['total_faults']} (База была: {r['base_final_diff_m']:+.2f}м, {r['base_final_rel_err_pct']:.3f}%)")

    print("\n" + "="*80)
    print("ТОП ХУДШИХ ПРОГОНОВ ПО АБСОЛЮТНОЙ ПОГРЕШНОСТИ (МЕТРЫ):")
    print("="*80)
    worst_by_m = sorted(gnss_res, key=lambda x: x["filt_final_abs_diff_m"], reverse=True)[:10]
    for i, r in enumerate(worst_by_m, 1):
        print(f"{i:2d}. {r['bag_id']} (Трамвай {r['vehicle']}): GNSS={r['total_r_gnss']:.1f}м, Ошибка={r['filt_final_diff_m']:+.2f}м ({r['filt_final_rel_err_pct']:.3f}%), МаксДрейф={r['filt_max_err_m']:.2f}м (База была: {r['base_final_diff_m']:+.2f}м)")

    print("\n" + "="*80)
    print("ТОП ХУДШИХ ПРОГОНОВ ПО МАКСИМАЛЬНОМУ ДРЕЙФУ (МЕТРЫ):")
    print("="*80)
    worst_by_drift = sorted(gnss_res, key=lambda x: x["filt_max_err_m"], reverse=True)[:10]
    for i, r in enumerate(worst_by_drift, 1):
        print(f"{i:2d}. {r['bag_id']} (Трамвай {r['vehicle']}): GNSS={r['total_r_gnss']:.1f}м, МаксДрейф={r['filt_max_err_m']:.2f}м ({r['filt_max_rel_to_total_pct']:.3f}%), Финал={r['filt_final_diff_m']:+.2f}м (База была: {r['base_max_err_m']:.2f}м)")

    return results


def main():
    ws_dir = _repo_dir
    data_dir = ws_dir / "data"
    run_batch(data_dir=data_dir, ws_dir=ws_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Детальный анализ общего износа колес (Overall Wheel Wear / Absolute Wheel Scale)
на всей выборке данных с чистым GNSS.

Определяет:
1. Истинный масштаб колес k_true = V_true / V_wheels (по крейсерской скорости V > 20 км/ч).
2. Физический эквивалент износа: отклонение радиуса/диаметра колеса в мм:
   номинальный диаметр колеса трамвая ~ 620-710 мм (стандарт 620 мм для низкопольных трамваев 71-931М Витязь-М / Львенок).
3. Сравнение между вагонами 30618 и 30639.
4. Временную стабильность (изменяется ли масштаб от пробега к пробегу или постоянен для вагона).
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

ws_dir = _repo_dir
typestore = setup_typestore(ws_dir)

results = []

for route_dir, route_name in [("data/щук-талл", "щук-талл"), ("data/талл-щук", "талл-щук")]:
    dir_path = ws_dir / route_dir
    if not dir_path.exists():
        continue
    for bag_path in sorted(dir_path.iterdir()):
        if not bag_path.is_dir():
            continue
        vehicle = bag_path.name.split("_")[0]
        
        try:
            with AnyReader([bag_path], default_typestore=typestore) as reader:
                v1_list, v2_list, gnss_list = [], [], []
                target = {"/vehicle/front_bogie_velocity", "/vehicle/rear_bogie_velocity", "/sensing/gnss/master/vel"}
                conns = [c for c in reader.connections if c.topic in target]
                for conn, ts, raw in reader.messages(connections=conns):
                    msg = reader.deserialize(raw, conn.msgtype)
                    t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                    if conn.topic == "/vehicle/front_bogie_velocity":
                        v1_list.append((t, msg.velocity / 3.6))
                    elif conn.topic == "/vehicle/rear_bogie_velocity":
                        v2_list.append((t, msg.velocity / 3.6))
                    elif conn.topic == "/sensing/gnss/master/vel":
                        v = np.hypot(msg.twist.linear.x, msg.twist.linear.y)
                        gnss_list.append((t, v))
        except Exception as e:
            continue
            
        if len(v1_list) < 100 or len(gnss_list) < 100:
            continue
            
        t1 = np.array([x[0] for x in v1_list])
        v1 = np.array([x[1] for x in v1_list])
        t2 = np.array([x[0] for x in v2_list])
        v2 = np.array([x[1] for x in v2_list])
        tg = np.array([x[0] for x in gnss_list])
        vg = np.array([x[1] for x in gnss_list])
        
        if t1[0] < 1e8 or tg[0] < 1e8:
            continue
            
        t_start = max(t1[0], tg[0])
        t_end = min(t1[-1], tg[-1])
        if t_end - t_start < 100:
            continue
            
        t_grid = np.linspace(t_start, t_end, int((t_end - t_start) * 10))
        dt = np.diff(t_grid, prepend=t_grid[0]); dt[0] = 0.0
        
        v1_i = np.interp(t_grid, t1, v1)
        v2_i = np.interp(t_grid, t2, v2)
        vg_i = np.interp(t_grid, tg, vg)
        
        vw_i = (v1_i + v2_i) / 2.0
        
        d_wheels = np.sum(vw_i * dt)
        d_gnss = np.sum(vg_i * dt)
        
        if d_gnss < 1000:
            continue
            
        # Оценка масштаба на установившихся скоростях движения (v > 20 км/ч = 5.55 м/с)
        cruise = (vw_i > 5.55) & (vg_i > 5.55)
        if np.sum(cruise) > 50:
            k_cruise = vg_i[cruise] / vw_i[cruise]
            scale_median = float(np.median(k_cruise))
            scale_mean = float(np.mean(k_cruise))
            scale_std = float(np.std(k_cruise))
        else:
            scale_median = d_gnss / d_wheels
            scale_mean = scale_median
            scale_std = 0.0
            
        # Номинальный диаметр бандажа колеса трамвая (например, 620 мм)
        # delta_diameter_mm = D_nom * (k_scale - 1.0)
        d_nom_mm = 620.0
        delta_dia_mm = d_nom_mm * (scale_median - 1.0)
        
        results.append({
            "bag": bag_path.name,
            "route": route_name,
            "vehicle": vehicle,
            "duration_sec": t_end - t_start,
            "d_wheels": d_wheels,
            "d_gnss": d_gnss,
            "scale_median": scale_median,
            "wear_pct": (scale_median - 1.0) * 100.0,
            "delta_dia_mm": delta_dia_mm,
            "scale_std": scale_std,
            "dist_err_m": d_wheels - d_gnss,
            "dist_err_pct": (d_wheels - d_gnss) / d_gnss * 100.0
        })

out_csv = ws_dir / "reports" / "overall_wheel_wear_summary.csv"
if results:
    keys = list(results[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
print(f"Обработано {len(results)} заездов. Отчет сохранен в {out_csv}")

print("\n" + "="*80)
print("АНАЛИЗ ОБЩЕГО ИЗНОСА КОЛЕС (OVERALL WHEEL WEAR)")
print("="*80)

vehicles = sorted(list(set(r["vehicle"] for r in results)))
for veh in vehicles:
    grp = [r for r in results if r["vehicle"] == veh]
    scales = np.array([r["scale_median"] for r in grp])
    wears = np.array([r["wear_pct"] for r in grp])
    dias = np.array([r["delta_dia_mm"] for r in grp])
    errs = np.array([r["dist_err_pct"] for r in grp])
    errs_m = np.array([r["dist_err_m"] for r in grp])
    med_scale = float(np.median(scales))
    med_wear = float(np.median(wears))
    med_dia = float(np.median(dias))
    print(f"\nВагон {veh} ({len(grp)} заездов):")
    print(f"  Медианный масштаб k_scale (V_true / V_колес): {med_scale:.5f}")
    print(f"  Общий износ / отклонение диаметра колес:       {med_wear:+.3f}% (разброс: {np.min(wears):+.3f}% .. {np.max(wears):+.3f}%)")
    print(f"  Физическое изменение диаметра (при D_ном=620мм): {med_dia:+.2f} мм (разброс: {np.min(dias):+.2f} .. {np.max(dias):+.2f} мм)")
    print(f"  Систематическая ошибка одометрии (без корр.):    {float(np.median(errs)):+.3f}% ({float(np.median(errs_m)):+.1f} м на поездку)")
    print(f"  Разброс масштаба между рейсами (std):            {float(np.std(scales)):.5f} (CV = {float(np.std(scales))/med_scale*100:.3f}%)")

print("\n" + "="*80)
print("ВЫВОДЫ:")
print("="*80)

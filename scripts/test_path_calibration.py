#!/usr/bin/env python3
"""
Скрипт проверки решения v3 с автокалибровкой износа колес на первых трех прямых участках пути:
1. Автоматическое определение первых 3 прямых участков на графе пути (pathgrath).
2. Запуск ноды PathCalibratedDeadReckoningNode на реальных заездах.
3. Сбор выборок v1/v2 на прямых и вычисление коэффициента износа k_rel.
4. Вывод отчета по каждому участку.
"""

import sys
from pathlib import Path
_repo_dir = Path(__file__).resolve().parent.parent
for _p in [str(_repo_dir), str(_repo_dir / "solution"), str(_repo_dir / "scripts")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from rosbags.highlevel import AnyReader
from batch_evaluate_filtered import setup_typestore
from solution.odometry_node_path_calibrated import (
    PathCalibratedDeadReckoningNode,
    extract_straight_sections
)

ws_dir = _repo_dir
typestore = setup_typestore(ws_dir)


def test_route(route_name, path_filename, bag_name):
    path_file = ws_dir / "pathgrath" / path_filename
    bag_path = ws_dir / "data" / route_name / bag_name
    
    print("=" * 80)
    print(f"МАРШРУТ: {route_name.upper()} ({path_filename})")
    print(f"Заезд: {bag_name}")
    print("=" * 80)
    
    # 1. Автоматическое определение прямых участков
    straights = extract_straight_sections(path_file, max_curvature=0.0005, min_length_m=40.0, top_n=3)
    print(f"\n1. Автоматически найдены первые {len(straights)} прямых участка пути (|кривизна| < 0.0005 1/м, R > 2000 м):")
    for s in straights:
        print(f"   Прямой участок #{s.idx}: координаты пути s = [{s.s_start:6.1f} .. {s.s_end:6.1f}] м, длина = {s.length:5.1f} м")
        
    # 2. Создание ноды с привязкой к прямым
    node = PathCalibratedDeadReckoningNode(straight_sections=straights, input_in_kmh=True)
    gnss_records = []
    
    with AnyReader([bag_path], default_typestore=typestore) as reader:
        conns = [c for c in reader.connections if c.topic in {
            "/vehicle/front_bogie_velocity", "/vehicle/rear_bogie_velocity", "/sensing/gnss/master/vel"
        }]
        for conn, timestamp, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            if t <= 0: t = timestamp * 1e-9
            if conn.topic == "/vehicle/front_bogie_velocity":
                node.update_front(t, msg.velocity)
            elif conn.topic == "/vehicle/rear_bogie_velocity":
                node.update_rear(t, msg.velocity)
            elif conn.topic == "/sensing/gnss/master/vel":
                v = float(np.hypot(msg.twist.linear.x, msg.twist.linear.y))
                gnss_records.append((t, v))
                
    odata = node.get_arrays()
    report = node.get_calibration_report()
    
    print("\n2. Результаты автокалибровки износа колес на прямых участках:")
    print(f"   - Статус калибровки: {'УСПЕШНО' if report['is_calibrated'] else 'НЕ ЗАВЕРШЕНО'}")
    print(f"   - Всего выборок на прямых: {report['total_samples']}")
    print(f"   - Относительный коэффициент износа k_rel (v1 / v2): {report['final_k_rel']:.5f}")
    print(f"   - Разница скоростей тележек: {report['wear_diff_pct']:+.3f}%")
    
    print("\n   Детализация по участкам:")
    for key, val in report["straight_sections"].items():
        print(f"     * {key}: s = {val['s_range_m'][0]:.0f}..{val['s_range_m'][1]:.0f}м | выборок: {val['samples_count']:3d} | k_rel = {val['k_rel_median']:.5f} ({val['wear_diff_pct']:+.3f}%)")
        
    # Сравнение пути с GNSS
    t_o = odata["timestamp"]
    r_odom = odata["distance"][-1]
    
    if gnss_records:
        t_g = np.array([x[0] for x in gnss_records])
        v_g = np.array([x[1] for x in gnss_records])
        s_idx = np.argsort(t_g)
        t_g, v_g = t_g[s_idx], v_g[s_idx]
        v_g_i = np.interp(t_o, t_g, v_g)
        dt = np.diff(t_o, prepend=t_o[0]); dt[0] = 0.0
        r_gnss = np.sum(v_g_i * dt)
        diff = r_odom - r_gnss
        pct = abs(diff) / r_gnss * 100.0
        print(f"\n3. Итоговый путь: Одометрия = {r_odom:.2f} м, GNSS = {r_gnss:.2f} м")
        print(f"   Финальная ошибка: {diff:+.2f} м ({pct:.3f}%)")
    print("\n")


if __name__ == "__main__":
    test_route("щук-талл", "щукинская - таллинская.json", "30618_0e41eac3")
    test_route("талл-щук", "таллинская - щукинская.json", "30618_073f08d1")
    test_route("талл-щук", "таллинская - щукинская.json", "30639_9c362687")

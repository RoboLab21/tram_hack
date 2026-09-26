#!/usr/bin/env python3
"""
Детальная диагностика худших заездов:
Определение конкретных моментов времени, участков пути и физических причин,
где накапливается погрешность одометрии.
"""

import sys
from pathlib import Path
_repo_dir = Path(__file__).resolve().parent.parent
for _p in [str(_repo_dir), str(_repo_dir / "solution"), str(_repo_dir / "scripts")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import matplotlib.pyplot as plt
from rosbags.highlevel import AnyReader
from batch_evaluate_filtered import setup_typestore
from solution.odometry_node_path_calibrated import (
    PathCalibratedDeadReckoningNode,
    extract_straight_sections
)

ws_dir = _repo_dir
typestore = setup_typestore(ws_dir)

straights_map = {
    "щук-талл": extract_straight_sections(ws_dir / "pathgrath" / "щукинская - таллинская.json"),
    "талл-щук": extract_straight_sections(ws_dir / "pathgrath" / "таллинская - щукинская.json"),
}

worst_bags = [
    ("30639_3b3d9eb8", "щук-талл"),
    ("30618_2366c74a", "талл-щук"),
    ("30639_927002c2", "талл-щук"),
    ("30639_9f0b519f", "талл-щук"),
    ("30618_88548b02", "талл-щук"),
]

for bag_name, route_name in worst_bags:
    bag_path = ws_dir / "data" / route_name / bag_name
    if not bag_path.exists():
        continue
    
    node = PathCalibratedDeadReckoningNode(
        straight_sections=straights_map[route_name],
        input_in_kmh=True,
        integration_method="trapezoidal"
    )
    
    v1_list, v2_list, gnss_records = [], [], []
    with AnyReader([bag_path], default_typestore=typestore) as reader:
        conns = [c for c in reader.connections if c.topic in {
            "/vehicle/front_bogie_velocity",
            "/vehicle/rear_bogie_velocity",
            "/sensing/gnss/master/vel"
        }]
        for conn, ts, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            if t <= 0: t = ts * 1e-9
            if conn.topic == "/vehicle/front_bogie_velocity":
                node.update_front(t, msg.velocity)
                v1_list.append((t, msg.velocity / 3.6))
            elif conn.topic == "/vehicle/rear_bogie_velocity":
                node.update_rear(t, msg.velocity)
                v2_list.append((t, msg.velocity / 3.6))
            elif conn.topic == "/sensing/gnss/master/vel":
                v = float(np.hypot(msg.twist.linear.x, msg.twist.linear.y))
                gnss_records.append((t, v))
                
    arr = node.get_arrays()
    t_o = arr["timestamp"]
    v_est = arr["v_est"]
    r_est = arr["distance"]
    k_scale = arr["k_scale"]
    
    tg = np.array([x[0] for x in gnss_records])
    vg = np.array([x[1] for x in gnss_records])
    s_i = np.argsort(tg); tg, vg = tg[s_i], vg[s_i]
    
    # Регулярная сетка
    t_start = max(t_o[0], tg[0])
    t_end = min(t_o[-1], tg[-1])
    t_grid = np.linspace(t_start, t_end, int((t_end - t_start) * 20)) # 20 Гц
    dt = np.diff(t_grid, prepend=t_grid[0]); dt[0] = 0.0
    
    v_est_i = np.interp(t_grid, t_o, v_est)
    v_gnss_i = np.interp(t_grid, tg, vg)
    v1_i = np.interp(t_grid, [x[0] for x in v1_list], [x[1] for x in v1_list])
    v2_i = np.interp(t_grid, [x[0] for x in v2_list], [x[1] for x in v2_list])
    k_scale_i = np.interp(t_grid, t_o, k_scale)
    
    r_est_grid = np.cumsum(v_est_i * dt)
    r_gnss_grid = np.cumsum(v_gnss_i * dt)
    
    err_curve = r_est_grid - r_gnss_grid # Ошибка одометрии от времени
    t_rel = t_grid - t_start
    
    # Находим интервалы максимального прироста ошибки:
    # Окно скользящего прироста ошибки за 10 секунд:
    window_pts = 200 # 10 сек при 20 Гц
    if len(err_curve) > window_pts:
        delta_err_10s = err_curve[window_pts:] - err_curve[:-window_pts]
        t_mid = t_rel[window_pts//2 : -window_pts//2]
        s_mid = r_gnss_grid[window_pts//2 : -window_pts//2]
        
        # Топ-3 интервала с наибольшим приростом ошибки:
        top_jump_idx = np.argsort(np.abs(delta_err_10s))[::-1]
        
        print("\n" + "="*80)
        print(f"ЗАЕЗД {bag_name} ({route_name}): Финальная ошибка = {err_curve[-1]:+.2f} м ({abs(err_curve[-1])/r_gnss_grid[-1]*100:.3f}%), k_scale = {node.k_scale:.5f}")
        print("="*80)
        
        # Разделение ошибки: постоянный линейный дрейф vs скачки
        # Оценка фонового масштаба (линейного дрейфа):
        p_fit = np.polyfit(r_gnss_grid, err_curve, 1) # p_fit[0] = residual scale slope
        slope_pct = p_fit[0] * 100.0
        print(f"  Фоновый остаточный дрейф масштаба (slope): {slope_pct:+.3f}% (дает {p_fit[0]*r_gnss_grid[-1]:+.1f} м ошибки)")
        
        # Анализ ключевых аномальных событий:
        print("\n  Ключевые моменты накопления ошибки:")
        reported_times = []
        for idx in top_jump_idx:
            t_event = t_mid[idx]
            if any(abs(t_event - rt) < 30.0 for rt in reported_times):
                continue
            reported_times.append(t_event)
            s_event = s_mid[idx]
            d_err = delta_err_10s[idx]
            
            # Локальные скорости в этот момент:
            loc_mask = (t_rel >= t_event - 5.0) & (t_rel <= t_event + 5.0)
            v1_loc = np.mean(v1_i[loc_mask]) * 3.6
            v2_loc = np.mean(v2_i[loc_mask]) * 3.6
            vg_loc = np.mean(v_gnss_i[loc_mask]) * 3.6
            a_loc = (v_est_i[loc_mask][-1] - v_est_i[loc_mask][0]) / 10.0
            
            event_desc = []
            if abs(v1_loc - v2_loc) > 3.0:
                event_desc.append(f"Сбой датчиков (|v1-v2|={abs(v1_loc-v2_loc):.1f} км/ч)")
            if v1_loc < 1.0 and vg_loc > 3.0:
                event_desc.append("Датчик 1 занулился на ходу")
            if v2_loc < 1.0 and vg_loc > 3.0:
                event_desc.append("Датчик 2 занулился на ходу")
            if abs(a_loc) > 1.2:
                event_desc.append(f"Интенсивный {'разгон' if a_loc > 0 else 'торможение'} (a={a_loc:+.2f} м/с²)")
            if vg_loc < 0.5 and (v1_loc > 0.5 or v2_loc > 0.5):
                event_desc.append("Буксование при трогании")
            if (v1_loc < 0.5 or v2_loc < 0.5) and vg_loc > 0.5:
                event_desc.append("Юз при торможении")
            if not event_desc:
                event_desc.append("Крейсерский ход (накопление систематического износа бандажа)")
                
            print(f"    - t = {t_event:6.1f} с (путь S = {s_event:6.1f} м): скачок ошибки {d_err:+5.2f} м за 10с. V_est={v1_loc:4.1f} км/ч, V_gnss={vg_loc:4.1f} км/ч. Событие: {', '.join(event_desc)}")
            if len(reported_times) >= 4:
                break
                
    # Создание диагностического графика
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    
    ax1.plot(t_rel, err_curve, color="crimson", lw=2, label="Накопленная ошибка ΔR (м)")
    ax1.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax1.set_ylabel("Ошибка одометрии ΔR, м", fontweight="bold")
    ax1.set_title(f"Динамика накопления ошибки: {bag_name} ({route_name})", fontweight="bold", fontsize=12)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left")
    
    ax2.plot(t_rel, v1_i * 3.6, color="blue", alpha=0.7, lw=1.2, label="Передняя тележка v1")
    ax2.plot(t_rel, v2_i * 3.6, color="green", alpha=0.7, lw=1.2, label="Задняя тележка v2")
    ax2.plot(t_rel, v_gnss_i * 3.6, color="black", linestyle="--", lw=1.5, label="GNSS эталон")
    ax2.set_ylabel("Скорость, км/ч", fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right")
    
    ax3.plot(t_rel, (v1_i - v2_i) * 3.6, color="purple", lw=1.2, label="Разность тележек (v1 - v2), км/ч")
    ax3.plot(t_rel, (v_est_i - v_gnss_i) * 3.6, color="orange", lw=1.2, label="Отклонение скорости (V_est - V_gnss), км/ч")
    ax3.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax3.set_ylabel("Расхождение, км/ч", fontweight="bold")
    ax3.set_xlabel("Время от начала заезда, сек", fontweight="bold")
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="upper right")
    
    plt.tight_layout()
    plot_file = ws_dir / "img" / f"error_diagnosis_{bag_name}.png"
    plt.savefig(plot_file, dpi=130)
    plt.close()
    print(f"  График сохранен: {plot_file.name}")

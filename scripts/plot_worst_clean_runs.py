import sys
from pathlib import Path
_repo_dir = Path(__file__).resolve().parent.parent
for _p in [str(_repo_dir), str(_repo_dir / "solution"), str(_repo_dir / "scripts")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

#!/usr/bin/env python3
"""
Построение детальных диагностических графиков по худшим прогонам чистого датасета.
Строит скорости, физические ускорения (м/с²), зумы проблемных участков и накопление погрешности.
"""

import os
from pathlib import Path
import numpy as np
import matplotlib
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rosbags.highlevel import AnyReader
from batch_evaluate_filtered import setup_typestore
from odometry_node import DeadReckoningNode
from odometry_node_filtered import FilteredDeadReckoningNode

ws_dir = _repo_dir
data_dir = ws_dir / "data"
img_dir = ws_dir / "img"
img_dir.mkdir(exist_ok=True)
typestore = setup_typestore(ws_dir)


def load_bag(bag_name):
    node_base = DeadReckoningNode()
    node_filt = FilteredDeadReckoningNode()
    gnss_records = []
    
    bpath = data_dir / bag_name
    with AnyReader([bpath], default_typestore=typestore) as reader:
        conns = [c for c in reader.connections if c.topic in {
            "/vehicle/front_bogie_velocity", "/vehicle/rear_bogie_velocity", "/sensing/gnss/master/vel"
        }]
        for conn, timestamp, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            if t <= 0: t = timestamp * 1e-9
            if conn.topic == "/vehicle/front_bogie_velocity":
                node_base.update_front(t, msg.velocity)
                node_filt.update_front(t, msg.velocity)
            elif conn.topic == "/vehicle/rear_bogie_velocity":
                node_base.update_rear(t, msg.velocity)
                node_filt.update_rear(t, msg.velocity)
            elif conn.topic == "/sensing/gnss/master/vel":
                v = float(np.hypot(msg.twist.linear.x, msg.twist.linear.y))
                gnss_records.append((t, v))
                
    base_data = node_base.get_arrays()
    filt_data = node_filt.get_arrays()
    
    t_o = filt_data["timestamp"]
    t_rel = t_o - t_o[0]
    
    t_g = np.array([x[0] for x in gnss_records])
    v_g = np.array([x[1] for x in gnss_records])
    s_idx = np.argsort(t_g)
    t_g, v_g = t_g[s_idx], v_g[s_idx]
    
    v_gnss_ms = np.interp(t_o, t_g, v_g)
    dt = np.diff(t_o, prepend=t_o[0]); dt[0] = 0.0
    r_gnss = np.cumsum(v_gnss_ms * dt)
    
    v1_ms = filt_data["v1_raw"]
    v2_ms = filt_data["v2_raw"]
    v_est_ms = filt_data["v_est"]

    # Рассчитываем физические ускорения на равномерной сетке 10 Гц (dt = 0.1 с)
    t_uniform = np.arange(0, t_rel[-1], 0.1)
    v1_u = np.interp(t_uniform, t_rel, v1_ms)
    v2_u = np.interp(t_uniform, t_rel, v2_ms)
    vest_u = np.interp(t_uniform, t_rel, v_est_ms)
    vgnss_u = np.interp(t_uniform, t_rel, v_gnss_ms)
    
    dt_u = 0.1
    # Сглаживание скорости окном 0.5с перед дифференцированием
    w = 5
    kernel = np.ones(w) / w
    v1_s = np.convolve(v1_u, kernel, mode="same")
    v2_s = np.convolve(v2_u, kernel, mode="same")
    vest_s = np.convolve(vest_u, kernel, mode="same")
    vgnss_s = np.convolve(vgnss_u, kernel, mode="same")
    
    a1 = np.gradient(v1_s, dt_u)
    a2 = np.gradient(v2_s, dt_u)
    a_filt = np.gradient(vest_s, dt_u)
    a_gnss = np.gradient(vgnss_s, dt_u)
    
    # Интерполируем ускорения обратно на исходную сетку времени t_rel
    a1 = np.interp(t_rel, t_uniform, a1)
    a2 = np.interp(t_rel, t_uniform, a2)
    a_filt = np.interp(t_rel, t_uniform, a_filt)
    a_gnss = np.interp(t_rel, t_uniform, a_gnss)
    
    return {
        "t_rel": t_rel,
        "v1_kmh": v1_ms * 3.6,
        "v2_kmh": v2_ms * 3.6,
        "v_filt_kmh": v_est_ms * 3.6,
        "v_base_kmh": base_data["v_avg"] * 3.6,
        "v_gnss_kmh": v_gnss_ms * 3.6,
        "a1": a1,
        "a2": a2,
        "a_filt": a_filt,
        "a_gnss": a_gnss,
        "r_filt": filt_data["distance"],
        "r_base": base_data["distance"],
        "r_gnss": r_gnss,
        "fault_types": [s.fault_type for s in node_filt.history]
    }


def plot_bag_1():
    # 30639_3b3d9eb8
    d = load_bag("30639_3b3d9eb8")
    t = d["t_rel"]
    
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.2, 1.0])
    
    # 1. Верхний график: весь прогон
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.plot(t, d["v_gnss_kmh"], label="GNSS эталон", color="black", alpha=0.7, lw=2)
    ax_top.plot(t, d["v1_kmh"], label="Датчик 1 (передняя тележка)", color="tab:blue", lw=1.2, alpha=0.8)
    ax_top.plot(t, d["v2_kmh"], label="Датчик 2 (задняя тележка, выпадение в 0)", color="tab:red", lw=1.2, ls="--")
    ax_top.plot(t, d["v_filt_kmh"], label="Оценка фильтра v2", color="tab:green", lw=1.6)
    ax_top.axvspan(375.9, 404.7, color="red", alpha=0.18, label="Зона 1 отказа датчика 2 (28.8с на 0 км/ч)")
    ax_top.axvspan(421.7, 494.6, color="red", alpha=0.25, label="Зона 2 отказа датчика 2 (72.9с на 0 км/ч)")
    ax_top.set_ylabel("Скорость, км/ч", fontsize=11)
    ax_top.set_title("Прогон 30639_3b3d9eb8 (Ошибка: База -294.4м → Фильтр +49.2м, 0.90%): Полный профиль скорости", fontsize=13, fontweight="bold")
    ax_top.grid(True, linestyle="--", alpha=0.6)
    ax_top.legend(loc="upper right", framealpha=0.9, fontsize=9)
    
    # 2. Зум проблемного участка: скорость
    ax_z_v = fig.add_subplot(gs[1, 0])
    mask_z = (t >= 360) & (t <= 510)
    ax_z_v.plot(t[mask_z], d["v_gnss_kmh"][mask_z], label="GNSS", color="black", lw=2)
    ax_z_v.plot(t[mask_z], d["v1_kmh"][mask_z], label="Датчик 1 (норма)", color="tab:blue", lw=1.8)
    ax_z_v.plot(t[mask_z], d["v2_kmh"][mask_z], label="Датчик 2 (нулевой отказ)", color="tab:red", lw=2, ls="--")
    ax_z_v.plot(t[mask_z], d["v_filt_kmh"][mask_z], label="Фильтр v2", color="tab:green", lw=2)
    ax_z_v.axvspan(375.9, 404.7, color="red", alpha=0.18)
    ax_z_v.axvspan(421.7, 494.6, color="red", alpha=0.25)
    ax_z_v.set_ylabel("Скорость, км/ч", fontsize=11)
    ax_z_v.set_xlabel("Время, сек", fontsize=11)
    ax_z_v.set_title("ЗУМ: Отказ датчика 2 при движении трамвая на скорости 25 км/ч", fontsize=11, fontweight="bold")
    ax_z_v.grid(True, linestyle="--", alpha=0.6)
    ax_z_v.legend(loc="upper right", fontsize=9)
    
    # 3. Зум проблемного участка: ускорения (производные)
    ax_z_a = fig.add_subplot(gs[1, 1])
    ax_z_a.plot(t[mask_z], d["a_gnss"][mask_z], label="a GNSS", color="black", lw=1.5, alpha=0.8)
    ax_z_a.plot(t[mask_z], d["a1"][mask_z], label="a датчика 1", color="tab:blue", lw=1.5)
    ax_z_a.plot(t[mask_z], d["a2"][mask_z], label="a датчика 2 (скачки при падении/возврате)", color="tab:red", lw=1.5, ls="--")
    ax_z_a.plot(t[mask_z], d["a_filt"][mask_z], label="a фильтра", color="tab:green", lw=1.8)
    ax_z_a.axvspan(375.9, 404.7, color="red", alpha=0.18)
    ax_z_a.axvspan(421.7, 494.6, color="red", alpha=0.25)
    ax_z_a.set_ylabel("Ускорение, м/с²", fontsize=11)
    ax_z_a.set_xlabel("Время, сек", fontsize=11)
    ax_z_a.set_ylim(-3.0, 3.0)
    ax_z_a.set_title("ЗУМ: Производные скорости (ускорения) в момент обрыва связи датчика", fontsize=11, fontweight="bold")
    ax_z_a.grid(True, linestyle="--", alpha=0.6)
    ax_z_a.legend(loc="upper right", fontsize=9)
    
    # 4. Нижний график: Накопление ошибки пути
    ax_bot = fig.add_subplot(gs[2, :])
    err_base = d["r_base"] - d["r_gnss"]
    err_filt = d["r_filt"] - d["r_gnss"]
    ax_bot.plot(t, err_base, label=f"Базовая нода (без фильтра, ошибка {err_base[-1]:+.1f}м)", color="tab:red", lw=2, ls="--")
    ax_bot.plot(t, err_filt, label=f"Фильтрованная нода v2 (ошибка {err_filt[-1]:+.1f}м, дрейф всего {np.max(np.abs(err_filt)):.1f}м)", color="tab:green", lw=2.2)
    ax_bot.axvspan(375.9, 404.7, color="red", alpha=0.18)
    ax_bot.axvspan(421.7, 494.6, color="red", alpha=0.25)
    ax_bot.axhline(0, color="gray", lw=1)
    ax_bot.set_ylabel("Ошибка пути ΔR, м", fontsize=11)
    ax_bot.set_xlabel("Время прогона, сек", fontsize=11)
    ax_bot.set_title("Накопленная погрешность пути: База теряет 294 метра за 101 сек; Фильтр полностью парирует сбой", fontsize=12)
    ax_bot.grid(True, linestyle="--", alpha=0.6)
    ax_bot.legend(loc="upper left", fontsize=10)
    
    plt.tight_layout()
    out = img_dir / "worst_run_1_30639_3b3d9eb8.png"
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"Сохранен: {out.name}")


def plot_bag_2():
    # 30618_88548b02 (-40.4м, -0.75%)
    d = load_bag("30618_88548b02")
    t = d["t_rel"]
    
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.2, 1.0])
    
    # 1. Верхний график: весь прогон
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.plot(t, d["v_gnss_kmh"], label="GNSS эталон", color="black", alpha=0.8, lw=1.8)
    ax_top.plot(t, d["v1_kmh"], label="Датчик 1 (передняя тележка)", color="tab:blue", lw=1.2, alpha=0.7)
    ax_top.plot(t, d["v2_kmh"], label="Датчик 2 (задняя тележка)", color="tab:orange", lw=1.2, alpha=0.7)
    ax_top.set_ylabel("Скорость, км/ч", fontsize=11)
    ax_top.set_title("Прогон 30618_88548b02 (Ошибка: -40.4м, -0.75%): Отсутствие сбоев датчиков, идеальная синхронность", fontsize=13, fontweight="bold")
    ax_top.grid(True, linestyle="--", alpha=0.6)
    ax_top.legend(loc="upper right", framealpha=0.9, fontsize=9)
    
    # 2. Зум маневра разгона и торможения: скорости
    mask_z = (t >= 630) & (t <= 750)
    ax_z_v = fig.add_subplot(gs[1, 0])
    ax_z_v.plot(t[mask_z], d["v_gnss_kmh"][mask_z], label="GNSS", color="black", lw=2)
    ax_z_v.plot(t[mask_z], d["v1_kmh"][mask_z], label="Колеса тележки 1", color="tab:blue", lw=1.8)
    ax_z_v.plot(t[mask_z], d["v2_kmh"][mask_z], label="Колеса тележки 2", color="tab:orange", lw=1.8, ls="--")
    ax_z_v.set_ylabel("Скорость, км/ч", fontsize=11)
    ax_z_v.set_xlabel("Время, сек", fontsize=11)
    ax_z_v.set_title("ЗУМ: Разгон до 47 км/ч и торможение (колеса работают синхронно, но на 0.75% медленнее GNSS)", fontsize=10, fontweight="bold")
    ax_z_v.grid(True, linestyle="--", alpha=0.6)
    ax_z_v.legend(loc="upper right", fontsize=9)
    
    # 3. Зум ускорений
    ax_z_a = fig.add_subplot(gs[1, 1])
    ax_z_a.plot(t[mask_z], d["a_gnss"][mask_z], label="a GNSS", color="black", lw=1.5)
    ax_z_a.plot(t[mask_z], d["a_filt"][mask_z], label="a одометрии (колеса)", color="tab:blue", lw=1.5)
    ax_z_a.set_ylabel("Ускорение, м/с²", fontsize=11)
    ax_z_a.set_xlabel("Время, сек", fontsize=11)
    ax_z_a.set_ylim(-2.0, 2.0)
    ax_z_a.set_title("ЗУМ: Ускорения трамвая (нет ни проскальзывания при разгоне, ни юза при торможении)", fontsize=10, fontweight="bold")
    ax_z_a.grid(True, linestyle="--", alpha=0.6)
    ax_z_a.legend(loc="upper right", fontsize=9)
    
    # 4. Накопленная погрешность
    ax_bot = fig.add_subplot(gs[2, :])
    err_filt = d["r_filt"] - d["r_gnss"]
    ideal_scale = -0.0075 * d["r_gnss"]
    ax_bot.plot(t, err_filt, label=f"Фактическая ошибка одометрии ΔR (финал: {err_filt[-1]:+.1f}м)", color="tab:blue", lw=2)
    ax_bot.plot(t, ideal_scale, label="Теоретическая прямая износа радиуса колеса (-0.75%)", color="tab:red", ls="--", lw=2)
    ax_bot.set_ylabel("Ошибка пути ΔR, м", fontsize=11)
    ax_bot.set_xlabel("Время прогона, сек", fontsize=11)
    ax_bot.set_title("Причина ошибки: 100% систематический масштабный недокат (износ бандажей колес трамвая 30618 на ~2.5 мм)", fontsize=12)
    ax_bot.grid(True, linestyle="--", alpha=0.6)
    ax_bot.legend(loc="lower left", fontsize=10)
    
    plt.tight_layout()
    out = img_dir / "worst_run_2_30618_88548b02.png"
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"Сохранен: {out.name}")


def plot_bag_3():
    # 30639_9c362687 (+28.9м, +0.53%)
    d = load_bag("30639_9c362687")
    t = d["t_rel"]
    
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.2, 1.0])
    
    # 1. Полный профиль
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.plot(t, d["v_gnss_kmh"], label="GNSS эталон", color="black", alpha=0.8, lw=1.8)
    ax_top.plot(t, d["v1_kmh"], label="Датчик 1", color="tab:blue", lw=1.2, alpha=0.7)
    ax_top.plot(t, d["v2_kmh"], label="Датчик 2", color="tab:orange", lw=1.2, alpha=0.7)
    ax_top.set_ylabel("Скорость, км/ч", fontsize=11)
    ax_top.set_title("Прогон 30639_9c362687 (Ошибка: +28.9м, +0.53%): Чистый прогон с масштабным перекатом", fontsize=13, fontweight="bold")
    ax_top.grid(True, linestyle="--", alpha=0.6)
    ax_top.legend(loc="upper right", framealpha=0.9, fontsize=9)
    
    # 2. Зум разгона/торможения
    mask_z = (t >= 750) & (t <= 900)
    ax_z_v = fig.add_subplot(gs[1, 0])
    ax_z_v.plot(t[mask_z], d["v_gnss_kmh"][mask_z], label="GNSS", color="black", lw=2)
    ax_z_v.plot(t[mask_z], d["v_filt_kmh"][mask_z], label="Одометрия", color="tab:green", lw=1.8)
    ax_z_v.set_ylabel("Скорость, км/ч", fontsize=11)
    ax_z_v.set_xlabel("Время, сек", fontsize=11)
    ax_z_v.set_title("ЗУМ: Скорость колес стабильно превышает GNSS на +0.5% на крейсерском ходу", fontsize=10, fontweight="bold")
    ax_z_v.grid(True, linestyle="--", alpha=0.6)
    ax_z_v.legend(loc="upper right", fontsize=9)
    
    # 3. Зум ускорений
    ax_z_a = fig.add_subplot(gs[1, 1])
    ax_z_a.plot(t[mask_z], d["a_gnss"][mask_z], label="a GNSS", color="black", lw=1.5)
    ax_z_a.plot(t[mask_z], d["a_filt"][mask_z], label="a одометрии", color="tab:green", lw=1.5)
    ax_z_a.set_ylabel("Ускорение, м/с²", fontsize=11)
    ax_z_a.set_xlabel("Время, сек", fontsize=11)
    ax_z_a.set_ylim(-2.0, 2.0)
    ax_z_a.set_title("ЗУМ: Физические ускорения (плавные профили разгона до 1.3 м/с² и торможения до -1.4 м/с²)", fontsize=10, fontweight="bold")
    ax_z_a.grid(True, linestyle="--", alpha=0.6)
    ax_z_a.legend(loc="upper right", fontsize=9)
    
    # 4. Накопление ошибки
    ax_bot = fig.add_subplot(gs[2, :])
    err_filt = d["r_filt"] - d["r_gnss"]
    ideal_scale = 0.0053 * d["r_gnss"]
    ax_bot.plot(t, err_filt, label=f"Фактическая ошибка одометрии ΔR (финал: {err_filt[-1]:+.1f}м)", color="tab:green", lw=2)
    ax_bot.plot(t, ideal_scale, label="Теоретическая прямая калибровки радиуса колеса (+0.53%)", color="tab:red", ls="--", lw=2)
    ax_bot.set_ylabel("Ошибка пути ΔR, м", fontsize=11)
    ax_bot.set_xlabel("Время прогона, сек", fontsize=11)
    ax_bot.set_title("Причина ошибки: монотонное нарастание ошибки строго пропорционально пройденному расстоянию", fontsize=12)
    ax_bot.grid(True, linestyle="--", alpha=0.6)
    ax_bot.legend(loc="upper left", fontsize=10)
    
    plt.tight_layout()
    out = img_dir / "worst_run_3_30639_9c362687.png"
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"Сохранен: {out.name}")


def plot_bag_4():
    # 30639_927002c2 (-21.1м, -0.39%)
    d = load_bag("30639_927002c2")
    t = d["t_rel"]
    
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.2, 1.0])
    
    # 1. Полный профиль
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.plot(t, d["v_gnss_kmh"], label="GNSS эталон", color="black", alpha=0.7, lw=2)
    ax_top.plot(t, d["v1_kmh"], label="Датчик 1 (передняя тележка)", color="tab:blue", lw=1.2)
    ax_top.plot(t, d["v2_kmh"], label="Датчик 2 (выпадал в 0 на 22с)", color="tab:red", lw=1.2, ls="--")
    ax_top.plot(t, d["v_filt_kmh"], label="Оценка фильтра v2", color="tab:green", lw=1.6)
    ax_top.axvspan(247.6, 267.2, color="red", alpha=0.22, label="Отказ 1 датчика 2 (19.6с на 0 км/ч)")
    ax_top.axvspan(919.7, 927.4, color="red", alpha=0.22, label="Отказ 2 датчика 2 (7.7с на 0 км/ч)")
    ax_top.set_ylabel("Скорость, км/ч", fontsize=11)
    ax_top.set_title("Прогон 30639_927002c2 (Ошибка: База -105.0м → Фильтр -21.1м, -0.39%): Полный профиль скорости", fontsize=13, fontweight="bold")
    ax_top.grid(True, linestyle="--", alpha=0.6)
    ax_top.legend(loc="upper right", framealpha=0.9, fontsize=9)
    
    # 2. Зум первого отказа
    mask_z = (t >= 235) & (t <= 285)
    ax_z_v = fig.add_subplot(gs[1, 0])
    ax_z_v.plot(t[mask_z], d["v_gnss_kmh"][mask_z], label="GNSS", color="black", lw=2)
    ax_z_v.plot(t[mask_z], d["v1_kmh"][mask_z], label="Датчик 1 (норма)", color="tab:blue", lw=1.8)
    ax_z_v.plot(t[mask_z], d["v2_kmh"][mask_z], label="Датчик 2 (нулевой отказ)", color="tab:red", lw=2, ls="--")
    ax_z_v.plot(t[mask_z], d["v_filt_kmh"][mask_z], label="Фильтр v2", color="tab:green", lw=2)
    ax_z_v.axvspan(247.6, 267.2, color="red", alpha=0.22)
    ax_z_v.set_ylabel("Скорость, км/ч", fontsize=11)
    ax_z_v.set_xlabel("Время, сек", fontsize=11)
    ax_z_v.set_title("ЗУМ: Отказ датчика 2 при движении на скорости 30-40 км/ч", fontsize=10, fontweight="bold")
    ax_z_v.grid(True, linestyle="--", alpha=0.6)
    ax_z_v.legend(loc="upper right", fontsize=9)
    
    # 3. Зум ускорений
    ax_z_a = fig.add_subplot(gs[1, 1])
    ax_z_a.plot(t[mask_z], d["a_gnss"][mask_z], label="a GNSS", color="black", lw=1.5)
    ax_z_a.plot(t[mask_z], d["a1"][mask_z], label="a датчика 1", color="tab:blue", lw=1.5)
    ax_z_a.plot(t[mask_z], d["a2"][mask_z], label="a датчика 2 (нефизичный сброс)", color="tab:red", lw=1.5, ls="--")
    ax_z_a.plot(t[mask_z], d["a_filt"][mask_z], label="a фильтра", color="tab:green", lw=1.8)
    ax_z_a.axvspan(247.6, 267.2, color="red", alpha=0.22)
    ax_z_a.set_ylabel("Ускорение, м/с²", fontsize=11)
    ax_z_a.set_xlabel("Время, сек", fontsize=11)
    ax_z_a.set_ylim(-3.0, 3.0)
    ax_z_a.set_title("ЗУМ: Ускорения (фильтр отсекает скачок зануления датчика 2)", fontsize=10, fontweight="bold")
    ax_z_a.grid(True, linestyle="--", alpha=0.6)
    ax_z_a.legend(loc="upper right", fontsize=9)
    
    # 4. Накопление ошибки
    ax_bot = fig.add_subplot(gs[2, :])
    err_base = d["r_base"] - d["r_gnss"]
    err_filt = d["r_filt"] - d["r_gnss"]
    ax_bot.plot(t, err_base, label=f"Базовая нода (без фильтра, ошибка {err_base[-1]:+.1f}м)", color="tab:red", lw=2, ls="--")
    ax_bot.plot(t, err_filt, label=f"Фильтрованная нода v2 (ошибка {err_filt[-1]:+.1f}м, дрейф всего {np.max(np.abs(err_filt)):.1f}м)", color="tab:green", lw=2.2)
    ax_bot.axvspan(247.6, 267.2, color="red", alpha=0.22)
    ax_bot.axvspan(919.7, 927.4, color="red", alpha=0.22)
    ax_bot.axhline(0, color="gray", lw=1)
    ax_bot.set_ylabel("Ошибка пути ΔR, м", fontsize=11)
    ax_bot.set_xlabel("Время прогона, сек", fontsize=11)
    ax_bot.set_title("Накопленная погрешность пути: База потеряла 105м за 2 отказа; Фильтр сократил ошибку в 5 раз (до 21м)", fontsize=12)
    ax_bot.grid(True, linestyle="--", alpha=0.6)
    ax_bot.legend(loc="lower left", fontsize=10)
    
    plt.tight_layout()
    out = img_dir / "worst_run_4_30639_927002c2.png"
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"Сохранен: {out.name}")


if __name__ == "__main__":
    plot_bag_1()
    plot_bag_2()
    plot_bag_3()
    plot_bag_4()

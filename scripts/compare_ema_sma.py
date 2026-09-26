#!/usr/bin/env python3
"""
Сравнительный анализ формул счисления пути:
Обычное среднее V_cp = (V_front + V_rear) / 2 vs SMA (Simple Moving Average) vs EMA (Exponential Moving Average).
С меняемым количеством точек N (3, 5, 10, 15, 20, 30).
"""

import sys
from pathlib import Path
_repo_dir = Path(__file__).resolve().parent.parent
for _p in [str(_repo_dir), str(_repo_dir / "solution"), str(_repo_dir / "scripts")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import json
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from rosbags.highlevel import AnyReader
from batch_evaluate_filtered import setup_typestore

ws_dir = _repo_dir
typestore = setup_typestore(ws_dir)

clean_bags = sorted(list((ws_dir / "data/щук-талл").glob("306*")) + list((ws_dir / "data/талл-щук").glob("306*")))
print(f"Всего валидных заездов для анализа: {len(clean_bags)}")

# Предзагрузка данных в память для быстрого многократного прогона
print("Кэширование данных заездов в память...")
cached_bags = []
for bp in clean_bags:
    t_v1, v1_list = [], []
    t_v2, v2_list = [], []
    t_g, vg_list = [], []
    with AnyReader([bp], default_typestore=typestore) as reader:
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
                t_v1.append(t)
                v1_list.append(msg.velocity / 3.6)
            elif conn.topic == "/vehicle/rear_bogie_velocity":
                t_v2.append(t)
                v2_list.append(msg.velocity / 3.6)
            elif conn.topic == "/sensing/gnss/master/vel":
                t_g.append(t)
                vg_list.append(float(np.hypot(msg.twist.linear.x, msg.twist.linear.y)))
                
    if len(vg_list) >= 10 and t_v1 and t_v2:
        events = [(t, 1, v) for t, v in zip(t_v1, v1_list)] + [(t, 2, v) for t, v in zip(t_v2, v2_list)]
        events.sort(key=lambda x: x[0])
        tg = np.array(t_g)
        vg = np.array(vg_list)
        s_idx = np.argsort(tg); tg, vg = tg[s_idx], vg[s_idx]
        cached_bags.append((bp.name, events, tg, vg))

print(f"Успешно загружено в память: {len(cached_bags)} заездов.")


def evaluate_filter(method: str, N: int):
    """
    Прогоняет счисление пути с выбранным методом усреднения скорости:
    method: "mean", "sma", "ema"
    N: количество точек сглаживания
    """
    final_errs_m = []
    final_pcts = []
    max_drifts_m = []
    rmse_drifts_m = []
    jitter_list = []

    alpha = 2.0 / (N + 1)

    for bname, events, tg, vg in cached_bags:
        v1, v2 = 0.0, 0.0
        last_t = None
        last_v = 0.0
        R = 0.0

        sma_buf = deque(maxlen=N)
        ema_val = None

        t_rec = []
        r_rec = []
        v_filt_rec = []

        for t, bogie, v in events:
            if bogie == 1: v1 = v
            else: v2 = v

            # Базовая полусумма V_cp(t) = (V_front + V_rear) / 2
            v_raw = (v1 + v2) / 2.0

            if method == "mean" or N <= 1:
                v_speed = v_raw
            elif method == "sma":
                sma_buf.append(v_raw)
                v_speed = float(np.mean(sma_buf))
            elif method == "ema":
                if ema_val is None:
                    ema_val = v_raw
                else:
                    ema_val = alpha * v_raw + (1.0 - alpha) * ema_val
                v_speed = ema_val
            else:
                v_speed = v_raw

            if last_t is None:
                last_t = t
                last_v = v_speed
                t_rec.append(t)
                r_rec.append(0.0)
                v_filt_rec.append(v_speed)
                continue

            dt = t - last_t
            if dt <= 0.0 or dt > 2.0:
                last_t = t
                continue

            v_eff = (last_v + v_speed) / 2.0
            R += v_eff * dt
            last_t = t
            last_v = v_speed
            t_rec.append(t)
            r_rec.append(R)
            v_filt_rec.append(v_speed)

        t_o = np.array(t_rec)
        r_o = np.array(r_rec)
        v_o = np.array(v_filt_rec)

        # Оценка шума/джиттера производной скорости dV/dt
        dt_v = np.diff(t_o)
        valid_dt = dt_v > 0.001
        acc = np.diff(v_o)[valid_dt] / dt_v[valid_dt]
        jitter = float(np.std(acc)) if len(acc) > 0 else 0.0
        jitter_list.append(jitter)

        # Синхронизация с GNSS
        v_gnss_i = np.interp(t_o, tg, vg)
        dt_arr = np.diff(t_o, prepend=t_o[0]); dt_arr[0] = 0.0
        r_gnss_arr = np.cumsum(v_gnss_i * dt_arr)
        tot_gnss = r_gnss_arr[-1]

        final_err = abs(R - tot_gnss)
        pct = final_err / tot_gnss * 100.0
        drift_arr = np.abs(r_o - r_gnss_arr)
        max_drift = float(np.max(drift_arr))
        rmse_drift = float(np.sqrt(np.mean(drift_arr ** 2)))

        final_errs_m.append(final_err)
        final_pcts.append(pct)
        max_drifts_m.append(max_drift)
        rmse_drifts_m.append(rmse_drift)

    return {
        "method": method.upper(),
        "N": N,
        "mean_final_err_m": float(np.mean(final_errs_m)),
        "median_final_err_m": float(np.median(final_errs_m)),
        "max_final_err_m": float(np.max(final_errs_m)),
        "mean_final_pct": float(np.mean(final_pcts)),
        "mean_max_drift_m": float(np.mean(max_drifts_m)),
        "max_max_drift_m": float(np.max(max_drifts_m)),
        "mean_rmse_drift_m": float(np.mean(rmse_drifts_m)),
        "mean_jitter_ms2": float(np.mean(jitter_list))
    }


# Экспериментальная сетка параметров
configurations = [
    ("mean", 1),
    ("sma", 3),
    ("sma", 5),
    ("sma", 10),
    ("sma", 15),
    ("sma", 20),
    ("sma", 30),
    ("ema", 3),
    ("ema", 5),
    ("ema", 10),
    ("ema", 15),
    ("ema", 20),
    ("ema", 30),
]

results = []
print("\nЗапуск оценки всех конфигураций...")
for m, n in configurations:
    res = evaluate_filter(m, n)
    results.append(res)
    print(f"  {res['method']:<4} (N={res['N']:2d}): Финал. ош. = {res['mean_final_err_m']:5.2f}м ({res['mean_final_pct']:.3f}%), Макс. дрейф = {res['mean_max_drift_m']:5.2f}м, Джиттер = {res['mean_jitter_ms2']:4.2f} м/с²")

# Печать сравнительной таблицы
print("\n" + "="*112)
print("СРАВНИТЕЛЬНАЯ ТАБЛИЦА: СРЕДНЯЯ СКОРОСТЬ vs SMA vs EMA С РАЗНЫМ КОЛИЧЕСТВОМ ТОЧЕК N (ВСЕ 56 БАГОВ)")
print("="*112)
header = (
    f"{'Метод':<10} | {'Точек N':<8} | {'Задержка τ':<10} | "
    f"{'Финальная ошибка ΔR':<24} | {'Макс. дрейф вдоль пути':<24} | {'Джиттер dV/dt':<14}\n"
    f"{'':<10} | {'':<8} | {'(шагов)':<10} | "
    f"{'Среднее':<8} {'Медиана':<8} {'Макс':<6} | {'Среднее':<10} {'Максимум':<12} | {'(м/с²)':<14}"
)
print(header)
print("-" * 112)

for r in results:
    tau_str = f"{(r['N']-1)/2:.1f} шагов" if r['N'] > 1 else "0 шагов"
    print(
        f"{r['method']:<10} | {r['N']:<8} | {tau_str:<10} | "
        f"{r['mean_final_err_m']:<8.2f} {r['median_final_err_m']:<8.2f} {r['max_final_err_m']:<6.2f} | "
        f"{r['mean_max_drift_m']:<10.2f} {r['max_max_drift_m']:<12.2f} | "
        f"{r['mean_jitter_ms2']:<14.2f}"
    )

# Построение сравнительного графика для одного характерного заезда
sample_bag = "30618_073f08d1"
sample_data = next((x for x in cached_bags if x[0] == sample_bag), None)

if sample_data:
    bname, events, tg, vg = sample_data
    # Возьмем отрезок интенсивного разгона и торможения (например, первые 100 секунд)
    t_start = events[0][0]
    sub_events = [e for e in events if e[0] - t_start <= 120.0]
    
    t_raw, v_raw_list = [], []
    v1_c, v2_c = 0.0, 0.0
    for t, b, v in sub_events:
        if b == 1: v1_c = v
        else: v2_c = v
        t_raw.append(t - t_start)
        v_raw_list.append((v1_c + v2_c) / 2.0 * 3.6)
        
    t_arr = np.array(t_raw)
    v_arr = np.array(v_raw_list)
    
    # Расчет SMA и EMA для N=10
    v_sma10 = np.empty_like(v_arr)
    v_ema10 = np.empty_like(v_arr)
    b_sma = deque(maxlen=10)
    ema_s = v_arr[0]
    alpha10 = 2.0 / (10 + 1)
    for i in range(len(v_arr)):
        b_sma.append(v_arr[i])
        v_sma10[i] = np.mean(b_sma)
        ema_s = alpha10 * v_arr[i] + (1.0 - alpha10) * ema_s
        v_ema10[i] = ema_s

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # 1. Профиль скорости
    ax1.plot(t_arr, v_arr, color="gray", alpha=0.5, lw=1.0, label="Сырая средняя V_cp (N=1)")
    ax1.plot(t_arr, v_sma10, color="blue", lw=1.8, label="SMA (N=10 точек)")
    ax1.plot(t_arr, v_ema10, color="crimson", lw=1.8, linestyle="--", label="EMA (N=10 точек, α=0.18)")
    ax1.set_ylabel("Скорость, км/ч", fontweight="bold")
    ax1.set_title(f"Сравнение сглаживания скорости: Сырая V_cp vs SMA vs EMA (заезд {bname})", fontweight="bold", fontsize=12)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right")
    
    # 2. Разность относительно сырой скорости (иллюстрация фазового запаздывания)
    ax2.plot(t_arr, v_sma10 - v_arr, color="blue", lw=1.5, label="Разность SMA - V_cp (км/ч)")
    ax2.plot(t_arr, v_ema10 - v_arr, color="crimson", lw=1.5, linestyle="--", label="Разность EMA - V_cp (км/ч)")
    ax2.axhline(0, color="black", linestyle=":", alpha=0.5)
    ax2.set_ylabel("Отклонение, км/ч", fontweight="bold")
    ax2.set_xlabel("Время от начала заезда, сек", fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right")
    
    plt.tight_layout()
    plot_file = ws_dir / "img" / "ema_vs_sma_comparison.png"
    plt.savefig(plot_file, dpi=130)
    plt.close()
    print(f"График сохранен: {plot_file}")

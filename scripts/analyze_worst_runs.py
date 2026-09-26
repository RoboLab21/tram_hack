import sys
from pathlib import Path
_repo_dir = Path(__file__).resolve().parent.parent
for _p in [str(_repo_dir), str(_repo_dir / "solution"), str(_repo_dir / "scripts")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

#!/usr/bin/env python3
"""
Скрипт детального анализа худших прогонов решения резервной одометрии.

Извлекает:
- Скорости тележек: v1 (передняя) и v2 (задняя)
- Изменения скорости: ускорения a1, a2, a_gnss (dv/dt)
- Разность скоростей тележек: |v1 - v2| (пробуксовка / юз / отказ датчика)
- Положение контроллера водителя (driver_position_cmd)
- Точные моменты времени и интервалы, в которые происходили скачки ошибки
- Причины возникновения ошибок (выпадение датчика в 0, проскальзывание, пробуксовка)
"""

import sys
import os
import argparse
import csv
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore


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


def get_top_worst_bags(summary_csv: Path, top_n: int = 5) -> List[Dict[str, Any]]:
    """Находит N худших прогонов из all_bags_summary.csv."""
    if not summary_csv.exists():
        return []
    with open(summary_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    gnss_rows = [r for r in rows if r["has_gnss"] == "YES" and float(r["r_gnss_m"]) > 100.0]
    gnss_rows.sort(key=lambda x: abs(float(x["final_diff_m"])), reverse=True)
    return gnss_rows[:top_n]


def analyze_bag_anomalies(
    bag_path: Path,
    ws_dir: Path,
    error_rate_thresh_ms: float = 1.0,     # Порог скорости накопления ошибки (м/с)
    bogie_diff_thresh_kmh: float = 3.0,     # Порог расхождения скоростей тележек (км/ч)
    save_plot: bool = True
):
    typestore = setup_typestore(ws_dir)
    bag_id = bag_path.name
    print(f"\n" + "=" * 90)
    print(f"  ДЕТАЛЬНЫЙ АНАЛИЗ ОШИБОК И АНОМАЛИЙ: {bag_id}")
    print("=" * 90)

    data = {"front": [], "rear": [], "gnss": [], "cmd": []}

    target_topics = {
        "/vehicle/front_bogie_velocity",
        "/vehicle/rear_bogie_velocity",
        "/sensing/gnss/master/vel",
        "/vehicle/driver_position_cmd"
    }

    with AnyReader([bag_path], default_typestore=typestore) as reader:
        conns = [c for c in reader.connections if c.topic in target_topics]
        for conn, timestamp, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            if t <= 0:
                t = timestamp * 1e-9

            if conn.topic == "/vehicle/front_bogie_velocity":
                data["front"].append((t, msg.velocity / 3.6))  # м/с
            elif conn.topic == "/vehicle/rear_bogie_velocity":
                data["rear"].append((t, msg.velocity / 3.6))   # м/с
            elif conn.topic == "/sensing/gnss/master/vel":
                v = np.hypot(msg.twist.linear.x, msg.twist.linear.y)
                data["gnss"].append((t, v))
            elif conn.topic == "/vehicle/driver_position_cmd":
                data["cmd"].append((t, msg.position))

    if not data["gnss"] or len(data["gnss"]) < 10:
        print(f"Прогон {bag_id} не содержит достаточных данных GNSS.")
        return

    # Синхронизация временных рядов на сетку GNSS
    t_f = np.array([x[0] for x in data["front"]])
    v_f = np.array([x[1] for x in data["front"]])
    t_r = np.array([x[0] for x in data["rear"]])
    v_r = np.array([x[1] for x in data["rear"]])
    t_g = np.array([x[0] for x in data["gnss"]])
    v_g = np.array([x[1] for x in data["gnss"]])

    # Сортировка по времени
    idx_g = np.argsort(t_g)
    t_g = t_g[idx_g]
    v_g = v_g[idx_g]

    # Интерполируем на общую сетку времени
    t_common = t_g
    v1 = np.interp(t_common, t_f, v_f)   # Передняя тележка (м/с)
    v2 = np.interp(t_common, t_r, v_r)   # Задняя тележка (м/с)
    v_cp = (v1 + v2) / 2.0               # Средняя скорость одометрии (м/с)

    # Контроллер водителя
    if data["cmd"]:
        t_c = np.array([x[0] for x in data["cmd"]])
        c_v = np.array([x[1] for x in data["cmd"]])
        idx_c = np.argsort(t_c)
        cmd_interp = np.interp(t_common, t_c[idx_c], c_v[idx_c])
    else:
        cmd_interp = np.zeros_like(t_common)

    # Приращение времени
    dt = np.diff(t_common, prepend=t_common[0])
    dt[0] = 0.0

    # Путь
    r_odom = np.cumsum(v_cp * dt)
    r_gnss = np.cumsum(v_g * dt)
    delta_r = r_odom - r_gnss            # Накопленная ошибка пути (м)

    # Ускорения a = dv/dt (м/с²)
    # Сгладим немного градиент окном для подавления дискретного шума
    a1 = np.gradient(v1, t_common)
    a2 = np.gradient(v2, t_common)
    a_cp = np.gradient(v_cp, t_common)
    a_gnss = np.gradient(v_g, t_common)

    # Разность скоростей тележек: |v1 - v2|
    v_diff = np.abs(v1 - v2)
    # Скорость накопления ошибки: d(Delta R)/dt = v_cp - v_gnss
    err_rate = v_cp - v_g

    t_start = t_common[0]
    t_rel = t_common - t_start

    # Поиск аномальных интервалов (где происходит сбой/скачок ошибки)
    # Условие аномалии: либо |v1 - v2| превышает порог, либо |v_cp - v_gnss| превышает порог
    is_anomaly = (np.abs(err_rate) >= error_rate_thresh_ms) | (v_diff * 3.6 >= bogie_diff_thresh_kmh)

    # Группировка непрерывных аномальных интервалов
    intervals: List[Dict[str, Any]] = []
    in_event = False
    start_idx = 0

    for i in range(len(is_anomaly)):
        if is_anomaly[i] and not in_event:
            in_event = True
            start_idx = i
        elif not is_anomaly[i] and in_event:
            in_event = False
            # Если событие длилось больше 1.0 сек
            if t_rel[i] - t_rel[start_idx] >= 1.0:
                intervals.append({"start_idx": start_idx, "end_idx": i})
    if in_event and (t_rel[-1] - t_rel[start_idx] >= 1.0):
        intervals.append({"start_idx": start_idx, "end_idx": len(is_anomaly) - 1})

    # Оценка каждого интервала
    events_summary = []
    for ev in intervals:
        s, e = ev["start_idx"], ev["end_idx"]
        t_s = t_rel[s]
        t_e = t_rel[e]
        dur = t_e - t_s
        err_jump = delta_r[e] - delta_r[s]
        mean_v1_kmh = float(np.mean(v1[s:e]) * 3.6)
        mean_v2_kmh = float(np.mean(v2[s:e]) * 3.6)
        mean_vg_kmh = float(np.mean(v_g[s:e]) * 3.6)
        mean_a_cp = float(np.mean(a_cp[s:e]))
        mean_a_gnss = float(np.mean(a_gnss[s:e]))
        mean_cmd = float(np.mean(cmd_interp[s:e]))
        max_v_diff_kmh = float(np.max(v_diff[s:e]) * 3.6)

        # Классификация причины сбоя
        if (mean_v1_kmh < 1.0 and mean_vg_kmh > 10.0) or (mean_v2_kmh < 1.0 and mean_vg_kmh > 10.0):
            cause = "⚠️ Выпадение датчика тележки в 0 (обрыв сигнала/юз)"
        elif max_v_diff_kmh > 10.0 and mean_cmd > 3:
            cause = "⚡ Пробуксовка при разгоне (тяга водителя)"
        elif max_v_diff_kmh > 10.0 and mean_cmd < -3:
            cause = "🛑 Проскальзывание/юз при торможении"
        elif abs(mean_v1_kmh - mean_v2_kmh) > 5.0:
            cause = "↔️ Асимметрия тележек (разность радиусов/кривая)"
        elif err_jump > 0:
            cause = "📈 Превышение скорости колес (пробуксовка)"
        else:
            cause = "📉 Недооценка скорости (скольжение/юз)"

        events_summary.append({
            "t_start": t_s,
            "t_end": t_e,
            "duration": dur,
            "err_jump": err_jump,
            "v1_kmh": mean_v1_kmh,
            "v2_kmh": mean_v2_kmh,
            "vg_kmh": mean_vg_kmh,
            "v_diff_kmh": max_v_diff_kmh,
            "acc_cp": mean_a_cp,
            "acc_gnss": mean_a_gnss,
            "cmd": mean_cmd,
            "cause": cause
        })

    # Сортируем интервалы по модулю накопленной ошибки
    events_summary.sort(key=lambda x: abs(x["err_jump"]), reverse=True)

    # -------------------------------------------------------------------------
    # ТАБЛИЦА ВЫЯВЛЕННЫХ АНОМАЛИЙ И МОМЕНТОВ ОШИБКИ
    # -------------------------------------------------------------------------
    print(f"\nИТОГИ ПРОГОНА: Путь одометрии = {r_odom[-1]:.1f} м, Путь GNSS = {r_gnss[-1]:.1f} м, Ошибка = {delta_r[-1]:+.2f} м ({abs(delta_r[-1])/r_gnss[-1]*100:.2f}%)")
    print(f"Всего выявлено аномальных интервалов: {len(events_summary)}")
    print("\n" + "=" * 135)
    print("                      МОМЕНТЫ ВРЕМЕНИ И ИНТЕРВАЛЫ ВОЗНИКНОВЕНИЯ НАИБОЛЬШЕЙ ОШИБКИ")
    print("=" * 135)
    header = (
        f"{'Интервал (сек)':^19} | {'Длит.':^7} | {'Скачок ΔR':^11} | "
        f"{'v1 (км/ч)':^10} | {'v2 (км/ч)':^10} | {'|v1-v2|':^8} | {'v_GNSS':^8} | "
        f"{'a_одом':^7} | {'Ручка':^6} | {'Причина аномалии':<35}"
    )
    print(header)
    print("-" * 135)

    for ev in events_summary[:12]:  # Топ-12 моментов
        t_range = f"{ev['t_start']:6.1f}с .. {ev['t_end']:6.1f}с"
        sign = "+" if ev['err_jump'] >= 0 else ""
        print(
            f"{t_range:^19} | {ev['duration']:5.1f}с | {f'{sign}{ev['err_jump']:.2f}м':^11} | "
            f"{ev['v1_kmh']:10.1f} | {ev['v2_kmh']:10.1f} | {ev['v_diff_kmh']:8.1f} | {ev['vg_kmh']:8.1f} | "
            f"{ev['acc_cp']:+6.2f} | {ev['cmd']:+5.0f} | {ev['cause']:<35}"
        )
    print("-" * 135)

    # -------------------------------------------------------------------------
    # ПОСТРОЕНИЕ ДИАГНОСТИЧЕСКОГО ГРАФИКА
    # -------------------------------------------------------------------------
    if save_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

        # 1. Скорости тележек и GNSS
        ax1.plot(t_rel, v1 * 3.6, label="v1: передняя тележка", color="#1f77b4", linewidth=1.2)
        ax1.plot(t_rel, v2 * 3.6, label="v2: задняя тележка", color="#ff7f0e", linewidth=1.2)
        ax1.plot(t_rel, v_g * 3.6, label="v_GNSS эталон", color="#2ca02c", linestyle="--", linewidth=1.5, alpha=0.85)
        ax1.set_ylabel("Скорость (км/ч)", fontsize=10)
        ax1.set_title(f"Диагностика худшего прогона {bag_id}: поведение тележек и ошибки", fontsize=12, fontweight="bold")
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="upper right", frameon=True)

        # Подсветка топ-3 аномальных зон
        for ev in events_summary[:3]:
            ax1.axvspan(ev["t_start"], ev["t_end"], color="red", alpha=0.18, label="Зона ошибки" if ev == events_summary[0] else "")

        # 2. Разность скоростей тележек |v1 - v2| и ускорение dv/dt
        ax2.plot(t_rel, v_diff * 3.6, label="|v1 - v2| разность колес (км/ч)", color="#d62728", linewidth=1.4)
        ax2.plot(t_rel, a_cp, label="a_одом = dv/dt (м/с²)", color="#9467bd", linewidth=1.0, alpha=0.7)
        ax2.axhline(0, color="gray", linestyle="--", linewidth=0.6)
        ax2.set_ylabel("Δv (км/ч) / a (м/с²)", fontsize=10)
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="upper right", frameon=True)

        # 3. Положение ручки водителя
        ax3.step(t_rel, cmd_interp, label="Положение контроллера водителя (driver_cmd)", color="#8c564b", linewidth=1.2)
        ax3.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax3.set_ylabel("Позиция ручки", fontsize=10)
        ax3.grid(True, linestyle=":", alpha=0.6)
        ax3.legend(loc="upper right", frameon=True)

        # 4. Накопление ошибки пути Delta R(t)
        ax4.plot(t_rel, delta_r, label="Накопленная ошибка ΔR = R_одом - R_gnss (м)", color="#e377c2", linewidth=1.6)
        ax4.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax4.set_xlabel("Время от начала прогона (сек)", fontsize=11)
        ax4.set_ylabel("Ошибка пути ΔR (м)", fontsize=10)
        ax4.grid(True, linestyle=":", alpha=0.6)
        ax4.legend(loc="lower left", frameon=True)

        plt.tight_layout()
        plot_path = ws_dir / f"anomaly_analysis_{bag_id}.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"📊 Диагностический график сохранен в файл: {plot_path.name}")

    return events_summary


def main():
    parser = argparse.ArgumentParser(
        description="Анализ худших прогонов одометрии и выявление моментов аномалий."
    )
    parser.add_argument(
        "--bag",
        type=str,
        default=None,
        help="Конкретный bag для анализа (по умолчанию анализирует топ худших)"
    )
    parser.add_argument(
        "--top",
        type=int,
        default=3,
        help="Количество худших прогонов для детального анализа (по умолчанию 3)"
    )

    args = parser.parse_args()
    ws_dir = Path(__file__).resolve().parent
    data_dir = ws_dir / "data"
    summary_csv = ws_dir / "all_bags_summary.csv"

    if args.bag:
        bag_path = Path(args.bag)
        if not bag_path.is_absolute():
            bag_path = data_dir / bag_path.name
        analyze_bag_anomalies(bag_path, ws_dir)
    else:
        worst = get_top_worst_bags(summary_csv, top_n=args.top)
        if not worst:
            print("all_bags_summary.csv не найден или пуст.")
            return

        print(f"\n================================================================================")
        print(f"      ТОП-{len(worst)} ХУДШИХ ПРОГОНОВ ПО РЕШЕНИЮ R = sum(V_cp * dt)")
        print(f"================================================================================")
        for i, r in enumerate(worst, 1):
            print(
                f"{i}. {r['bag_id']}: Ошибка = {float(r['final_diff_m']):+7.2f} м "
                f"({float(r['final_rel_err_pct']):5.2f}%), Макс. дрейф = {float(r['max_err_m']):6.2f} м, "
                f"Дистанция = {float(r['r_gnss_m']):.1f} м, Длительность = {float(r['duration_sec'])/60:.1f} мин"
            )

        for r in worst:
            bag_path = data_dir / r["bag_id"]
            analyze_bag_anomalies(bag_path, ws_dir)


if __name__ == "__main__":
    main()

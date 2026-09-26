import sys
from pathlib import Path
_repo_dir = Path(__file__).resolve().parent.parent
for _p in [str(_repo_dir), str(_repo_dir / "solution"), str(_repo_dir / "scripts")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

#!/usr/bin/env python3
"""
Скрипт прямого сравнения базовой ноды и новой фильтрованной ноды (v2):
- Базовая нода: R_баз = sum((v1+v2)/2 * dt)
- Фильтрованная нода: R_фильтр = sum(v_est * dt) с детекцией нулей и выбросов через (v1 - v2)^2 и (dv/dt)^2

Выводит:
1. Сравнительную таблицу прогона: Базовая vs Фильтрованная vs GNSS.
2. Статистику выявленных аномалий (выпадения в 0, выбросы, асимметрия).
3. Сохраняет сравнительный график.
"""

import sys
import os
import argparse
from pathlib import Path
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

from odometry_node import DeadReckoningNode
from odometry_node_filtered import FilteredDeadReckoningNode


def setup_typestore(ws_dir: Path):
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    msg_dir = ws_dir / "tram_vehicle_msgs" / "msg"
    typestore.register(get_types_from_msg((msg_dir / "VelocitySensor.msg").read_text(), "tram_vehicle_msgs/msg/VelocitySensor"))
    typestore.register(get_types_from_msg((msg_dir / "DriverControllerCommand.msg").read_text(), "tram_vehicle_msgs/msg/DriverControllerCommand"))
    return typestore


def compare_nodes_on_bag(bag_path: Path, ws_dir: Path, step_dist_m: float = 500.0, save_plot: bool = True):
    typestore = setup_typestore(ws_dir)
    bag_id = bag_path.name

    print("\n" + "=" * 90)
    print(f"  СРАВНЕНИЕ БАЗОВОЙ И ФИЛЬТРОВАННОЙ НОДЫ: {bag_id}")
    print("=" * 90)

    base_node = DeadReckoningNode(input_in_kmh=True, integration_method="trapezoidal")
    filt_node = FilteredDeadReckoningNode(input_in_kmh=True, integration_method="trapezoidal")

    gnss_records = []
    fault_counts = {"ZERO_V1": 0, "ZERO_V2": 0, "SPIKE_V1": 0, "SPIKE_V2": 0, "EXCESSIVE_DIFF": 0}

    target_topics = {
        "/vehicle/front_bogie_velocity",
        "/vehicle/rear_bogie_velocity",
        "/sensing/gnss/master/vel"
    }

    with AnyReader([bag_path], default_typestore=typestore) as reader:
        conns = [c for c in reader.connections if c.topic in target_topics]
        for conn, timestamp, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            if t <= 0: t = timestamp * 1e-9

            if conn.topic == "/vehicle/front_bogie_velocity":
                base_node.update_front(t, msg.velocity)
                st = filt_node.update_front(t, msg.velocity)
                if st and st.fault_type != "OK":
                    fault_counts[st.fault_type] = fault_counts.get(st.fault_type, 0) + 1
            elif conn.topic == "/vehicle/rear_bogie_velocity":
                base_node.update_rear(t, msg.velocity)
                st = filt_node.update_rear(t, msg.velocity)
                if st and st.fault_type != "OK":
                    fault_counts[st.fault_type] = fault_counts.get(st.fault_type, 0) + 1
            elif conn.topic == "/sensing/gnss/master/vel":
                v = np.hypot(msg.twist.linear.x, msg.twist.linear.y)
                gnss_records.append((t, v))

    base_data = base_node.get_arrays()
    filt_data = filt_node.get_arrays()

    if not base_data or not filt_data:
        print("Ошибка: данные не получены.")
        return

    t_b = base_data["timestamp"]
    r_base = base_data["distance"]
    v_base = base_data["v_avg"]

    t_f = filt_data["timestamp"]
    r_filt = filt_data["distance"]
    v_filt = filt_data["v_est"]

    has_gnss = len(gnss_records) > 10

    if not has_gnss:
        print("\n[Внимание] В данном bag-файле нет GNSS.")
        print(f"Пройденный путь базовой ноды: {r_base[-1]:.2f} м")
        print(f"Пройденный путь фильтрованной: {r_filt[-1]:.2f} м")
        print(f"Разница между нодами: {r_filt[-1] - r_base[-1]:+.2f} м")
        return

    t_g = np.array([x[0] for x in gnss_records])
    v_g = np.array([x[1] for x in gnss_records])
    idx_g = np.argsort(t_g)
    t_g, v_g = t_g[idx_g], v_g[idx_g]

    # Интерполяция GNSS на моменты времени
    v_g_interp = np.interp(t_f, t_g, v_g)
    dt_f = np.diff(t_f, prepend=t_f[0])
    dt_f[0] = 0.0
    r_gnss = np.cumsum(v_g_interp * dt_f)

    final_rgb = r_gnss[-1]
    final_base = r_base[-1]
    final_filt = r_filt[-1]

    err_base = final_base - final_rgb
    err_filt = final_filt - final_rgb

    rel_base = abs(err_base) / final_rgb * 100.0
    rel_filt = abs(err_filt) / final_rgb * 100.0

    mae_base = np.mean(np.abs(r_base - r_gnss))
    mae_filt = np.mean(np.abs(r_filt - r_gnss))

    # -------------------------------------------------------------------------
    # СРАВНИТЕЛЬНАЯ ТАБЛИЦА МЕТРИК
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("                  СРАВНЕНИЕ ТОЧНОСТИ РЕШЕНИЙ С ЭТАЛОНОМ GNSS")
    print("=" * 80)
    print(f" {'Показатель':<36} | {'Базовая нода':^18} | {'Фильтрованная (v2)':^18}")
    print("-" * 80)
    print(f" {'Пройденный путь R (м)':<36} | {final_base:16.2f} м | {final_filt:16.2f} м")
    print(f" {'Путь по эталону GNSS (м)':<36} | {final_rgb:16.2f} м | {final_rgb:16.2f} м")
    print(f" {'Финальная ошибка пути ΔR':<36} | {err_base:+16.2f} м | {err_filt:+16.2f} м")
    print(f" {'Погрешность относительно ВСЕГО пути':<36} | {rel_base:15.3f} % | {rel_filt:15.3f} %")
    print(f" {'Средняя ошибка по трассе (MAE)':<36} | {mae_base:16.2f} м | {mae_filt:16.2f} м")
    print("=" * 80)

    print("\nДетекция аномалий датчиков через квадратичное отклонение:")
    print(f" - Выпадений передней тележки в ноль (ZERO_V1): {fault_counts.get('ZERO_V1', 0)}")
    print(f" - Выпадений задней тележки в ноль (ZERO_V2):   {fault_counts.get('ZERO_V2', 0)}")
    print(f" - Выбросов / аномальных скачков скорости (SPIKE): {fault_counts.get('SPIKE_V1', 0) + fault_counts.get('SPIKE_V2', 0)}")
    print(f" - Превышений порога квадратичной разности:       {fault_counts.get('EXCESSIVE_DIFF', 0)}")

    # -------------------------------------------------------------------------
    # СРАВНИТЕЛЬНАЯ ТАБЛИЦА ПО УЧАСТКАМ
    # -------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print(f"               ТАБЛИЦА ПО УЧАСТКАМ: БАЗОВАЯ vs ФИЛЬТРОВАННАЯ (шаг ~{step_dist_m:.0f} м)")
    print("=" * 115)
    header = (
        f"{'Участок':^9} | {'Время':^9} | {'R GNSS':^10} | "
        f"{'R Базовая':^12} | {'ΔR Баз (м)':^11} | {'% Баз':^8} | "
        f"{'R Фильтр':^12} | {'ΔR Фил (м)':^11} | {'% Фил':^8}"
    )
    print(header)
    print("-" * 115)

    checkpoints = np.arange(step_dist_m, final_rgb, step_dist_m)
    if len(checkpoints) == 0 or checkpoints[-1] < final_rgb - (step_dist_m * 0.2):
        checkpoints = np.append(checkpoints, final_rgb)

    t_start = t_f[0]

    for idx, cp in enumerate(checkpoints, 1):
        k = min(np.searchsorted(r_gnss, cp), len(r_gnss) - 1)
        t_cur = t_f[k] - t_start
        rg = r_gnss[k]
        rb = r_base[k]
        rf = r_filt[k]
        db = rb - rg
        df = rf - rg
        pct_b = abs(db) / final_rgb * 100.0
        pct_f = abs(df) / final_rgb * 100.0

        sb = "+" if db >= 0 else ""
        sf = "+" if df >= 0 else ""

        print(
            f"{f'#{idx}':^9} | {t_cur:7.1f} с | {rg:9.1f}м | "
            f"{rb:11.1f}м | {f'{sb}{db:.2f}':^11} | {pct_b:6.2f}% | "
            f"{rf:11.1f}м | {f'{sf}{df:.2f}':^11} | {pct_f:6.2f}%"
        )
    print("-" * 115)

    # -------------------------------------------------------------------------
    # ПОСТРОЕНИЕ СРАВНИТЕЛЬНОГО ГРАФИКА
    # -------------------------------------------------------------------------
    if save_plot:
        t_rel = t_f - t_start
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

        # 1. Скорости
        ax1.plot(t_rel, v_base * 3.6, label="V_базовая = (v1+v2)/2", color="#1f77b4", linewidth=1.4, alpha=0.7)
        ax1.plot(t_rel, v_filt * 3.6, label="V_фильтрованная (v2)", color="#d62728", linewidth=1.6)
        ax1.plot(t_rel, v_g_interp * 3.6, label="v_GNSS эталон", color="#2ca02c", linestyle="--", linewidth=1.8, alpha=0.9)
        ax1.set_ylabel("Скорость (км/ч)", fontsize=11)
        ax1.set_title(f"Сравнение базовой и фильтрованной ноды одометрии ({bag_id})", fontsize=13, fontweight="bold")
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="upper right", frameon=True)

        # 2. Ошибки пути во времени
        ax2.plot(t_rel, r_base - r_gnss, label="Ошибка базовой ноды ΔR_баз (м)", color="#1f77b4", linewidth=1.6)
        ax2.plot(t_rel, r_filt - r_gnss, label="Ошибка фильтрованной ноды ΔR_фильтр (м)", color="#d62728", linewidth=2.0)
        ax2.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax2.set_xlabel("Время от начала прогона (сек)", fontsize=11)
        ax2.set_ylabel("Ошибка пути ΔR (м)", fontsize=11)
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="lower left", frameon=True)

        img_dir = ws_dir / "img"
        img_dir.mkdir(exist_ok=True)
        out_png = img_dir / f"compare_{bag_id}.png"
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"📊 Сравнительный график сохранен: {out_png.name}")


def main():
    parser = argparse.ArgumentParser(description="Сравнение базовой и фильтрованной одометрии.")
    parser.add_argument("bag_path", type=str, nargs="?", default="data/30639_3b3d9eb8", help="Путь к bag")
    parser.add_argument("--step", type=float, default=500.0, help="Шаг участков (м)")
    args = parser.parse_args()

    ws_dir = _repo_dir
    bag_path = Path(args.bag_path)
    if not bag_path.is_absolute():
        bag_path = ws_dir / bag_path

    compare_nodes_on_bag(bag_path, ws_dir, step_dist_m=args.step)


if __name__ == "__main__":
    main()

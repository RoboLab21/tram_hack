import sys
from pathlib import Path
_repo_dir = Path(__file__).resolve().parent.parent
for _p in [str(_repo_dir), str(_repo_dir / "solution"), str(_repo_dir / "scripts")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

#!/usr/bin/env python3
"""
Скрипт детального анализа и визуализации моментов проскальзывания:
- Графики скоростей: v1 (передняя тележка), v2 (задняя тележка), v_cp (средняя), v_GNSS (эталон).
- Производные скоростей (ускорения): a1 = dv1/dt, a2 = dv2/dt, a_cp, a_GNSS.
- Скорость проскальзывания Delta v_slip = v_колес - v_GNSS.
- Команда контроллера водителя (тяга / торможение).
"""

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


def setup_typestore(ws_dir: Path):
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    msg_dir = ws_dir / "tram_vehicle_msgs" / "msg"
    vel_msg_path = msg_dir / "VelocitySensor.msg"
    cmd_msg_path = msg_dir / "DriverControllerCommand.msg"
    if vel_msg_path.exists():
        typestore.register(get_types_from_msg(vel_msg_path.read_text(), "tram_vehicle_msgs/msg/VelocitySensor"))
    if cmd_msg_path.exists():
        typestore.register(get_types_from_msg(cmd_msg_path.read_text(), "tram_vehicle_msgs/msg/DriverControllerCommand"))
    return typestore


def load_bag_telemetry(bag_path: Path, typestore):
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
                data["gnss"].append((t, v))                    # м/с
            elif conn.topic == "/vehicle/driver_position_cmd":
                data["cmd"].append((t, msg.position))

    t_f = np.array([x[0] for x in data["front"]])
    v_f = np.array([x[1] for x in data["front"]])
    t_r = np.array([x[0] for x in data["rear"]])
    v_r = np.array([x[1] for x in data["rear"]])
    t_g = np.array([x[0] for x in data["gnss"]])
    v_g = np.array([x[1] for x in data["gnss"]])

    idx_g = np.argsort(t_g)
    t_g = t_g[idx_g]
    v_g = v_g[idx_g]

    t_common = t_g
    v1 = np.interp(t_common, t_f, v_f)
    v2 = np.interp(t_common, t_r, v_r)
    v_cp = (v1 + v2) / 2.0

    if data["cmd"]:
        t_c = np.array([x[0] for x in data["cmd"]])
        c_v = np.array([x[1] for x in data["cmd"]])
        idx_c = np.argsort(t_c)
        cmd = np.interp(t_common, t_c[idx_c], c_v[idx_c])
    else:
        cmd = np.zeros_like(t_common)

    # Производные скоростей (ускорения dv/dt в м/с²)
    # Применяем центральные разности
    a1 = np.gradient(v1, t_common)
    a2 = np.gradient(v2, t_common)
    a_cp = np.gradient(v_cp, t_common)
    a_gnss = np.gradient(v_g, t_common)

    t_rel = t_common - t_common[0]
    slip_speed = (v_cp - v_g) * 3.6  # км/ч

    return {
        "t": t_rel,
        "v1_kmh": v1 * 3.6,
        "v2_kmh": v2 * 3.6,
        "v_cp_kmh": v_cp * 3.6,
        "v_gnss_kmh": v_g * 3.6,
        "a1": a1,
        "a2": a2,
        "a_cp": a_cp,
        "a_gnss": a_gnss,
        "slip_kmh": slip_speed,
        "cmd": cmd,
    }


def plot_slip_event(
    telemetry: dict,
    t_start: float,
    t_end: float,
    title: str,
    output_png: Path,
    slip_type: str = "spin"
):
    """
    Строит подробный 4-панельный график момента проскальзывания в заданном окне [t_start, t_end].
    """
    t = telemetry["t"]
    mask = (t >= t_start) & (t <= t_end)
    if not np.any(mask):
        print(f"Нет данных в интервале [{t_start}, {t_end}]")
        return

    sub_t = t[mask]
    v1 = telemetry["v1_kmh"][mask]
    v2 = telemetry["v2_kmh"][mask]
    v_cp = telemetry["v_cp_kmh"][mask]
    v_g = telemetry["v_gnss_kmh"][mask]
    a1 = telemetry["a1"][mask]
    a2 = telemetry["a2"][mask]
    a_cp = telemetry["a_cp"][mask]
    a_g = telemetry["a_gnss"][mask]
    slip = telemetry["slip_kmh"][mask]
    cmd = telemetry["cmd"][mask]

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True, gridspec_kw={"height_ratios": [2.5, 2.2, 1.6, 1.2]})

    # 1. График скоростей
    ax1 = axes[0]
    ax1.plot(sub_t, v1, label="v₁: передняя тележка (колёса)", color="#1f77b4", linewidth=2.0)
    ax1.plot(sub_t, v2, label="v₂: задняя тележка (колёса)", color="#ff7f0e", linewidth=2.0)
    ax1.plot(sub_t, v_cp, label="V_ср = (v₁+v₂)/2", color="#9467bd", linestyle="-.", linewidth=1.5)
    ax1.plot(sub_t, v_g, label="v_GNSS: реальная скорость трамвая", color="#2ca02c", linestyle="--", linewidth=2.2)

    # Закраска зоны расхождения
    highlight_color = "#ffcccc" if slip_type == "spin" else "#cce5ff"
    ax1.fill_between(sub_t, v_cp, v_g, color=highlight_color, alpha=0.6, label="Зона расхождения (проскальзывание)")

    ax1.set_ylabel("Скорость (км/ч)", fontsize=11, fontweight="bold")
    ax1.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax1.grid(True, linestyle=":", alpha=0.7)
    ax1.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)

    # 2. Производные скоростей (ускорения a = dv/dt)
    ax2 = axes[1]
    ax2.plot(sub_t, a1, label="a₁ = dv₁/dt (передняя)", color="#1f77b4", linewidth=1.6, alpha=0.9)
    ax2.plot(sub_t, a2, label="a₂ = dv₂/dt (задняя)", color="#ff7f0e", linewidth=1.6, alpha=0.9)
    ax2.plot(sub_t, a_g, label="a_GNSS = dv_GNSS/dt (истинное)", color="#2ca02c", linestyle="--", linewidth=2.0)
    ax2.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax2.set_ylabel("Ускорение dv/dt (м/с²)", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.7)
    ax2.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)

    # 3. Скорость проскальзывания Delta v_slip = v_колес - v_GNSS
    ax3 = axes[2]
    slip_color = "#d62728" if slip_type == "spin" else "#17becf"
    ax3.plot(sub_t, slip, label="Δv_slip = V_колёс - V_GNSS (км/ч)", color=slip_color, linewidth=2.0)
    ax3.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax3.fill_between(sub_t, 0, slip, where=(slip >= 0), color="#ff9999", alpha=0.4, label="Пробуксовка (колёса быстрее)")
    ax3.fill_between(sub_t, 0, slip, where=(slip < 0), color="#99ccff", alpha=0.4, label="Юз / блокировка (колёса медленнее)")
    ax3.set_ylabel("Проскальзывание (км/ч)", fontsize=11, fontweight="bold")
    ax3.grid(True, linestyle=":", alpha=0.7)
    ax3.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)

    # 4. Положение контроллера водителя (driver_cmd)
    ax4 = axes[3]
    ax4.step(sub_t, cmd, label="Положение контроллера водителя (тяга > 0, тормоз < 0)", color="#8c564b", linewidth=1.8)
    ax4.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax4.set_xlabel("Время от начала записи (сек)", fontsize=12, fontweight="bold")
    ax4.set_ylabel("Ручка", fontsize=11, fontweight="bold")
    ax4.grid(True, linestyle=":", alpha=0.7)
    ax4.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(output_png, dpi=160)
    plt.close(fig)
    print(f"📊 График проскальзывания сохранен: {output_png.name}")


def main():
    ws_dir = Path(__file__).resolve().parent
    typestore = setup_typestore(ws_dir)

    # -------------------------------------------------------------------------
    # СОБЫТИЕ 1: Чистая пробуксовка на месте при разгоне (30618_4d487b0d)
    # Колеса раскручиваются до 26 км/ч, трамвай стоит на месте (v_GNSS = 0)
    # -------------------------------------------------------------------------
    bag1 = ws_dir / "data" / "30618_4d487b0d"
    if bag1.exists():
        print("\nЗагрузка телеметрии события 1 (пробуксовка на месте)...")
        tel1 = load_bag_telemetry(bag1, typestore)
        plot_slip_event(
            telemetry=tel1,
            t_start=90.0,
            t_end=150.0,
            title="Событие 1: Пробуксовка колёс на месте при трогании с остановки (прогон 30618_4d487b0d)\nКолёса v₁=v₂ вращаются до 26 км/ч, а трамвай стоит на месте (v_GNSS=0)",
            output_png=ws_dir / "slip_event_1_wheel_spin.png",
            slip_type="spin"
        )

    # -------------------------------------------------------------------------
    # СОБЫТИЕ 2: Юз / выпадение датчика задней тележки (30639_3b3d9eb8)
    # Задняя тележка падает в 0, передняя и GNSS едут со скоростью 17-25 км/ч
    # -------------------------------------------------------------------------
    bag2 = ws_dir / "data" / "30639_3b3d9eb8"
    if bag2.exists():
        print("\nЗагрузка телеметрии события 2 (юз / отказ датчика тележки)...")
        tel2 = load_bag_telemetry(bag2, typestore)
        plot_slip_event(
            telemetry=tel2,
            t_start=360.0,
            t_end=480.0,
            title="Событие 2: Асимметрия / юз задней тележки (прогон 30639_3b3d9eb8)\nЗадняя тележка v₂ падает в 0, передняя v₁ и GNSS едут на 17-25 км/ч (ошибка -135 м)",
            output_png=ws_dir / "slip_event_2_wheel_lock.png",
            slip_type="lock"
        )

    # -------------------------------------------------------------------------
    # СОБЫТИЕ 3: Динамическое проскальзывание в движении на кривой (30639_dce52be4)
    # Расхождение скоростей колес и GNSS при интенсивном разгоне и торможении
    # -------------------------------------------------------------------------
    bag3 = ws_dir / "data" / "30639_dce52be4"
    if bag3.exists():
        print("\nЗагрузка телеметрии события 3 (динамическое проскальзывание)...")
        tel3 = load_bag_telemetry(bag3, typestore)
        plot_slip_event(
            telemetry=tel3,
            t_start=450.0,
            t_end=580.0,
            title="Событие 3: Динамическое проскальзывание колёс при разгоне и торможении (прогон 30639_dce52be4)\nОтклонение скорости тележек от реальной скорости движения трамвая",
            output_png=ws_dir / "slip_event_3_dynamic_slip.png",
            slip_type="spin"
        )


if __name__ == "__main__":
    main()

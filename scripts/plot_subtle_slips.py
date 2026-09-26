import sys
from pathlib import Path
_repo_dir = Path(__file__).resolve().parent.parent
for _p in [str(_repo_dir), str(_repo_dir / "solution"), str(_repo_dir / "scripts")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

#!/usr/bin/env python3
"""
Скрипт детальной визуализации реальных умеренных (физических) проскальзываний в движении:
1. Динамическая пробуксовка при интенсивном разгоне (тяга +15, a ≈ +1.0 м/с²).
2. Динамический юз при служебном торможении (тормоз -11, a ≈ -1.3 м/с²).
3. Дифференциальное скольжение тележек на радиусной кривой (разность v1 и v2).
"""

import os
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
    typestore.register(get_types_from_msg((msg_dir / "VelocitySensor.msg").read_text(), "tram_vehicle_msgs/msg/VelocitySensor"))
    typestore.register(get_types_from_msg((msg_dir / "DriverControllerCommand.msg").read_text(), "tram_vehicle_msgs/msg/DriverControllerCommand"))
    return typestore


def load_telemetry(bag_path: Path, typestore):
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
            if t <= 0: t = timestamp * 1e-9

            if conn.topic == "/vehicle/front_bogie_velocity":
                data["front"].append((t, msg.velocity / 3.6))
            elif conn.topic == "/vehicle/rear_bogie_velocity":
                data["rear"].append((t, msg.velocity / 3.6))
            elif conn.topic == "/sensing/gnss/master/vel":
                v = np.hypot(msg.twist.linear.x, msg.twist.linear.y)
                data["gnss"].append((t, v))
            elif conn.topic == "/vehicle/driver_position_cmd":
                data["cmd"].append((t, msg.position))

    t_f, v_f = np.array([x[0] for x in data["front"]]), np.array([x[1] for x in data["front"]])
    t_r, v_r = np.array([x[0] for x in data["rear"]]), np.array([x[1] for x in data["rear"]])
    t_g, v_g = np.array([x[0] for x in data["gnss"]]), np.array([x[1] for x in data["gnss"]])

    idx_g = np.argsort(t_g)
    t_g, v_g = t_g[idx_g], v_g[idx_g]

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

    # Производные (ускорения) в м/с²
    a1 = np.gradient(v1, t_common)
    a2 = np.gradient(v2, t_common)
    a_cp = np.gradient(v_cp, t_common)
    a_gnss = np.gradient(v_g, t_common)

    t_rel = t_common - t_common[0]
    slip_kmh = (v_cp - v_g) * 3.6

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
        "slip_kmh": slip_kmh,
        "cmd": cmd,
    }


def plot_dynamic_slip_maneuver(tel: dict, t_start: float, t_end: float, title: str, output_png: Path):
    t = tel["t"]
    mask = (t >= t_start) & (t <= t_end)
    sub_t = t[mask]

    v1 = tel["v1_kmh"][mask]
    v2 = tel["v2_kmh"][mask]
    v_cp = tel["v_cp_kmh"][mask]
    v_g = tel["v_gnss_kmh"][mask]
    a_cp = tel["a_cp"][mask]
    a_g = tel["a_gnss"][mask]
    slip = tel["slip_kmh"][mask]
    cmd = tel["cmd"][mask]

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 12), sharex=True, gridspec_kw={"height_ratios": [2.5, 2.2, 1.6, 1.2]})

    # 1. СКОРОСТИ
    ax1.plot(sub_t, v1, label="v₁: передняя тележка", color="#1f77b4", linewidth=1.8)
    ax1.plot(sub_t, v2, label="v₂: задняя тележка", color="#ff7f0e", linewidth=1.8, linestyle=":")
    ax1.plot(sub_t, v_cp, label="V_ср одометрии", color="#9467bd", linewidth=2.0)
    ax1.plot(sub_t, v_g, label="v_GNSS: истинная скорость", color="#2ca02c", linestyle="--", linewidth=2.2)

    # Подсветка фаз: разгон (желтая/красная) и торможение (синяя)
    ax1.axvspan(175.0, 185.0, color="#ff9999", alpha=0.25, label="Фаза тяги (микропробуксовка)")
    ax1.axvspan(186.0, 193.0, color="#cccccc", alpha=0.2, label="Фаза выбега (нулевое скольжение)")
    ax1.axvspan(194.0, 204.0, color="#99ccff", alpha=0.25, label="Фаза торможения (микроюз)")

    ax1.set_ylabel("Скорость (км/ч)", fontsize=11, fontweight="bold")
    ax1.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax1.grid(True, linestyle=":", alpha=0.7)
    ax1.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9, ncol=2)

    # 2. ПРОИЗВОДНЫЕ СКОРОСТЕЙ (УСКОРЕНИЯ dv/dt)
    ax2.plot(sub_t, a_cp, label="a_одом = dv_колёс/dt", color="#9467bd", linewidth=2.0)
    ax2.plot(sub_t, a_g, label="a_GNSS = dv_вагона/dt (истинное)", color="#2ca02c", linestyle="--", linewidth=2.0)
    ax2.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax2.set_ylabel("Ускорение dv/dt (м/с²)", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.7)
    ax2.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)

    # 3. ВЕЛИЧИНА ПРОСКАЛЬЗЫВАНИЯ Δv_slip
    ax3.plot(sub_t, slip, label="Δv_slip = V_колёс - V_GNSS (км/ч)", color="#d62728", linewidth=2.0)
    ax3.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax3.fill_between(sub_t, 0, slip, where=(slip >= 0), color="#ff9999", alpha=0.5, label="Пробуксовка под тягой (+0.3…+0.7 км/ч)")
    ax3.fill_between(sub_t, 0, slip, where=(slip < 0), color="#99ccff", alpha=0.5, label="Юз при торможении (-0.3…-0.6 км/ч)")
    ax3.set_ylabel("Скольжение (км/ч)", fontsize=11, fontweight="bold")
    ax3.grid(True, linestyle=":", alpha=0.7)
    ax3.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)

    # 4. РУЧКА КОНТРОЛЛЕРА ВОДИТЕЛЯ
    ax4.step(sub_t, cmd, label="Положение ручки водителя (тяга > 0, тормоз < 0)", color="#8c564b", linewidth=2.0)
    ax4.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax4.set_xlabel("Время от начала записи (сек)", fontsize=12, fontweight="bold")
    ax4.set_ylabel("Ручка", fontsize=11, fontweight="bold")
    ax4.grid(True, linestyle=":", alpha=0.7)
    ax4.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(output_png, dpi=160)
    plt.close(fig)
    print(f"📊 График реального маневра сохранен: {output_png.name}")


def plot_curve_differential(tel: dict, t_start: float, t_end: float, title: str, output_png: Path):
    t = tel["t"]
    mask = (t >= t_start) & (t <= t_end)
    sub_t = t[mask]

    v1 = tel["v1_kmh"][mask]
    v2 = tel["v2_kmh"][mask]
    v_cp = tel["v_cp_kmh"][mask]
    v_g = tel["v_gnss_kmh"][mask]
    a1 = tel["a1"][mask]
    a2 = tel["a2"][mask]
    b_diff = v1 - v2
    cmd = tel["cmd"][mask]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True, gridspec_kw={"height_ratios": [2.5, 2.0, 1.2]})

    # 1. Скорости
    ax1.plot(sub_t, v1, label="v₁: передняя тележка (внешняя/внутренняя)", color="#1f77b4", linewidth=2.0)
    ax1.plot(sub_t, v2, label="v₂: задняя тележка", color="#ff7f0e", linewidth=2.0)
    ax1.plot(sub_t, v_g, label="v_GNSS: истинная скорость центра вагона", color="#2ca02c", linestyle="--", linewidth=2.2)
    ax1.set_ylabel("Скорость (км/ч)", fontsize=11, fontweight="bold")
    ax1.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax1.grid(True, linestyle=":", alpha=0.7)
    ax1.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)

    # 2. Разность скоростей тележек и производные
    ax2.plot(sub_t, b_diff, label="Дифференциал скоростей v₁ - v₂ (км/ч)", color="#d62728", linewidth=2.0)
    ax2.plot(sub_t, a1, label="a₁ = dv₁/dt", color="#1f77b4", linestyle=":", alpha=0.8)
    ax2.plot(sub_t, a2, label="a₂ = dv₂/dt", color="#ff7f0e", linestyle=":", alpha=0.8)
    ax2.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax2.set_ylabel("Разность (км/ч) / a (м/с²)", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.7)
    ax2.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)

    # 3. Ручка
    ax3.step(sub_t, cmd, label="Положение контроллера водителя", color="#8c564b", linewidth=1.8)
    ax3.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax3.set_xlabel("Время от начала записи (сек)", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Ручка", fontsize=11, fontweight="bold")
    ax3.grid(True, linestyle=":", alpha=0.7)
    ax3.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(output_png, dpi=160)
    plt.close(fig)
    print(f"📊 График дифференциала кривой сохранен: {output_png.name}")


def main():
    ws_dir = Path(__file__).resolve().parent
    typestore = setup_typestore(ws_dir)

    # 1. Чистый динамический маневр: разгон под полной тягой + выбег + служебное торможение
    bag1 = ws_dir / "data" / "30618_1cc230fa"
    if bag1.exists():
        tel1 = load_telemetry(bag1, typestore)
        plot_dynamic_slip_maneuver(
            tel=tel1,
            t_start=174.0,
            t_end=209.0,
            title="Реальное динамическое микропроскальзывание в движении (прогон 30618_1cc230fa)\nРазгон под тягой (+15, slip +0.7 км/ч) → Выбег (slip 0 км/ч) → Служебное торможение (-11, slip -0.6 км/ч)",
            output_png=ws_dir / "realistic_slip_maneuver.png"
        )

    # 2. Дифференциальное скольжение тележек на кривой
    bag2 = ws_dir / "data" / "30618_0e41eac3"
    if bag2.exists():
        tel2 = load_telemetry(bag2, typestore)
        plot_curve_differential(
            tel=tel2,
            t_start=120.0,
            t_end=145.0,
            title="Дифференциальное скольжение тележек на криволинейном участке пути (прогон 30618_0e41eac3)\nРасхождение скоростей v₁ и v₂ при входе в поворот под тягой (+11)",
            output_png=ws_dir / "realistic_curve_differential.png"
        )


if __name__ == "__main__":
    main()

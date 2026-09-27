#!/usr/bin/env python3
"""
Скрипт параллельной оценки динамической модели для 3 масс: 24.0 т, 24.5 т, 25.0 т
по всем 56 валидным багам с качественным GNSS (щук-талл и талл-щук).

Чтение каждого bag-файла выполняется один раз, параллельно обновляя 3 экземпляра модели.
"""

import sys
from pathlib import Path

# Пути
_current_dir = Path(__file__).resolve().parent
_repo_dir = _current_dir.parent
for p in [str(_repo_dir), str(_current_dir), str(_repo_dir / "scripts"), str(_repo_dir / "solution")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import csv
import numpy as np
from rosbags.highlevel import AnyReader
from batch_evaluate_filtered import setup_typestore

from tram_dynamic_model import TramParameters
from dynamic_odometry_node import DynamicOdometryNode

ws_dir = _repo_dir
typestore = setup_typestore(ws_dir)

geom_files = {
    "щук-талл": ws_dir / "pathgrath" / "щукинская - таллинская.json",
    "талл-щук": ws_dir / "pathgrath" / "таллинская - щукинская.json",
}

target_topics = {
    "/vehicle/driver_position_cmd",
    "/vehicle/front_bogie_velocity",
    "/vehicle/rear_bogie_velocity",
    "/sensing/gnss/master/vel"
}


def process_bag_multi_mass(bag_path: Path, route_name: str, masses=(24000.0, 24500.0, 25000.0)):
    path_file = geom_files[route_name]
    
    # Создаем 3 экземпляра модели под каждую массу
    nodes = {
        m: DynamicOdometryNode(path_geometry_file_or_data=path_file, tram_params=TramParameters(mass_kg=m))
        for m in masses
    }

    gnss_vel_records = []
    bogie_vel_records = []
    front_v_last = 0.0
    rear_v_last = 0.0

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

                if conn.topic == "/vehicle/driver_position_cmd":
                    for node in nodes.values():
                        node.update_driver_cmd(t, msg.position)
                elif conn.topic == "/vehicle/front_bogie_velocity":
                    front_v_last = float(msg.velocity) / 3.6
                    bogie_vel_records.append((t, 0.5 * (front_v_last + rear_v_last)))
                elif conn.topic == "/vehicle/rear_bogie_velocity":
                    rear_v_last = float(msg.velocity) / 3.6
                    bogie_vel_records.append((t, 0.5 * (front_v_last + rear_v_last)))
                elif conn.topic == "/sensing/gnss/master/vel":
                    vx = msg.twist.linear.x
                    vy = msg.twist.linear.y
                    v_gnss = float(np.hypot(vx, vy))
                    gnss_vel_records.append((t, v_gnss))
    except Exception as e:
        print(f"Ошибка чтения {bag_path.name}: {e}")
        return None

    if len(gnss_vel_records) < 20:
        return None

    # Берем временную сетку первой модели (у всех моделей временные сетки идентичны)
    data0 = nodes[masses[0]].get_arrays()
    if not data0 or len(data0["timestamp"]) < 10:
        return None

    t_m = data0["timestamp"]
    duration_sec = float(t_m[-1] - t_m[0])
    if duration_sec < 10.0:
        return None

    # GNSS интерполяция
    t_g = np.array([x[0] for x in gnss_vel_records])
    v_g = np.array([x[1] for x in gnss_vel_records])
    s_idx = np.argsort(t_g)
    t_g, v_g = t_g[s_idx], v_g[s_idx]

    v_gnss_interp = np.interp(t_m, t_g, v_g)
    dt_m = np.diff(t_m, prepend=t_m[0])
    dt_m[0] = 0.0
    r_gnss = np.cumsum(v_gnss_interp * dt_m)
    tot_gnss = float(r_gnss[-1])

    if tot_gnss < 150.0:  # Пропускаем чистые стоянки
        return None

    # Оценка пути по колесам
    if bogie_vel_records:
        t_b = np.array([x[0] for x in bogie_vel_records])
        v_b = np.array([x[1] for x in bogie_vel_records])
        s_idx_b = np.argsort(t_b)
        t_b, v_b = t_b[s_idx_b], v_b[s_idx_b]
        v_bogie_interp = np.interp(t_m, t_b, v_b)
        tot_bogie = float(np.sum(v_bogie_interp * dt_m))
    else:
        tot_bogie = tot_gnss

    err_bogie_m = tot_bogie - tot_gnss
    pct_bogie = abs(err_bogie_m) / tot_gnss * 100.0

    res = {
        "bag_id": bag_path.name,
        "route": route_name,
        "duration_sec": duration_sec,
        "r_gnss_m": tot_gnss,
        "r_bogie_m": tot_bogie,
        "err_bogie_m": err_bogie_m,
        "pct_bogie": pct_bogie,
    }

    # Метрики по каждой массе
    for m in masses:
        m_tag = f"{m/1000:.1f}t".replace(".", "_")
        d = nodes[m].get_arrays()
        s_m = d["distance"]
        v_m = d["v_est"]
        tot_m = float(s_m[-1])
        err_m = tot_m - tot_gnss
        pct_m = abs(err_m) / tot_gnss * 100.0

        v_diff = v_m - v_gnss_interp
        rmse_kmh = float(np.sqrt(np.mean(v_diff ** 2))) * 3.6
        mae_kmh = float(np.mean(np.abs(v_diff))) * 3.6

        res[f"r_{m_tag}_m"] = tot_m
        res[f"err_{m_tag}_m"] = err_m
        res[f"pct_{m_tag}"] = pct_m
        res[f"rmse_{m_tag}_kmh"] = rmse_kmh
        res[f"mae_{m_tag}_kmh"] = mae_kmh

    return res


def main():
    routes = ["щук-талл", "талл-щук"]
    masses = (24000.0, 24500.0, 25000.0)
    all_results = []

    print("=" * 100)
    print("ПРОГОН ДИНАМИЧЕСКОЙ МОДЕЛИ (24.0 т, 24.5 т, 25.0 т) ПО ВСЕМ ВАЛИДНЫМ БАГАМ")
    print("=" * 100)

    for route_name in routes:
        folder = ws_dir / "data" / route_name
        bags = sorted([d for d in folder.iterdir() if d.is_dir() and (d / "metadata.yaml").exists()])
        print(f"\nМаршрут {route_name.upper()} ({len(bags)} заездов):")
        for b in bags:
            res = process_bag_multi_mass(b, route_name, masses)
            if res:
                all_results.append(res)
                print(
                    f"[{len(all_results):2d}/56] {res['bag_id']}: L_gnss={res['r_gnss_m']:5.0f}м | "
                    f"24.0т: {res['err_24_0t_m']:+6.1f}м ({res['pct_24_0t']:4.1f}%) | "
                    f"24.5т: {res['err_24_5t_m']:+6.1f}м ({res['pct_24_5t']:4.1f}%) | "
                    f"25.0т: {res['err_25_0t_m']:+6.1f}м ({res['pct_25_0t']:4.1f}%)"
                )

    print(f"\nВсего успешно обработано валидных заездов: {len(all_results)}")

    if not all_results:
        print("Нет данных.")
        return

    # Сохраняем полный CSV
    reports_dir = ws_dir / "reports"
    reports_dir.mkdir(exist_ok=True)
    csv_file = reports_dir / "dynamic_model_masses_comparison_24_24.5_25.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)
    print(f"Полный CSV отчет сохранен в: {csv_file}")


if __name__ == "__main__":
    main()

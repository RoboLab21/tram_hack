import sys
from pathlib import Path
_repo_dir = Path(__file__).resolve().parent.parent
for _p in [str(_repo_dir), str(_repo_dir / "solution"), str(_repo_dir / "scripts")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

#!/usr/bin/env python3
"""
Скрипт пакетного прогона резервной одометрии по всем bag-файлам датасета.

Формирует:
1. Итоговый структурированный отчет (Markdown и текст) со сводными таблицами
   и детальными таблицами погрешностей по участкам для каждого прогона.
2. CSV-файл сводной статистики по всем 122 прогонам (all_bags_summary.csv).
3. CSV-файл детальных участков по всем прогонам (all_bags_checkpoints.csv).
"""

import sys
import os
import argparse
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

from odometry_node import DeadReckoningNode


def setup_typestore(ws_dir: Path):
    """Регистрирует кастомные типы сообщений tram_vehicle_msgs в rosbags."""
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


def process_single_bag(bag_path: Path, typestore, step_dist_m: float = 500.0) -> Optional[Dict[str, Any]]:
    """
    Обрабатывает один bag-файл, возвращает словарь с метриками и таблицей участков.
    """
    node = DeadReckoningNode(input_in_kmh=True, integration_method="trapezoidal")
    gnss_records = []

    target_topics = {
        "/vehicle/front_bogie_velocity",
        "/vehicle/rear_bogie_velocity",
        "/sensing/gnss/master/vel"
    }

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

                if conn.topic == "/vehicle/front_bogie_velocity":
                    node.update_front(t, msg.velocity)
                elif conn.topic == "/vehicle/rear_bogie_velocity":
                    node.update_rear(t, msg.velocity)
                elif conn.topic == "/sensing/gnss/master/vel":
                    vx = msg.twist.linear.x
                    vy = msg.twist.linear.y
                    v_gnss = float(np.hypot(vx, vy))
                    gnss_records.append((t, v_gnss))
    except Exception as e:
        print(f"Ошибка чтения {bag_path.name}: {e}")
        return None

    odom_data = node.get_arrays()
    if not odom_data or len(odom_data["timestamp"]) == 0:
        return None

    t_odom = odom_data["timestamp"]
    v_odom = odom_data["v_avg"]
    r_odom = odom_data["distance"]
    duration_sec = float(t_odom[-1] - t_odom[0])
    total_ro = float(r_odom[-1])
    vehicle_id = bag_path.name.split("_")[0]

    has_gnss = len(gnss_records) > 10

    res = {
        "bag_id": bag_path.name,
        "vehicle": vehicle_id,
        "duration_sec": duration_sec,
        "total_r_odom": total_ro,
        "has_gnss": has_gnss,
        "total_r_gnss": None,
        "final_diff_m": None,
        "final_rel_err_pct": None,
        "max_err_m": None,
        "max_rel_to_total_pct": None,
        "mae_dist_m": None,
        "rmse_dist_m": None,
        "mae_vel_kmh": None,
        "checkpoints": []
    }

    if not has_gnss:
        return res

    t_gnss_raw = np.array([x[0] for x in gnss_records])
    v_gnss_raw = np.array([x[1] for x in gnss_records])
    sort_idx = np.argsort(t_gnss_raw)
    t_gnss_raw = t_gnss_raw[sort_idx]
    v_gnss_raw = v_gnss_raw[sort_idx]

    # Интерполяция и расчет пути GNSS
    v_gnss_interp = np.interp(t_odom, t_gnss_raw, v_gnss_raw)
    dt_odom = np.diff(t_odom, prepend=t_odom[0])
    dt_odom[0] = 0.0
    r_gnss = np.cumsum(v_gnss_interp * dt_odom)

    total_rg = float(r_gnss[-1])
    delta_r = r_odom - r_gnss
    abs_delta_r = np.abs(delta_r)

    final_diff = total_ro - total_rg
    final_rel_err = (abs(final_diff) / total_rg * 100.0) if total_rg > 1e-2 else 0.0
    max_err_m = float(np.max(abs_delta_r))
    max_rel_total = (max_err_m / total_rg * 100.0) if total_rg > 1e-2 else 0.0

    mae_dist = float(np.mean(abs_delta_r))
    rmse_dist = float(np.sqrt(np.mean(delta_r ** 2)))

    vel_err = np.abs(v_odom - v_gnss_interp)
    mae_vel_kmh = float(np.mean(vel_err) * 3.6)

    res.update({
        "total_r_gnss": total_rg,
        "final_diff_m": final_diff,
        "final_rel_err_pct": final_rel_err,
        "max_err_m": max_err_m,
        "max_rel_to_total_pct": max_rel_total,
        "mae_dist_m": mae_dist,
        "rmse_dist_m": rmse_dist,
        "mae_vel_kmh": mae_vel_kmh
    })

    # Расчет контрольных точек участков
    # Если путь меньше 500 м, берем адаптивный шаг
    actual_step = step_dist_m if total_ro >= step_dist_m else max(50.0, total_ro / 4.0)
    checkpoints = np.arange(actual_step, total_ro, actual_step)
    if len(checkpoints) == 0 or checkpoints[-1] < total_ro - (actual_step * 0.2):
        checkpoints = np.append(checkpoints, total_ro)

    t_start = t_odom[0]
    prev_ro, prev_rg = 0.0, 0.0

    for idx, cp in enumerate(checkpoints, 1):
        k = min(np.searchsorted(r_odom, cp), len(r_odom) - 1)
        t_cur = float(t_odom[k] - t_start)
        ro = float(r_odom[k])
        rg = float(r_gnss[k])
        diff = ro - rg

        rel_curr = (abs(diff) / rg * 100.0) if rg > 1e-2 else 0.0
        rel_tot = (abs(diff) / total_rg * 100.0) if total_rg > 1e-2 else 0.0
        seg_diff = (ro - prev_ro) - (rg - prev_rg)
        seg_rg = rg - prev_rg
        seg_rel = (abs(seg_diff) / seg_rg * 100.0) if seg_rg > 1e-2 else 0.0

        res["checkpoints"].append({
            "num": idx,
            "time_sec": t_cur,
            "r_odom": ro,
            "r_gnss": rg,
            "diff_m": diff,
            "rel_curr_pct": rel_curr,
            "rel_total_pct": rel_tot,
            "seg_diff_m": seg_diff,
            "seg_rel_pct": seg_rel,
            "v_odom_kmh": float(v_odom[k] * 3.6),
            "v_gnss_kmh": float(v_gnss_interp[k] * 3.6)
        })
        prev_ro, prev_rg = ro, rg

    return res


def run_batch_evaluation(data_dir: Path, ws_dir: Path, output_md: Path, step_dist_m: float = 500.0):
    """
    Пакетный прогон по всем директориям в data_dir.
    """
    typestore = setup_typestore(ws_dir)

    bag_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and (d / "metadata.yaml").exists()])
    total_bags = len(bag_dirs)
    print(f"\nНайдено {total_bags} bag-директорий в {data_dir}. Запуск пакетной обработки...\n")

    results: List[Dict[str, Any]] = []

    for idx, bpath in enumerate(bag_dirs, 1):
        print(f"[{idx:3d}/{total_bags}] Обработка {bpath.name}...", end="\r", flush=True)
        res = process_single_bag(bpath, typestore, step_dist_m)
        if res:
            results.append(res)

    print(f"\nУспешно обработано: {len(results)} из {total_bags} прогонов.\n")

    # -------------------------------------------------------------------------
    # Экспорт сводной статистики в CSV
    # -------------------------------------------------------------------------
    reports_dir = ws_dir / "reports"
    reports_dir.mkdir(exist_ok=True)
    csv_summary_path = reports_dir / "all_bags_summary.csv"
    with open(csv_summary_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "bag_id", "vehicle", "duration_sec", "has_gnss",
            "r_odom_m", "r_gnss_m", "final_diff_m", "final_rel_err_pct",
            "max_err_m", "max_rel_to_total_pct", "mae_dist_m", "rmse_dist_m", "mae_vel_kmh"
        ])
        for r in results:
            writer.writerow([
                r["bag_id"],
                r["vehicle"],
                f"{r['duration_sec']:.1f}",
                "YES" if r["has_gnss"] else "NO",
                f"{r['total_r_odom']:.2f}",
                f"{r['total_r_gnss']:.2f}" if r["has_gnss"] else "",
                f"{r['final_diff_m']:+.2f}" if r["has_gnss"] else "",
                f"{r['final_rel_err_pct']:.3f}" if r["has_gnss"] else "",
                f"{r['max_err_m']:.2f}" if r["has_gnss"] else "",
                f"{r['max_rel_to_total_pct']:.3f}" if r["has_gnss"] else "",
                f"{r['mae_dist_m']:.2f}" if r["has_gnss"] else "",
                f"{r['rmse_dist_m']:.2f}" if r["has_gnss"] else "",
                f"{r['mae_vel_kmh']:.2f}" if r["has_gnss"] else "",
            ])
    print(f"📄 Сводная статистика сохранена в CSV: {csv_summary_path.name}")

    # -------------------------------------------------------------------------
    # Экспорт детальных чекпоинтов всех бэгов в CSV
    # -------------------------------------------------------------------------
    csv_checkpoints_path = reports_dir / "all_bags_checkpoints.csv"
    with open(csv_checkpoints_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "bag_id", "checkpoint", "time_sec", "r_odom_m", "r_gnss_m",
            "delta_r_m", "rel_err_curr_pct", "rel_err_total_pct",
            "seg_delta_r_m", "seg_rel_err_pct", "v_odom_kmh", "v_gnss_kmh"
        ])
        for r in results:
            if not r["has_gnss"]:
                continue
            for cp in r["checkpoints"]:
                writer.writerow([
                    r["bag_id"],
                    f"#{cp['num']}",
                    f"{cp['time_sec']:.1f}",
                    f"{cp['r_odom']:.2f}",
                    f"{cp['r_gnss']:.2f}",
                    f"{cp['diff_m']:+.2f}",
                    f"{cp['rel_curr_pct']:.3f}",
                    f"{cp['rel_total_pct']:.3f}",
                    f"{cp['seg_diff_m']:+.2f}",
                    f"{cp['seg_rel_pct']:.3f}",
                    f"{cp['v_odom_kmh']:.1f}",
                    f"{cp['v_gnss_kmh']:.1f}",
                ])
    print(f"📄 Детальные участки всех бэгов сохранены в CSV: {csv_checkpoints_path.name}")

    # -------------------------------------------------------------------------
    # Формирование подробного Markdown / Текстового отчета
    # -------------------------------------------------------------------------
    gnss_results = [r for r in results if r["has_gnss"] and r["total_r_gnss"] > 10.0]
    blind_results = [r for r in results if not r["has_gnss"]]

    lines = []
    lines.append("# Пакетный отчет проверки резервной одометрии по всем 122 прогонам\n")
    lines.append("Формула решения: **$R = \\sum V_{cp}(t) \\cdot \\Delta t$**, где входные скорости колес переводятся из км/ч в м/с ($V / 3.6$).\n")

    # Общая статистика
    if gnss_results:
        tot_km_odom = sum(r["total_r_odom"] for r in gnss_results) / 1000.0
        tot_km_gnss = sum(r["total_r_gnss"] for r in gnss_results) / 1000.0
        rel_errors = [r["final_rel_err_pct"] for r in gnss_results]
        max_drifts = [r["max_rel_to_total_pct"] for r in gnss_results]

        lines.append("## Общая статистика по валидационным прогонам с эталоном GNSS\n")
        lines.append(f"- Всего прогонов в датасете: **{len(results)}**")
        lines.append(f"- Прогонов с эталоном GNSS: **{len(gnss_results)}**")
        lines.append(f"- Прогонов без эталона GNSS («слепые» тестовые): **{len(blind_results)}**")
        lines.append(f"- Суммарная дистанция прогонов с GNSS: **{tot_km_gnss:.2f} км** (по одометрии: **{tot_km_odom:.2f} км**)")
        lines.append(f"- Средняя финальная погрешность относительно всего пути: **{np.mean(rel_errors):.3f} %**")
        lines.append(f"- Медианная финальная погрешность: **{np.median(rel_errors):.3f} %**")
        lines.append(f"- Средний максимальный дрейф пути: **{np.mean(max_drifts):.3f} %**")
        lines.append(f"- Худшая финальная погрешность среди всех прогонов: **{np.max(rel_errors):.3f} %**\n")

    lines.append("---\n")
    lines.append("## 1. Сводная таблица по всем прогонам\n")
    lines.append("| № | Идентификатор bag | Трамвай | Время (мин) | Путь одом (м) | Путь GNSS (м) | Ошибка ΔR (м) | Погр. к ВСЕМУ пути (%) | Макс. дрейф (%) | MAE скорости (км/ч) |")
    lines.append("|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    for i, r in enumerate(results, 1):
        dur_min = r["duration_sec"] / 60.0
        ro_str = f"{r['total_r_odom']:.1f}"
        if r["has_gnss"]:
            rg_str = f"{r['total_r_gnss']:.1f}"
            diff_str = f"{r['final_diff_m']:+.2f}"
            rel_str = f"{r['final_rel_err_pct']:.3f}%"
            max_drift_str = f"{r['max_rel_to_total_pct']:.3f}%"
            vel_str = f"{r['mae_vel_kmh']:.2f}"
        else:
            rg_str = "—"
            diff_str = "—"
            rel_str = "—"
            max_drift_str = "—"
            vel_str = "—"

        lines.append(
            f"| {i:3d} | `{r['bag_id']}` | {r['vehicle']} | {dur_min:4.1f} | "
            f"{ro_str} | {rg_str} | {diff_str} | {rel_str} | {max_drift_str} | {vel_str} |"
        )

    lines.append("\n---\n")
    lines.append("## 2. Детальные таблицы по участкам маршрута (для прогонов с GNSS)\n")

    for r in gnss_results:
        lines.append(f"### Прогон `{r['bag_id']}` (трамвай {r['vehicle']}, {r['duration_sec']/60.0:.1f} мин)\n")
        lines.append(f"- **Путь по одометрии**: {r['total_r_odom']:.2f} м")
        lines.append(f"- **Путь по эталону GNSS**: {r['total_r_gnss']:.2f} м")
        lines.append(f"- **Финальная ошибка**: {r['final_diff_m']:+.2f} м ({r['final_rel_err_pct']:.3f}% от всего пути)")
        lines.append(f"- **Макс. дрейф**: {r['max_err_m']:.2f} м ({r['max_rel_to_total_pct']:.3f}%)\n")

        lines.append("| Участок | Время (с) | R одом (м) | R GNSS (м) | ΔR након (м) | % тек | % от ВСЕГО | ΔR отрезка (м) | % отрезка | V_ср (км/ч) | V_GNSS (км/ч) |")
        lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

        for cp in r["checkpoints"]:
            lines.append(
                f"| #{cp['num']} | {cp['time_sec']:.1f} | {cp['r_odom']:.1f} | {cp['r_gnss']:.1f} | "
                f"{cp['diff_m']:+.2f} | {cp['rel_curr_pct']:.2f}% | {cp['rel_total_pct']:.3f}% | "
                f"{cp['seg_diff_m']:+.2f} | {cp['seg_rel_pct']:.2f}% | "
                f"{cp['v_odom_kmh']:.1f} | {cp['v_gnss_kmh']:.1f} |"
            )
        lines.append("\n")

    with open(output_md, mode="w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"📑 Полный структурированный отчет сохранен в файл: {output_md.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Пакетный прогон резервной одометрии по всем bag-файлам датасета."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Путь к папке с распакованными bag (по умолчанию 'data')"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="all_bags_report.md",
        help="Имя выходного Markdown-отчета (по умолчанию 'all_bags_report.md')"
    )
    parser.add_argument(
        "--step",
        type=float,
        default=500.0,
        help="Шаг участков в метрах для таблиц (по умолчанию 500 м)"
    )

    args = parser.parse_args()
    ws_dir = _repo_dir
    data_dir = ws_dir / args.data_dir
    reports_dir = ws_dir / "reports"
    reports_dir.mkdir(exist_ok=True)
    output_md = reports_dir / args.output

    run_batch_evaluation(
        data_dir=data_dir,
        ws_dir=ws_dir,
        output_md=output_md,
        step_dist_m=args.step
    )


if __name__ == "__main__":
    main()

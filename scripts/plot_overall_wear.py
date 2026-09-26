#!/usr/bin/env python3
"""
Генерация наглядного графика анализа общего износа колес (Overall Wheel Wear)
и его влияния на одометрию до и после калибровки.
"""

import sys
from pathlib import Path
_repo_dir = Path(__file__).resolve().parent.parent
for _p in [str(_repo_dir), str(_repo_dir / "solution"), str(_repo_dir / "scripts")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

ws_dir = _repo_dir
csv_path = ws_dir / "reports" / "overall_wheel_wear_summary.csv"

with open(csv_path, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

# Привязка дат
from rosbags.highlevel import AnyReader
from batch_evaluate_filtered import setup_typestore
typestore = setup_typestore(ws_dir)

for r in rows:
    p = None
    for route in ["data/щук-талл", "data/талл-щук"]:
        cand = ws_dir / route / r["bag"]
        if cand.exists():
            p = cand; break
    if p:
        try:
            with AnyReader([p], default_typestore=typestore) as reader:
                t0 = reader.start_time * 1e-9
                r["dt"] = datetime.fromtimestamp(t0)
        except Exception:
            r["dt"] = datetime(2026, 7, 27)
    else:
        r["dt"] = datetime(2026, 7, 27)

rows = sorted(rows, key=lambda x: x["dt"])

dates = [r["dt"] for r in rows]
scales = [float(r["scale_median"]) for r in rows]
wears = [float(r["wear_pct"]) for r in rows]
dias = [float(r["delta_dia_mm"]) for r in rows]
errs = [float(r["dist_err_m"]) for r in rows]
vehs = [r["vehicle"] for r in rows]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

# 1. Отклонение диаметра колес (общий износ)
for v, color, marker, label in [("30618", "#1f77b4", "o", "Вагон 30618"), ("30639", "#d62728", "s", "Вагон 30639")]:
    sub_d = [dates[i] for i in range(len(rows)) if vehs[i] == v]
    sub_dia = [dias[i] for i in range(len(rows)) if vehs[i] == v]
    sub_wear = [wears[i] for i in range(len(rows)) if vehs[i] == v]
    ax1.scatter(sub_d, sub_dia, color=color, marker=marker, s=60, alpha=0.85, label=label, zorder=5)

ax1.axhline(0, color="black", linestyle="--", linewidth=1.2, alpha=0.7, label="Номинальный диаметр D_ном = 620 мм")
ax1.set_ylabel("Отклонение диаметра $\\Delta D$, мм", fontsize=11, fontweight="bold")
ax1.set_title("Общий износ колес и изменение эффективного диаметра бандажей во времени", fontsize=13, fontweight="bold")
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend(loc="lower right", frameon=True, fontsize=10)

# Вторая ось Y для % износа
ax1_r = ax1.twinx()
ax1_r.set_ylabel("Масштаб скорости $\\Delta k$, %", fontsize=11)
ymin, ymax = ax1.get_ylim()
ax1_r.set_ylim(ymin / 620.0 * 100.0, ymax / 620.0 * 100.0)

# 2. Систематическая ошибка одометрии из-за износа
for v, color, marker, label in [("30618", "#1f77b4", "o", "30618"), ("30639", "#d62728", "s", "30639")]:
    sub_d = [dates[i] for i in range(len(rows)) if vehs[i] == v]
    sub_err = [errs[i] for i in range(len(rows)) if vehs[i] == v]
    ax2.scatter(sub_d, sub_err, color=color, marker=marker, s=60, alpha=0.85, label=f"Ошибка одометрии ({v})", zorder=5)

ax2.axhline(0, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
ax2.set_ylabel("Ошибка одометрии, м", fontsize=11, fontweight="bold")
ax2.set_xlabel("Дата записи заезда", fontsize=11, fontweight="bold")
ax2.set_title("Систематический уход одометрии (без компенсации общего износа колес)", fontsize=12, fontweight="bold")
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend(loc="lower right", frameon=True, fontsize=10)

plt.tight_layout()
out_png = ws_dir / "img" / "overall_wheel_wear_analysis.png"
plt.savefig(out_png, dpi=150)
print(f"График сохранен в {out_png}")

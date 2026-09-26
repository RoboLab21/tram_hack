import sys
from pathlib import Path
_repo_dir = Path(__file__).resolve().parent.parent
for _p in [str(_repo_dir), str(_repo_dir / "solution"), str(_repo_dir / "scripts")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

#!/usr/bin/env python3
"""
Скрипт распределения bag-файлов:
1. data_no_gnss/     - пробеги без GNSS (36 шт.)
2. data_gnss_errors/ - пробеги с ошибками и сбоями GNSS (21 шт.)
3. data/             - валидные эталонные прогоны с корректным GNSS (65 шт.)
"""

import os
import shutil
import json
from pathlib import Path

ws_dir = Path(__file__).resolve().parent
data_dir = ws_dir / "data"

no_gnss_dir = ws_dir / "data_no_gnss"
gnss_errors_dir = ws_dir / "data_gnss_errors"

no_gnss_dir.mkdir(exist_ok=True)
gnss_errors_dir.mkdir(exist_ok=True)

# 1. Список прогонов без GNSS (36 шт.)
no_gnss_bags = [
    "30618_0259fe53", "30618_0a83c933", "30618_117c2d02", "30618_20096314",
    "30618_2161b58b", "30618_2255aade", "30618_2d2fa7be", "30618_2f2f1175",
    "30618_3ba2326f", "30618_3e012faf", "30618_46e21b9b", "30618_4e1e3181",
    "30618_5036aa78", "30618_6236f680", "30618_74559c73", "30618_79b204dc",
    "30618_7a152380", "30618_7bfbb5ed", "30618_7d88fa79", "30618_81c22fee",
    "30618_887a2b9a", "30618_8eb8615c", "30618_8f08df08", "30618_92b1b453",
    "30618_95c49c30", "30618_9bbe6faa", "30618_9c10cd0d", "30618_b1098bdb",
    "30618_bcc9e7a2", "30618_cfd9fd5a", "30618_ddad08d6", "30618_e151d6e4",
    "30618_e392e5bd", "30618_e7dcdab1", "30618_f28179bb", "30639_e4379d7f"
]

# 2. Словарь прогонов с ошибками GNSS и причинами (21 шт.)
gnss_error_bags = {
    "30618_01f73500": "Разрыв потока GNSS на ходу (7.9с), 10 скачков координат (до 33.1м)",
    "30618_0686195f": "323 аномальных скачка координат (до 47.7м), дрейф координат +8.6 км",
    "30618_27e994fc": "41 аномальный скачок координат (до 48.0м), дрейф координат +4.6 км, разрыв 15.7с",
    "30618_49fe4c54": "Разрыв потока GNSS на ходу (4.1с), 14 скачков координат (до 33.9м)",
    "30618_4d487b0d": "Критическое зануление скорости GNSS на ходу трамвая (12.3с), разрыв связи 33.4с",
    "30618_616ec56b": "13 аномальных скачков координат (до 42.7м)",
    "30618_68d1748a": "Разрыв потока GNSS на ходу (4.1с), 18 скачков координат (до 25.7м)",
    "30618_b3042f78": "Критическое зануление скорости GNSS на ходу трамвая (12.3с), разрыв связи 33.4с",
    "30618_defd0170": "448 аномальных скачков координат (до 48.7м), дрейф координат +14.9 км",
    "30639_0be558e2": "2 разрыва потока GNSS на скорости (до 5.9с)",
    "30639_253671cc": "14 аномальных скачков координат (до 21.7м)",
    "30639_4285f2bc": "Разрыв потока GNSS на ходу (7.5с)",
    "30639_44226bde": "12 аномальных скачков координат (до 42.8м)",
    "30639_50956d6e": "2 разрыва потока GNSS на скорости (до 6.8с)",
    "30639_584b6e32": "4 разрыва потока GNSS на скорости (до 8.6с)",
    "30639_92226df0": "2 разрыва потока GNSS на скорости (до 5.6с)",
    "30639_c31df386": "2 разрыва потока GNSS на скорости (до 4.7с)",
    "30639_d3c43d69": "Разрыв потока GNSS на ходу (12.8с), 75 скачков координат (до 75.7м)",
    "30639_d601d28f": "10 серийных разрывов потока GNSS на скорости 30-45 км/ч (до 7.8с)",
    "30639_d927f360": "Разрыв потока GNSS на ходу (4.3с)",
    "30639_dce52be4": "Разрыв потока GNSS на ходу (4.1с), зануление скорости (1.7с), 71 скачок координат (до 46.8м)"
}

print(f"Перемещение {len(no_gnss_bags)} прогонов без GNSS в {no_gnss_dir.name}...")
moved_no_gnss = 0
for b in no_gnss_bags:
    src = data_dir / b
    dst = no_gnss_dir / b
    if src.exists():
        shutil.move(str(src), str(dst))
        moved_no_gnss += 1

print(f"Перемещение {len(gnss_error_bags)} прогонов со сбоями GNSS в {gnss_errors_dir.name}...")
moved_gnss_err = 0
for b in gnss_error_bags:
    src = data_dir / b
    dst = gnss_errors_dir / b
    if src.exists():
        shutil.move(str(src), str(dst))
        moved_gnss_err += 1

remaining_in_data = len([d for d in data_dir.iterdir() if d.is_dir() and (d / "metadata.yaml").exists()])

print("\n" + "="*60)
print(f"Итоги перемещения:")
print(f"1. Прогоны без GNSS ({no_gnss_dir.name}): {len(list(no_gnss_dir.iterdir()))} шт. (перемещено {moved_no_gnss})")
print(f"2. Прогоны со сбоями GNSS ({gnss_errors_dir.name}): {len(list(gnss_errors_dir.iterdir()))} шт. (перемещено {moved_gnss_err})")
print(f"3. Чистые прогоны с валидным GNSS ({data_dir.name}): {remaining_in_data} шт.")
print(f"Всего: {len(list(no_gnss_dir.iterdir())) + len(list(gnss_errors_dir.iterdir())) + remaining_in_data} шт.")
print("="*60)

# Сохранение реестров
with open(no_gnss_dir / "manifest.json", "w", encoding="utf-8") as f:
    json.dump({"description": "Прогоны без топика /sensing/gnss/master/vel", "count": len(no_gnss_bags), "bags": no_gnss_bags}, f, indent=2, ensure_ascii=False)

with open(gnss_errors_dir / "manifest.json", "w", encoding="utf-8") as f:
    json.dump({"description": "Прогоны со сбоями эталона GNSS", "count": len(gnss_error_bags), "reasons": gnss_error_bags}, f, indent=2, ensure_ascii=False)

print("Реестры manifest.json сформированы в обеих папках.")

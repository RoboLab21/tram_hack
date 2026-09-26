"""
Модуль фильтрованной резервной одометрии беспилотного трамвая с автокалибровкой
износа колес по прямым участкам пути (v3).

Дубликат фильтрованного решения (v2) с добавлением:
1. Автоматическое определение первых 3 прямых участков на цифровой карте пути (pathgrath)
   по кривизне (|curv| < threshold, R_кривизны > 2000 м).
2. Накопление выборок v1 / v2 на прямых участках, где линейные скорости тележек строго равны.
3. Оценка относительного коэффициента износа колес (k_rel = v1 / v2) без использования гироскопа.
4. Уравновешивание скоростей колес с учетом коэффициента износа:
   V_ср = (v1 + k_rel * v2) / 2.
"""

import json
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
from collections import deque
from pathlib import Path
import numpy as np


@dataclass
class StraightSection:
    idx: int            # Номер прямого участка (1, 2, 3...)
    s_start: float      # Начальная путевая координата на треке (м)
    s_end: float        # Конечная путевая координата на треке (м)
    length: float       # Длина прямого участка (м)


@dataclass
class CalibratedOdometryState:
    timestamp: float        # Текущая метка времени (сек)
    v1_raw: float           # Исходная скорость передней тележки (м/с)
    v2_raw: float           # Исходная скорость задней тележки (м/с)
    v1_valid: bool          # Флаг исправности датчика 1
    v2_valid: bool          # Флаг исправности датчика 2
    v_est: float            # Отфильтрованная скорость с учетом износа (м/с)
    diff_sq: float          # Квадратичное отклонение между датчиками (v1 - v2)^2
    fault_type: str         # "OK", "ZERO_V1", "ZERO_V2", "SPIKE_V1", "SPIKE_V2", "EXCESSIVE_DIFF"
    delta_t: float          # Шаг времени (сек)
    delta_r: float          # Приращение пути (м)
    distance: float         # Накопленный путь R (м)
    in_straight: bool       # Флаг нахождения на прямом участке
    straight_id: int        # Номер активного прямого участка (0 - вне прямых)
    k_rel: float            # Текущий коэффициент износа колес тележек (v1 / v2)


def extract_straight_sections(
    path_file_or_data: Any,
    max_curvature: float = 0.0005,  # Радиус кривизны > 2000 м (|curv| < 0.0005 1/м)
    min_length_m: float = 40.0,     # Минимальная длина прямого отрезка для калибровки
    top_n: int = 3                  # Количество прямых участков (первые три)
) -> List[StraightSection]:
    """
    Автоматически находит первые top_n прямых участков на графе пути (pathgrath).
    """
    if isinstance(path_file_or_data, (str, Path)):
        p = Path(path_file_or_data)
        d = json.loads(p.read_text(encoding="utf-8"))
    elif isinstance(path_file_or_data, dict):
        d = path_file_or_data
    else:
        raise ValueError("path_file_or_data должен быть путем к файлу или словарем")

    pts = d.get("points", [])
    if not pts:
        return []

    xs = np.array([pt["x"] for pt in pts])
    ys = np.array([pt["y"] for pt in pts])
    zs = np.array([pt.get("z", 0.0) for pt in pts])
    curvs = np.array([pt.get("curv", 0.0) for pt in pts])

    # Кумулятивное путевое расстояние вдоль точек графа
    dists = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2 + np.diff(zs) ** 2)
    s = np.insert(np.cumsum(dists), 0, 0.0)

    # Маска прямых точек
    is_straight = np.abs(curvs) < max_curvature

    sections: List[StraightSection] = []
    in_seg = False
    start_idx = 0

    for i in range(len(is_straight)):
        if is_straight[i] and not in_seg:
            in_seg = True
            start_idx = i
        elif not is_straight[i] and in_seg:
            in_seg = False
            length = float(s[i - 1] - s[start_idx])
            if length >= min_length_m:
                idx = len(sections) + 1
                sections.append(StraightSection(idx, float(s[start_idx]), float(s[i - 1]), length))
                if len(sections) >= top_n:
                    break

    if in_seg and len(sections) < top_n:
        length = float(s[-1] - s[start_idx])
        if length >= min_length_m:
            idx = len(sections) + 1
            sections.append(StraightSection(idx, float(s[start_idx]), float(s[-1]), length))

    return sections


class PathCalibratedDeadReckoningNode:
    """
    Нода одометрии с фильтрацией сбоев по квадратичному отклонению и
    автокалибровкой коэффициента износа колес на первых трех прямых участках пути.
    """
    def __init__(
        self,
        path_file: Optional[Path] = None,
        straight_sections: Optional[List[StraightSection]] = None,
        input_in_kmh: bool = True,
        integration_method: str = "trapezoidal",
        diff_thresh_kmh: float = 3.6,
        a_max_ms2: float = 2.5,
        v_max_kmh: float = 75.0,
        zero_thresh_kmh: float = 1.0,
        window_size: int = 15,
        min_calib_speed_kmh: float = 10.0  # Минимальная скорость для калибровки (м/с -> 2.78)
    ):
        self.input_in_kmh = input_in_kmh
        self.scale_factor = (1.0 / 3.6) if input_in_kmh else 1.0
        self.integration_method = integration_method

        self.diff_sq_thresh = (diff_thresh_kmh / 3.6) ** 2
        self.a_max_sq = (a_max_ms2) ** 2
        self.v_max_ms = v_max_kmh / 3.6
        self.zero_thresh_ms = zero_thresh_kmh / 3.6
        self.min_calib_speed_ms = min_calib_speed_kmh / 3.6
        self.window_size = window_size

        # Загрузка прямых участков
        if straight_sections is not None:
            self.straight_sections = straight_sections
        elif path_file is not None:
            self.straight_sections = extract_straight_sections(path_file)
        else:
            self.straight_sections = []

        self.reset()

    def set_path(self, path_file: Path):
        """Динамическая загрузка графа пути и определение прямых участков."""
        self.straight_sections = extract_straight_sections(path_file)

    def reset(self):
        """Сброс состояния ноды."""
        self.R: float = 0.0
        self.last_time: Optional[float] = None
        self.v1: float = 0.0
        self.v2: float = 0.0
        self.v_est: float = 0.0
        self.last_v_est: float = 0.0

        # Параметры автокалибровки износа колес
        self.k_rel: float = 1.0               # v1 / v2
        self.calib_samples: List[float] = []   # Собранные мгновенные отношения v1/v2 на прямых
        self.section_samples: Dict[int, List[float]] = {s.idx: [] for s in self.straight_sections}
        self.is_calibrated: bool = False

        self.history_v1 = deque(maxlen=self.window_size)
        self.history_v2 = deque(maxlen=self.window_size)
        self.history_vest = deque(maxlen=self.window_size)
        self.history: List[CalibratedOdometryState] = []

    def update_front(self, timestamp: float, velocity_raw: float) -> Optional[CalibratedOdometryState]:
        self.v1 = float(velocity_raw) * self.scale_factor
        return self._step(timestamp)

    def update_rear(self, timestamp: float, velocity_raw: float) -> Optional[CalibratedOdometryState]:
        self.v2 = float(velocity_raw) * self.scale_factor
        return self._step(timestamp)

    def step(self, timestamp: float, v1_raw: float, v2_raw: float) -> CalibratedOdometryState:
        self.v1 = float(v1_raw) * self.scale_factor
        self.v2 = float(v2_raw) * self.scale_factor
        return self._step(timestamp)

    def _check_straight_section(self, r_current: float) -> Tuple[bool, int]:
        """Проверяет, попадает ли текущий пройденный путь в один из прямых участков."""
        for s in self.straight_sections:
            if s.s_start <= r_current <= s.s_end:
                return True, s.idx
        return False, 0

    def _step(self, timestamp: float) -> CalibratedOdometryState:
        in_straight, straight_id = self._check_straight_section(self.R)

        if self.last_time is None:
            self.last_time = timestamp
            self.v_est = (self.v1 + self.v2) / 2.0
            self.last_v_est = self.v_est
            self.history_v1.append(self.v1)
            self.history_v2.append(self.v2)
            self.history_vest.append(self.v_est)

            state = CalibratedOdometryState(
                timestamp=timestamp,
                v1_raw=self.v1,
                v2_raw=self.v2,
                v1_valid=True,
                v2_valid=True,
                v_est=self.v_est,
                diff_sq=(self.v1 - self.v2) ** 2,
                fault_type="OK",
                delta_t=0.0,
                delta_r=0.0,
                distance=self.R,
                in_straight=in_straight,
                straight_id=straight_id,
                k_rel=self.k_rel
            )
            self.history.append(state)
            return state

        dt = timestamp - self.last_time
        if dt <= 0.0:
            dt = 0.0
        elif dt > 2.0:
            dt = 0.0
            self.last_time = timestamp

        # ---------------------------------------------------------------------
        # 1. КВАДРАТИЧНЫЙ ФИЛЬТР СБОЕВ ДАТЧИКОВ
        # ---------------------------------------------------------------------
        diff_sq = (self.v1 - self.v2) ** 2

        if dt > 0.005:
            a1_sq = ((self.v1 - self.last_v_est) / dt) ** 2
            a2_sq = ((self.v2 - self.last_v_est) / dt) ** 2
        else:
            a1_sq, a2_sq = 0.0, 0.0

        v1_valid = True
        v2_valid = True
        fault_type = "OK"

        if self.v1 > self.v_max_ms:
            v1_valid = False
            fault_type = "SPIKE_V1"
        if self.v2 > self.v_max_ms:
            v2_valid = False
            fault_type = "SPIKE_V2"

        if diff_sq > self.diff_sq_thresh:
            v1_is_zero = (self.v1 < self.zero_thresh_ms)
            v2_is_zero = (self.v2 < self.zero_thresh_ms)

            if v1_is_zero and (self.v2 >= self.zero_thresh_ms * 2.0):
                v1_valid = False
                fault_type = "ZERO_V1"
            elif v2_is_zero and (self.v1 >= self.zero_thresh_ms * 2.0):
                v2_valid = False
                fault_type = "ZERO_V2"
            elif a1_sq > self.a_max_sq and a2_sq <= self.a_max_sq:
                v1_valid = False
                fault_type = "SPIKE_V1"
            elif a2_sq > self.a_max_sq and a1_sq <= self.a_max_sq:
                v2_valid = False
                fault_type = "SPIKE_V2"
            else:
                fault_type = "EXCESSIVE_DIFF"
                if len(self.history_vest) >= 5:
                    mu_recent = np.mean(self.history_vest)
                    sq_dev1 = (self.v1 - mu_recent) ** 2
                    sq_dev2 = (self.v2 - mu_recent) ** 2
                    if sq_dev1 > 4.0 * sq_dev2 and sq_dev1 > self.diff_sq_thresh:
                        v1_valid = False
                    elif sq_dev2 > 4.0 * sq_dev1 and sq_dev2 > self.diff_sq_thresh:
                        v2_valid = False

        # ---------------------------------------------------------------------
        # 2. АВТОКАЛИБРОВКА КОЭФФИЦИЕНТА ИЗНОСА НА ПРЯМЫХ УЧАСТКАХ
        # ---------------------------------------------------------------------
        # На прямом участке скорости обязаны быть равны. Если оба датчика исправны
        # и трамвай уверенно едет (> 10 км/ч), накапливаем отношение v1 / v2.
        if in_straight and v1_valid and v2_valid:
            if self.v1 > self.min_calib_speed_ms and self.v2 > self.min_calib_speed_ms:
                # Отсеиваем случайные шумы (|v1/v2 - 1| < 5%)
                ratio = self.v1 / self.v2
                if 0.95 <= ratio <= 1.05:
                    self.calib_samples.append(ratio)
                    if straight_id in self.section_samples:
                        self.section_samples[straight_id].append(ratio)

                    # Робастное обновление k_rel (медиана по всем выборкам)
                    if len(self.calib_samples) >= 30:
                        self.k_rel = float(np.median(self.calib_samples))
                        self.is_calibrated = True

        # ---------------------------------------------------------------------
        # 3. УРАВНОВЕШИВАНИЕ СКОРОСТИ С УЧЕТОМ ИЗНОСА (k_rel)
        # ---------------------------------------------------------------------
        if v1_valid and v2_valid:
            # Тележка 2 масштабируется к передней тележке через k_rel
            v2_calibrated = self.k_rel * self.v2
            v_est = (self.v1 + v2_calibrated) / 2.0
        elif v1_valid and not v2_valid:
            v_est = self.v1
        elif v2_valid and not v1_valid:
            v_est = self.k_rel * self.v2
        else:
            v_est = self.last_v_est

        self.history_v1.append(self.v1)
        self.history_v2.append(self.v2)
        self.history_vest.append(v_est)

        # ---------------------------------------------------------------------
        # 4. ИНТЕГРИРОВАНИЕ ПУТИ
        # ---------------------------------------------------------------------
        if dt > 0.0:
            if self.integration_method == "trapezoidal":
                v_eff = (self.last_v_est + v_est) / 2.0
            else:
                v_eff = v_est
            delta_r = v_eff * dt
            self.R += delta_r
            self.last_time = timestamp
            self.last_v_est = v_est
        else:
            delta_r = 0.0

        self.v_est = v_est

        state = CalibratedOdometryState(
            timestamp=timestamp,
            v1_raw=self.v1,
            v2_raw=self.v2,
            v1_valid=v1_valid,
            v2_valid=v2_valid,
            v_est=v_est,
            diff_sq=diff_sq,
            fault_type=fault_type,
            delta_t=dt,
            delta_r=delta_r,
            distance=self.R,
            in_straight=in_straight,
            straight_id=straight_id,
            k_rel=self.k_rel
        )
        self.history.append(state)
        return state

    def get_calibration_report(self) -> Dict[str, Any]:
        """Возвращает отчет об определении прямых и калибровке износа."""
        section_stats = {}
        for s in self.straight_sections:
            samples = self.section_samples.get(s.idx, [])
            section_stats[f"straight_{s.idx}"] = {
                "s_range_m": [s.s_start, s.s_end],
                "length_m": s.length,
                "samples_count": len(samples),
                "k_rel_median": float(np.median(samples)) if samples else 1.0,
                "k_rel_mean": float(np.mean(samples)) if samples else 1.0,
                "wear_diff_pct": float((np.median(samples) - 1.0) * 100.0) if samples else 0.0
            }
        return {
            "is_calibrated": self.is_calibrated,
            "total_samples": len(self.calib_samples),
            "final_k_rel": self.k_rel,
            "wear_diff_pct": (self.k_rel - 1.0) * 100.0,
            "straight_sections": section_stats
        }

    def get_arrays(self) -> Dict[str, np.ndarray]:
        if not self.history:
            return {}
        return {
            "timestamp": np.array([s.timestamp for s in self.history]),
            "v1_raw": np.array([s.v1_raw for s in self.history]),
            "v2_raw": np.array([s.v2_raw for s in self.history]),
            "v1_valid": np.array([s.v1_valid for s in self.history]),
            "v2_valid": np.array([s.v2_valid for s in self.history]),
            "v_est": np.array([s.v_est for s in self.history]),
            "diff_sq": np.array([s.diff_sq for s in self.history]),
            "delta_t": np.array([s.delta_t for s in self.history]),
            "distance": np.array([s.distance for s in self.history]),
            "in_straight": np.array([s.in_straight for s in self.history]),
            "straight_id": np.array([s.straight_id for s in self.history]),
            "k_rel": np.array([s.k_rel for s in self.history]),
        }

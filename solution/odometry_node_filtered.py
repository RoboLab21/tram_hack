"""
Модуль фильтрованной резервной одометрии беспилотного трамвая (v2).

Дубликат базовой ноды с алгоритмом фильтрации ошибок датчиков:
1. Квадратичное отклонение между датчиками: (v1 - v2)^2.
2. Детекция выпадения в 0 (один датчик на нуле при движении второго).
3. Детекция слишком больших значений и аномальных скачков скорости (квадратичное ускорение a^2 > a_max^2).
4. Скользящее квадратичное отклонение (дисперсия) для выявления выбросов.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from collections import deque
import numpy as np


@dataclass
class FilteredOdometryState:
    timestamp: float        # Текущая метка времени (сек)
    v1_raw: float           # Исходная скорость передней тележки (м/с)
    v2_raw: float           # Исходная скорость задней тележки (м/с)
    v1_valid: bool          # Флаг валидности датчика 1
    v2_valid: bool          # Флаг валидности датчика 2
    v_est: float            # Оцененная отфильтрованная скорость трамвая (м/с)
    diff_sq: float          # Квадратичное отклонение между датчиками (v1 - v2)^2 (м²/с²)
    fault_type: str         # Тип состояния/сбоя: "OK", "ZERO_V1", "ZERO_V2", "SPIKE_V1", "SPIKE_V2", "EXCESSIVE_DIFF"
    delta_t: float          # Шаг времени (сек)
    delta_r: float          # Приращение пути (м)
    distance: float         # Накопленное расстояние R (м)


class FilteredDeadReckoningNode:
    """
    Нода одометрии с проверкой ошибок датчиков через квадратичное отклонение.
    """
    def __init__(
        self,
        input_in_kmh: bool = True,
        integration_method: str = "trapezoidal",
        diff_thresh_kmh: float = 3.6,        # Порог расхождения между датчиками (км/ч) -> 1.0 м/с
        a_max_ms2: float = 2.5,               # Максимальное физическое ускорение трамвая (м/с²)
        v_max_kmh: float = 75.0,              # Максимально допустимая скорость трамвая (км/ч)
        zero_thresh_kmh: float = 1.0,         # Порог околонулевой скорости (км/ч)
        window_size: int = 15                 # Размер скользящего окна для расчета дисперсии (~1.5 сек)
    ):
        self.input_in_kmh = input_in_kmh
        self.scale_factor = (1.0 / 3.6) if input_in_kmh else 1.0
        self.integration_method = integration_method

        # Квадратичные пороги в (м/с)²
        self.diff_sq_thresh = (diff_thresh_kmh / 3.6) ** 2
        self.a_max_sq = (a_max_ms2) ** 2
        self.v_max_ms = v_max_kmh / 3.6
        self.zero_thresh_ms = zero_thresh_kmh / 3.6
        self.window_size = window_size

        self.reset()

    def reset(self):
        """Сброс состояния ноды."""
        self.R: float = 0.0
        self.last_time: Optional[float] = None
        self.v1: float = 0.0
        self.v2: float = 0.0
        self.v_est: float = 0.0
        self.last_v_est: float = 0.0

        # Скользящие буферы для оценки дисперсии и тренда
        self.history_v1 = deque(maxlen=self.window_size)
        self.history_v2 = deque(maxlen=self.window_size)
        self.history_vest = deque(maxlen=self.window_size)

        self.history: List[FilteredOdometryState] = []

    def update_front(self, timestamp: float, velocity_raw: float) -> Optional[FilteredOdometryState]:
        """Обновление показаний передней тележки."""
        self.v1 = float(velocity_raw) * self.scale_factor
        return self._step(timestamp)

    def update_rear(self, timestamp: float, velocity_raw: float) -> Optional[FilteredOdometryState]:
        """Обновление показаний задней тележки."""
        self.v2 = float(velocity_raw) * self.scale_factor
        return self._step(timestamp)

    def step(self, timestamp: float, v1_raw: float, v2_raw: float) -> FilteredOdometryState:
        """Синхронный шаг с двумя скоростями."""
        self.v1 = float(v1_raw) * self.scale_factor
        self.v2 = float(v2_raw) * self.scale_factor
        return self._step(timestamp)

    def _step(self, timestamp: float) -> FilteredOdometryState:
        """
        Фильтрация датчиков на основе квадратичных отклонений и интегрирование.
        """
        if self.last_time is None:
            self.last_time = timestamp
            self.v_est = (self.v1 + self.v2) / 2.0
            self.last_v_est = self.v_est
            self.history_v1.append(self.v1)
            self.history_v2.append(self.v2)
            self.history_vest.append(self.v_est)

            state = FilteredOdometryState(
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
                distance=self.R
            )
            self.history.append(state)
            return state

        dt = timestamp - self.last_time

        # Защита от нарушений монотонности времени
        if dt <= 0.0:
            dt = 0.0
        elif dt > 2.0:
            # Разрыв связи
            dt = 0.0
            self.last_time = timestamp

        # ---------------------------------------------------------------------
        # 1. КВАДРАТИЧНОЕ ОТКЛОНЕНИЕ МЕЖДУ ДАТЧИКАМИ
        # ---------------------------------------------------------------------
        diff_sq = (self.v1 - self.v2) ** 2

        # 2. Квадратичные ускорения (темп изменения скорости каждого датчика)
        if dt > 0.005:
            a1_sq = ((self.v1 - self.last_v_est) / dt) ** 2
            a2_sq = ((self.v2 - self.last_v_est) / dt) ** 2
        else:
            a1_sq, a2_sq = 0.0, 0.0

        v1_valid = True
        v2_valid = True
        fault_type = "OK"

        # ---------------------------------------------------------------------
        # 3. АНАЛИЗ ОШИБОК ДАТЧИКОВ
        # ---------------------------------------------------------------------
        # А) Проверка на физический потолок максимальной скорости ("слишком большие значения")
        if self.v1 > self.v_max_ms:
            v1_valid = False
            fault_type = "SPIKE_V1"
        if self.v2 > self.v_max_ms:
            v2_valid = False
            fault_type = "SPIKE_V2"

        # Б) Анализ расхождения между датчиками
        if diff_sq > self.diff_sq_thresh:
            v1_is_zero = (self.v1 < self.zero_thresh_ms)
            v2_is_zero = (self.v2 < self.zero_thresh_ms)

            # Случай: один датчик выпал в 0, а второй показывает уверенное движение
            if v1_is_zero and (self.v2 >= self.zero_thresh_ms * 2.0):
                v1_valid = False
                fault_type = "ZERO_V1"
            elif v2_is_zero and (self.v1 >= self.zero_thresh_ms * 2.0):
                v2_valid = False
                fault_type = "ZERO_V2"

            # Случай: один из датчиков имеет физически невозможное квадратичное ускорение (скачок/выброс)
            elif a1_sq > self.a_max_sq and a2_sq <= self.a_max_sq:
                v1_valid = False
                fault_type = "SPIKE_V1"
            elif a2_sq > self.a_max_sq and a1_sq <= self.a_max_sq:
                v2_valid = False
                fault_type = "SPIKE_V2"

            # Случай: оба ненулевые, но значительно разошлись (проскальзывание/кривая)
            else:
                fault_type = "EXCESSIVE_DIFF"
                # Если в скользящем окне есть история, выбираем датчик, чье квадратичное отклонение
                # от скользящего среднего меньше
                if len(self.history_vest) >= 5:
                    mu_recent = np.mean(self.history_vest)
                    sq_dev1 = (self.v1 - mu_recent) ** 2
                    sq_dev2 = (self.v2 - mu_recent) ** 2
                    if sq_dev1 > 4.0 * sq_dev2 and sq_dev1 > self.diff_sq_thresh:
                        v1_valid = False
                    elif sq_dev2 > 4.0 * sq_dev1 and sq_dev2 > self.diff_sq_thresh:
                        v2_valid = False

        # ---------------------------------------------------------------------
        # 4. ФОРМИРОВАНИЕ ИТОГОВОЙ ОЦЕНКИ СКОРОСТИ
        # ---------------------------------------------------------------------
        if v1_valid and v2_valid:
            # Оба датчика исправны -> берем среднее
            v_est = (self.v1 + self.v2) / 2.0
        elif v1_valid and not v2_valid:
            # Доверяем только датчику 1
            v_est = self.v1
        elif v2_valid and not v1_valid:
            # Доверяем только датчику 2
            v_est = self.v2
        else:
            # Оба датчика забракованы -> сохраняем предыдущую скорость (экстраполяция)
            v_est = self.last_v_est

        # Обновляем скользящие буферы
        self.history_v1.append(self.v1)
        self.history_v2.append(self.v2)
        self.history_vest.append(v_est)

        # ---------------------------------------------------------------------
        # 5. ИНТЕГРИРОВАНИЕ ПУТИ
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

        state = FilteredOdometryState(
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
            distance=self.R
        )
        self.history.append(state)
        return state

    def get_arrays(self) -> Dict[str, np.ndarray]:
        """Возвращает историю в виде массивов numpy."""
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
        }

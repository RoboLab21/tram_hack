"""
Модуль резервной одометрии для беспилотного трамвая (без ROS, чистый Python).

Формула счисления пути:
    R = sum(V_cp(t) * delta_t)

Поддерживаемые методы расчета скорости V_cp(t):
1. "mean": мгновенная полусумма тележек:
       V_cp = (V_front(t) + V_rear(t)) / 2
2. "sma": простое скользящее среднее (Simple Moving Average) по окну из N точек:
       V_sma(t) = 1/N * sum(V_raw(t - k), k=0..N-1)
3. "ema": экспоненциальное скользящее среднее (Exponential Moving Average) с фактором alpha:
       alpha = 2 / (N + 1)
       V_ema(t) = alpha * V_raw(t) + (1 - alpha) * V_ema(t-1)

Количество точек сглаживания N (window_size) конфигурируется пользователем.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
from collections import deque
import numpy as np


@dataclass
class OdometryState:
    timestamp: float        # Текущая метка времени (сек)
    v_front: float          # Скорость передней тележки (м/с)
    v_rear: float           # Скорость задней тележки (м/с)
    v_raw: float            # Сырая полусумма скоростей тележек (м/с)
    v_avg: float            # Отфильтрованная скорость V_cp (м/с)
    delta_t: float          # Шаг времени (сек)
    delta_r: float          # Приращение пути за шаг (м)
    distance: float         # Накопленное расстояние R (м)
    filter_method: str      # Метод фильтрации ("mean", "sma", "ema")
    window_size: int        # Количество точек N в окне фильтра


class DeadReckoningNode:
    """
    Нода одометрии, эмулирующая онлайн-обработку сообщений с тележек трамвая.
    Поддерживает сглаживание скорости через SMA и EMA с настраиваемым числом точек N.
    """
    def __init__(
        self,
        input_in_kmh: bool = True,
        integration_method: str = "trapezoidal",
        velocity_filter: str = "ema",
        window_size: int = 5,
        alpha: Optional[float] = None
    ):
        """
        :param input_in_kmh: Если True, входные скорости с тележек переводятся из км/ч в м/с (делим на 3.6).
        :param integration_method: "trapezoidal" ((v_prev + v_curr)/2 * dt) или "rectangular" (v_curr * dt).
        :param velocity_filter: Метод расчета скорости:
            - "mean": прямое среднее (v_front + v_rear) / 2 без сглаживания во времени.
            - "sma": Simple Moving Average по последним N точкам.
            - "ema": Exponential Moving Average с эквивалентным числом точек N.
        :param window_size: Количество точек N для SMA/EMA (по умолчанию 5).
        :param alpha: Коэффициент сглаживания EMA (если None, вычисляется автоматически: alpha = 2 / (N + 1)).
        """
        self.input_in_kmh = input_in_kmh
        self.scale_factor = (1.0 / 3.6) if input_in_kmh else 1.0
        self.integration_method = integration_method
        
        self.velocity_filter = velocity_filter.lower()
        if self.velocity_filter not in {"mean", "sma", "ema"}:
            raise ValueError(f"Неизвестный метод фильтрации '{velocity_filter}'. Допустимы: 'mean', 'sma', 'ema'")

        self.window_size = max(1, int(window_size))
        if alpha is not None:
            self.alpha = float(alpha)
        else:
            self.alpha = 2.0 / (self.window_size + 1.0)
        
        self.reset()

    def set_filter_params(
        self,
        velocity_filter: Optional[str] = None,
        window_size: Optional[int] = None,
        alpha: Optional[float] = None
    ):
        """Динамическое изменение метода фильтрации и числа точек N."""
        if velocity_filter is not None:
            self.velocity_filter = velocity_filter.lower()
        if window_size is not None:
            self.window_size = max(1, int(window_size))
            self.sma_buffer = deque(maxlen=self.window_size)
            if alpha is None:
                self.alpha = 2.0 / (self.window_size + 1.0)
        if alpha is not None:
            self.alpha = float(alpha)

    def reset(self):
        """Сброс состояния ноды в начальное положение."""
        self.R: float = 0.0
        self.last_time: Optional[float] = None
        self.v_front: float = 0.0
        self.v_rear: float = 0.0
        self.v_raw: float = 0.0
        self.v_avg: float = 0.0
        self.last_v_avg: float = 0.0
        
        # Индивидуальные метки времени прихода данных от тележек
        self.last_front_time: Optional[float] = None
        self.last_rear_time: Optional[float] = None
        
        # Состояние фильтров SMA и EMA
        self.sma_buffer: deque = deque(maxlen=self.window_size)
        self.v_ema: Optional[float] = None
        
        # История состояний
        self.history: List[OdometryState] = []

    def update_front(self, timestamp: float, velocity_raw: float) -> Optional[OdometryState]:
        """Обновление скорости передней тележки."""
        self.v_front = float(velocity_raw) * self.scale_factor
        self.last_front_time = timestamp
        return self._step(timestamp)

    def update_rear(self, timestamp: float, velocity_raw: float) -> Optional[OdometryState]:
        """Обновление скорости задней тележки."""
        self.v_rear = float(velocity_raw) * self.scale_factor
        self.last_rear_time = timestamp
        return self._step(timestamp)

    def step(self, timestamp: float, v_front_raw: float, v_rear_raw: float) -> OdometryState:
        """Синхронный шаг с известными скоростями обеих тележек."""
        self.v_front = float(v_front_raw) * self.scale_factor
        self.v_rear = float(v_rear_raw) * self.scale_factor
        return self._step(timestamp)

    def _apply_velocity_filter(self, v_raw: float) -> float:
        """Применяет выбранный метод усреднения скорости (MEAN / SMA / EMA)."""
        if self.velocity_filter == "mean" or self.window_size <= 1:
            return v_raw
        
        elif self.velocity_filter == "sma":
            self.sma_buffer.append(v_raw)
            return float(np.mean(self.sma_buffer))
        
        elif self.velocity_filter == "ema":
            if self.v_ema is None:
                self.v_ema = v_raw
            else:
                self.v_ema = self.alpha * v_raw + (1.0 - self.alpha) * self.v_ema
            return self.v_ema
        
        return v_raw

    def _step(self, timestamp: float) -> OdometryState:
        """Вычисление отфильтрованной скорости V_cp и интегрирование R."""
        self.v_raw = (self.v_front + self.v_rear) / 2.0
        self.v_avg = self._apply_velocity_filter(self.v_raw)
        
        if self.last_time is None:
            self.last_time = timestamp
            self.last_v_avg = self.v_avg
            state = OdometryState(
                timestamp=timestamp,
                v_front=self.v_front,
                v_rear=self.v_rear,
                v_raw=self.v_raw,
                v_avg=self.v_avg,
                delta_t=0.0,
                delta_r=0.0,
                distance=self.R,
                filter_method=self.velocity_filter,
                window_size=self.window_size
            )
            self.history.append(state)
            return state

        dt = timestamp - self.last_time
        
        # Защита от сбоев/инверсии временных меток и аномально больших пауз (> 2.0 сек)
        if dt <= 0:
            delta_r = 0.0
        elif dt > 2.0:
            delta_r = 0.0
            self.last_time = timestamp
            self.last_v_avg = self.v_avg
        else:
            if self.integration_method == "trapezoidal":
                v_eff = (self.last_v_avg + self.v_avg) / 2.0
            else:
                v_eff = self.v_avg
            
            delta_r = v_eff * dt
            self.R += delta_r
            self.last_time = timestamp
            self.last_v_avg = self.v_avg

        state = OdometryState(
            timestamp=timestamp,
            v_front=self.v_front,
            v_rear=self.v_rear,
            v_raw=self.v_raw,
            v_avg=self.v_avg,
            delta_t=max(0.0, dt),
            delta_r=delta_r,
            distance=self.R,
            filter_method=self.velocity_filter,
            window_size=self.window_size
        )
        self.history.append(state)
        return state

    def get_arrays(self) -> Dict[str, np.ndarray]:
        """Возвращает историю в виде numpy-массивов."""
        if not self.history:
            return {}
        return {
            "timestamp": np.array([s.timestamp for s in self.history]),
            "v_front": np.array([s.v_front for s in self.history]),
            "v_rear": np.array([s.v_rear for s in self.history]),
            "v_raw": np.array([s.v_raw for s in self.history]),
            "v_avg": np.array([s.v_avg for s in self.history]),
            "delta_t": np.array([s.delta_t for s in self.history]),
            "distance": np.array([s.distance for s in self.history]),
        }

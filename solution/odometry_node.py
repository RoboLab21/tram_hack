"""
Модуль резервной одометрии для беспилотного трамвая (без ROS, чистый Python).

Формула счисления пути:
    R = sum(V_cp(t) * delta_t)
где V_cp(t) = (V_front(t) + V_rear(t)) / 2.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
import numpy as np


@dataclass
class OdometryState:
    timestamp: float        # Текущая метка времени (сек)
    v_front: float          # Скорость передней тележки (м/с)
    v_rear: float           # Скорость задней тележки (м/с)
    v_avg: float            # Средняя скорость V_cp (м/с)
    delta_t: float          # Шаг времени (сек)
    delta_r: float          # Приращение пути за шаг (м)
    distance: float         # Накопленное расстояние R (м)


class DeadReckoningNode:
    """
    Нода одометрии, эмулирующая онлайн-обработку сообщений с тележек трамвая.
    """
    def __init__(self, input_in_kmh: bool = True, integration_method: str = "trapezoidal"):
        """
        :param input_in_kmh: Если True, входные скорости с тележек переводятся из км/ч в м/с (делим на 3.6).
        :param integration_method: "trapezoidal" ((v_prev + v_curr)/2 * dt) или "rectangular" (v_curr * dt).
        """
        self.input_in_kmh = input_in_kmh
        self.scale_factor = (1.0 / 3.6) if input_in_kmh else 1.0
        self.integration_method = integration_method
        
        self.reset()

    def reset(self):
        """Сброс состояния ноды в начальное положение."""
        self.R: float = 0.0
        self.last_time: Optional[float] = None
        self.v_front: float = 0.0
        self.v_rear: float = 0.0
        self.v_avg: float = 0.0
        self.last_v_avg: float = 0.0
        
        # Индивидуальные метки времени прихода данных от тележек
        self.last_front_time: Optional[float] = None
        self.last_rear_time: Optional[float] = None
        
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

    def _step(self, timestamp: float) -> OdometryState:
        """Вычисление V_cp и интегрирование R."""
        self.v_avg = (self.v_front + self.v_rear) / 2.0
        
        if self.last_time is None:
            self.last_time = timestamp
            self.last_v_avg = self.v_avg
            state = OdometryState(
                timestamp=timestamp,
                v_front=self.v_front,
                v_rear=self.v_rear,
                v_avg=self.v_avg,
                delta_t=0.0,
                delta_r=0.0,
                distance=self.R
            )
            self.history.append(state)
            return state

        dt = timestamp - self.last_time
        
        # Защита от сбоев/инверсии временных меток и аномально больших пауз (> 2.0 сек)
        if dt <= 0:
            delta_r = 0.0
        elif dt > 2.0:
            # При разрыве связи считаем приращение нулевым и обновляем время
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
            v_avg=self.v_avg,
            delta_t=max(0.0, dt),
            delta_r=delta_r,
            distance=self.R
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
            "v_avg": np.array([s.v_avg for s in self.history]),
            "delta_t": np.array([s.delta_t for s in self.history]),
            "distance": np.array([s.distance for s in self.history]),
        }

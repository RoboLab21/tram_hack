"""
Модуль динамической резервной одометрии (счисление пути по матмодели движения).

Работает БЕЗ ДАТЧИКОВ СКОРОСТИ ТЕЛЕЖЕК И БЕЗ GNSS:
Входные данные:
- Положение ручки водителя: driver_cmd in [-15..+15]
- Цифровая карта пути (pathgrath): 3D-геометрия, уклоны и кривизна

Особенности:
1. Автоматический расчет уклона i(s) = dz/ds * 1000 и кривизны kappa(s) вдоль пути.
2. Конечно-автоматная защита стоянки (FSM STOPPED/MOTION): при остановке вагон удерживается
   механическим тормозом, исключая сползание под уклон и дрейф R при нулевой скорости.
3. Численное интегрирование уравнений движения методом трапеций с суб-шагами для численной стабильности.
4. Привязка пройденной путевой координаты s к 3D-координатам карты (x, y, z).
"""

import json
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import numpy as np

try:
    from .tram_dynamic_model import TramDynamicModel, TramParameters
except ImportError:
    from tram_dynamic_model import TramDynamicModel, TramParameters



@dataclass
class DynamicOdometryState:
    timestamp: float        # Временная метка (сек)
    v_est: float            # Оцененная продольная скорость (м/с)
    v_kmh: float            # Оцененная продольная скорость (км/ч)
    acceleration: float     # Расчетное ускорение (м/с²)
    distance: float         # Накопленное пройденное расстояние s (м)
    x: float                # 3D координата X по карте (м)
    y: float                # 3D координата Y по карте (м)
    z: float                # 3D координата Z по карте (м)
    driver_cmd: int         # Положение ручки контроллера водителя (-15..+15)
    f_prop: float           # Сила тяги/торможения (Н)
    w_res: float            # Сила сопротивления (Н)
    slope_permille: float   # Текущий уклон пути (‰)
    curvature: float        # Текущая кривизна пути (1/м)
    fsm_state: str          # Состояние автомата ("STOPPED", "MOTION")


class DynamicOdometryNode:
    """
    Нода счисления пути на основе дифференциальных уравнений движения трамвая «Львёнок».
    """
    def __init__(
        self,
        path_geometry_file_or_data: Optional[Any] = None,
        tram_params: Optional[TramParameters] = None,
        initial_s: float = 0.0,
        substep_dt: float = 0.01  # макс. шаг внутреннего интегрирования для высокой точности
    ):
        # 1. Физическая модель вагона (по умолчанию масса 23 т)
        self.params = tram_params if tram_params is not None else TramParameters(mass_kg=23000.0)
        self.model = TramDynamicModel(self.params)
        self.substep_dt = substep_dt

        # 2. Загрузка карты пути (pathgrath)
        self.s_map: np.ndarray = np.array([0.0, 100000.0])
        self.xs: np.ndarray = np.array([0.0, 0.0])
        self.ys: np.ndarray = np.array([0.0, 0.0])
        self.zs: np.ndarray = np.array([0.0, 0.0])
        self.slopes: np.ndarray = np.array([0.0, 0.0])
        self.curvs: np.ndarray = np.array([0.0, 0.0])
        self.has_map = False

        if path_geometry_file_or_data is not None:
            self.load_path_geometry(path_geometry_file_or_data)

        # 3. Начальное состояние
        self.initial_s = float(initial_s)
        self.reset()

    def set_mass(self, mass_kg: float):
        """Возможность гибко изменить расчетную массу вагона на лету."""
        self.model.set_mass(mass_kg)

    def load_path_geometry(self, path_file_or_data: Any):
        """Загрузка карты пути из JSON-файла или словаря."""
        if isinstance(path_file_or_data, (str, Path)):
            p = Path(path_file_or_data)
            if not p.exists():
                return
            data = json.loads(p.read_text(encoding="utf-8"))
        elif isinstance(path_file_or_data, dict):
            data = path_file_or_data
        elif isinstance(path_file_or_data, tuple) and len(path_file_or_data) >= 3:
            # Уже распакованные массивы
            self.s_map = np.asarray(path_file_or_data[0], dtype=np.float64)
            self.xs = np.asarray(path_file_or_data[1], dtype=np.float64)
            self.ys = np.asarray(path_file_or_data[2], dtype=np.float64)
            self.zs = np.asarray(path_file_or_data[3], dtype=np.float64)
            self.slopes = np.asarray(path_file_or_data[4], dtype=np.float64)
            self.curvs = np.asarray(path_file_or_data[5], dtype=np.float64)
            self.has_map = True
            return
        else:
            return

        pts = data.get("points", [])
        if not pts:
            return

        xs = np.array([pt["x"] for pt in pts], dtype=np.float64)
        ys = np.array([pt["y"] for pt in pts], dtype=np.float64)
        zs = np.array([pt.get("z", 0.0) for pt in pts], dtype=np.float64)
        curvs = np.array([abs(float(pt.get("curv", 0.0))) for pt in pts], dtype=np.float64)

        # 3D кумулятивное расстояние
        dxs = np.diff(xs)
        dys = np.diff(ys)
        dzs = np.diff(zs)
        dists = np.sqrt(dxs ** 2 + dys ** 2 + dzs ** 2)
        
        # Защита от нулевых шагов
        dists_safe = np.where(dists < 1e-4, 1e-4, dists)
        slopes_segment = (dzs / dists_safe) * 1000.0  # уклон в тысячных (‰)
        
        s_map = np.insert(np.cumsum(dists), 0, 0.0)
        slopes = np.insert(slopes_segment, 0, slopes_segment[0] if len(slopes_segment) > 0 else 0.0)

        self.s_map = s_map
        self.xs = xs
        self.ys = ys
        self.zs = zs
        self.slopes = slopes
        self.curvs = curvs
        self.has_map = True

    def reset(self):
        """Сброс состояния модели."""
        self.v: float = 0.0
        self.s: float = self.initial_s
        self.a: float = 0.0
        self.f_prop: float = 0.0
        self.w_res: float = 0.0
        self.current_cmd: int = 0
        self.last_time: Optional[float] = None
        self.fsm_state: str = "STOPPED"
        self.history: List[DynamicOdometryState] = []

    def get_track_properties(self, s: float) -> Tuple[float, float, float, float, float]:
        """
        Возвращает параметры пути по координате s:
        (x, y, z, slope_permille, curvature)
        """
        if not self.has_map:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        s_clamped = min(max(s, float(self.s_map[0])), float(self.s_map[-1]))
        x = float(np.interp(s_clamped, self.s_map, self.xs))
        y = float(np.interp(s_clamped, self.s_map, self.ys))
        z = float(np.interp(s_clamped, self.s_map, self.zs))
        slope = float(np.interp(s_clamped, self.s_map, self.slopes))
        curv = float(np.interp(s_clamped, self.s_map, self.curvs))
        return x, y, z, slope, curv

    def update_driver_cmd(self, timestamp: float, cmd: int) -> DynamicOdometryState:
        """Обновление команды машиниста и продвижение симуляции до момента timestamp."""
        # 1. Сначала шагаем до времени прихода нового cmd со старой командой
        state = self.step_to(timestamp)
        # 2. Обновляем текущую команду
        self.current_cmd = int(cmd)
        return state

    def step_to(self, timestamp: float) -> DynamicOdometryState:
        """Продвижение физического интегратора от self.last_time до timestamp."""
        if self.last_time is None:
            self.last_time = timestamp
            x, y, z, slope, curv = self.get_track_properties(self.s)
            state = DynamicOdometryState(
                timestamp=timestamp,
                v_est=self.v,
                v_kmh=self.v * 3.6,
                acceleration=0.0,
                distance=self.s,
                x=x, y=y, z=z,
                driver_cmd=self.current_cmd,
                f_prop=0.0,
                w_res=0.0,
                slope_permille=slope,
                curvature=curv,
                fsm_state=self.fsm_state
            )
            self.history.append(state)
            return state

        total_dt = timestamp - self.last_time
        if total_dt <= 0:
            return self.history[-1]

        # Защита от гигантских временных скачков (например, пауза более 3 сек)
        if total_dt > 3.0:
            total_dt = 0.05

        # Суб-шаги интегрирования для повышенной точности и гладкости
        n_steps = max(1, int(np.ceil(total_dt / self.substep_dt)))
        dt = total_dt / n_steps

        for _ in range(n_steps):
            self._integration_step(dt)

        self.last_time = timestamp
        x, y, z, slope, curv = self.get_track_properties(self.s)

        state = DynamicOdometryState(
            timestamp=timestamp,
            v_est=self.v,
            v_kmh=self.v * 3.6,
            acceleration=self.a,
            distance=self.s,
            x=x, y=y, z=z,
            driver_cmd=self.current_cmd,
            f_prop=self.f_prop,
            w_res=self.w_res,
            slope_permille=slope,
            curvature=curv,
            fsm_state=self.fsm_state
        )
        self.history.append(state)
        return state

    def _integration_step(self, dt: float):
        """Один суб-шаг численного интегрирования продольной динамики."""
        cmd = self.current_cmd
        _, _, _, slope, curv = self.get_track_properties(self.s)

        # 1. Логика конечного автомата (FSM)
        if self.fsm_state == "STOPPED":
            self.v = 0.0
            self.a = 0.0
            self.f_prop = 0.0
            self.w_res = 0.0
            
            # Трогание с места возможно только при подаче тяги (cmd > 0)
            if cmd > 0:
                self.fsm_state = "MOTION"
            else:
                # На остановке вагон удерживается стояночным механическим тормозом
                return

        # 2. Состояние движения (MOTION)
        # Расчет текущих ускорений методом предиктора-корректора (Heun / трапеция)
        a1, f1, w1 = self.model.calc_net_acceleration(cmd, self.v, slope, curv)
        
        # Прогноз скорости
        v_pred = max(0.0, self.v + a1 * dt)
        a2, f2, w2 = self.model.calc_net_acceleration(cmd, v_pred, slope, curv)
        
        a_eff = 0.5 * (a1 + a2)
        v_next = self.v + a_eff * dt

        # 3. Контроль торможения до нуля и переходов в STOPPED
        if cmd <= 0 and v_next <= 0.0:
            # Трамвай затормозил до полной остановки
            v_avg = 0.5 * self.v
            self.s += v_avg * dt
            self.v = 0.0
            self.a = 0.0
            self.f_prop = 0.0
            self.w_res = 0.0
            self.fsm_state = "STOPPED"
        else:
            # Ограничение физической скорости (не может быть отрицательной, ограничена v_max)
            v_next_clamped = min(self.params.v_max, max(0.0, v_next))
            v_avg = 0.5 * (self.v + v_next_clamped)
            self.s += v_avg * dt
            self.v = v_next_clamped
            self.a = a_eff
            self.f_prop = 0.5 * (f1 + f2)
            self.w_res = 0.5 * (w1 + w2)

    def get_arrays(self) -> Dict[str, np.ndarray]:
        """Возвращает историю в виде словаря numpy-массивов."""
        if not self.history:
            return {}
        return {
            "timestamp": np.array([s.timestamp for s in self.history]),
            "v_est": np.array([s.v_est for s in self.history]),
            "v_kmh": np.array([s.v_kmh for s in self.history]),
            "acceleration": np.array([s.acceleration for s in self.history]),
            "distance": np.array([s.distance for s in self.history]),
            "x": np.array([s.x for s in self.history]),
            "y": np.array([s.y for s in self.history]),
            "z": np.array([s.z for s in self.history]),
            "driver_cmd": np.array([s.driver_cmd for s in self.history]),
            "slope": np.array([s.slope_permille for s in self.history]),
            "curvature": np.array([s.curvature for s in self.history]),
        }

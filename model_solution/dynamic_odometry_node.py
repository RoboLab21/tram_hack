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
    from .tram_dynamic_model import TramDynamicModel, TramParameters, MassNoiseParameters, SpeedRegimeParameters
except ImportError:
    from tram_dynamic_model import TramDynamicModel, TramParameters, MassNoiseParameters, SpeedRegimeParameters



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
    mass_kg: float = 24500.0# Текущая масса вагона с учетом стохастического шума (кг)
    speed_limit_kmh: float = 60.0 # Действующее ограничение скорости на участке пути (км/ч)
    radius_m: float = 10000.0     # Текущий радиус кривизны пути R = 1 / kappa (м)


class DynamicOdometryNode:
    """
    Нода счисления пути на основе дифференциальных уравнений движения трамвая «Львёнок».
    Поддерживает скоростные режимы (городской 60 км/ч, повороты 10-20 км/ч, разворотные кольца),
    стохастическую модель шума массы (случайное блуждание на остановках)
    и непрерывный шум процесса (для фильтров Кальмана / Particle Filter).
    """
    def __init__(
        self,
        path_geometry_file_or_data: Optional[Any] = None,
        tram_params: Optional[TramParameters] = None,
        noise_params: Optional[MassNoiseParameters] = None,
        speed_params: Optional[SpeedRegimeParameters] = None,
        random_seed: Optional[int] = None,
        initial_s: float = 0.0,
        substep_dt: float = 0.01  # макс. шаг внутреннего интегрирования для высокой точности
    ):
        # 1. Физическая модель вагона (по умолчанию масса 24.5 т)
        self.params = tram_params if tram_params is not None else TramParameters(mass_kg=24500.0)
        self.model = TramDynamicModel(self.params)
        self.noise_params = noise_params
        self.speed_params = speed_params if speed_params is not None else SpeedRegimeParameters()
        self.rng = np.random.default_rng(random_seed)
        self.substep_dt = substep_dt
        self.stopped_duration: float = 0.0

        # 2. Загрузка карты пути (pathgrath)
        self.s_map: np.ndarray = np.array([0.0, 100000.0])
        self.xs: np.ndarray = np.array([0.0, 0.0])
        self.ys: np.ndarray = np.array([0.0, 0.0])
        self.zs: np.ndarray = np.array([0.0, 0.0])
        self.slopes: np.ndarray = np.array([0.0, 0.0])
        self.curvs: np.ndarray = np.array([0.0, 0.0])
        self.rads: np.ndarray = np.array([10000.0, 10000.0])
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
            self.s_map = np.asarray(path_file_or_data[0], dtype=np.float64)
            self.xs = np.asarray(path_file_or_data[1], dtype=np.float64)
            self.ys = np.asarray(path_file_or_data[2], dtype=np.float64)
            self.zs = np.asarray(path_file_or_data[3], dtype=np.float64)
            self.slopes = np.asarray(path_file_or_data[4], dtype=np.float64)
            self.curvs = np.asarray(path_file_or_data[5], dtype=np.float64)
            if len(path_file_or_data) >= 7:
                self.rads = np.asarray(path_file_or_data[6], dtype=np.float64)
            else:
                self.rads = np.where(self.curvs > 1e-4, 1.0 / np.maximum(self.curvs, 1e-4), 10000.0)
            self.has_map = True
            return
        else:
            return

        pts = data.get("points", [])
        if not pts:
            return

        xs = np.array([pt["x"] for pt in pts], dtype=np.float64)
        ys = np.array([pt["y"] for pt in pts], dtype=np.float64)
        zs_raw = np.array([pt.get("z", 0.0) for pt in pts], dtype=np.float64)
        curvs_raw = np.array([abs(float(pt.get("curv", 0.0))) for pt in pts], dtype=np.float64)

        # 3D кумулятивное расстояние
        dxs = np.diff(xs)
        dys = np.diff(ys)
        dzs = np.diff(zs_raw)
        dists = np.sqrt(dxs ** 2 + dys ** 2 + dzs ** 2)
        s_map = np.insert(np.cumsum(dists), 0, 0.0)

        # 1. Фильтрация профиля высот Z и расчет физического уклона i(s) (в промилле ‰)
        # Окно сглаживания 20м (сопоставимо с длиной 16.7м трамвая) устраняет шум оцифровки
        w_z = min(20, max(3, len(zs_raw) // 10))
        zs_padded = np.pad(zs_raw, w_z, mode="edge")
        zs_smooth = np.convolve(zs_padded, np.ones(w_z) / w_z, mode="same")[w_z:-w_z]
        slopes = np.gradient(zs_smooth, s_map) * 1000.0

        # 2. Фильтрация кривизны и расчет точного радиуса R(s) = 1 / max(|kappa|, 1e-5)
        w_c = min(15, max(3, len(curvs_raw) // 10))
        curvs_padded = np.pad(curvs_raw, w_c, mode="edge")
        curvs_smooth = np.convolve(curvs_padded, np.ones(w_c) / w_c, mode="same")[w_c:-w_c]
        rads = np.where(curvs_smooth > 1e-4, 1.0 / curvs_smooth, 10000.0)

        self.s_map = s_map
        self.xs = xs
        self.ys = ys
        self.zs = zs_smooth
        self.slopes = slopes
        self.curvs = curvs_smooth
        self.rads = rads
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

    def get_track_properties(self, s: float) -> Tuple[float, float, float, float, float, float]:
        """
        Возвращает параметры пути по координате s с учетом двухопорной базы вагона (B = 7.5 м):
        (x, y, z, slope_permille, curvature, radius_m)
        Трамвай опирается на 2 тележки:
        - передняя тележка: s1 = s + B / 2 = s + 3.75 м
        - задняя тележка:   s2 = s - B / 2 = s - 3.75 м
        Результирующий уклон и кривизна кузова определяются полусуммой реакций двух тележек.
        """
        if not self.has_map:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 10000.0

        s_min = float(self.s_map[0])
        s_max = float(self.s_map[-1])
        s_clamped = min(max(s, s_min), s_max)

        x = float(np.interp(s_clamped, self.s_map, self.xs))
        y = float(np.interp(s_clamped, self.s_map, self.ys))
        z = float(np.interp(s_clamped, self.s_map, self.zs))

        # База вагона B = 7.5 м между шкворнями тележек
        half_b = 0.5 * getattr(self.params, "car_wheelbase_m", 7.5)
        s1 = min(max(s + half_b, s_min), s_max)
        s2 = min(max(s - half_b, s_min), s_max)

        slope1 = float(np.interp(s1, self.s_map, self.slopes))
        slope2 = float(np.interp(s2, self.s_map, self.slopes))
        eff_slope = 0.5 * (slope1 + slope2)

        curv1 = float(np.interp(s1, self.s_map, self.curvs))
        curv2 = float(np.interp(s2, self.s_map, self.curvs))
        eff_curv = 0.5 * (curv1 + curv2)

        if eff_curv > 1e-5:
            eff_rad = min(10000.0, 1.0 / eff_curv)
        else:
            eff_rad = 10000.0

        return x, y, z, eff_slope, eff_curv, eff_rad

    def get_speed_limit(self, s: float, rad: float) -> float:
        """
        Возвращает допустимую скорость движения (м/с) с учетом скоростных режимов и радиуса R:
        - городское ограничение: до 60 км/ч на прямых участках (R >= 250 м);
        - кривые и повороты: плавное ограничение от 5-10 км/ч (R <= 38м) до 60 км/ч.
        """
        if self.speed_params is None or not self.speed_params.enable_speed_regimes:
            return float(self.params.v_max)

        return float(self.speed_params.calc_speed_limit(rad))


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
            x, y, z, slope, curv, rad = self.get_track_properties(self.s)
            v_lim = self.get_speed_limit(self.s, rad)
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
                fsm_state=self.fsm_state,
                mass_kg=float(self.model.params.mass_kg),
                speed_limit_kmh=float(v_lim * 3.6),
                radius_m=rad
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
        x, y, z, slope, curv, rad = self.get_track_properties(self.s)
        v_lim = self.get_speed_limit(self.s, rad)

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
            fsm_state=self.fsm_state,
            mass_kg=float(self.model.params.mass_kg),
            speed_limit_kmh=float(v_lim * 3.6),
            radius_m=rad
        )
        self.history.append(state)
        return state

    def _integration_step(self, dt: float):
        """Один суб-шаг численного интегрирования продольной динамики."""
        cmd = self.current_cmd
        _, _, _, slope, curv, rad = self.get_track_properties(self.s)

        # 1. Логика конечного автомата (FSM)
        if self.fsm_state == "STOPPED":
            self.v = 0.0
            self.a = 0.0
            self.f_prop = 0.0
            self.w_res = 0.0
            self.stopped_duration += dt
            
            # Трогание с места: по тяге (cmd > 0) ИЛИ при отпускании тормоза (cmd == 0) на крутом уклоне (slope < -3.0‰)
            can_roll_down = (cmd == 0 and slope < -3.0)
            if cmd > 0 or can_roll_down:
                self.fsm_state = "MOTION"
                # Если стоянка была достаточно длинной (остановка) — разыгрываем обмен пассажирами
                if (
                    self.noise_params is not None
                    and self.noise_params.enable_stop_noise
                    and self.stopped_duration >= self.noise_params.min_stop_duration_s
                ):
                    # Физическое ограничение пассажирообмена через двери вагона (4 проёма по 730 мм)
                    # 1 человек проходит через проём 730 мм за ~1.3 сек
                    t_boarding = max(0.0, self.stopped_duration - 3.0)
                    door_rate = getattr(self.params, "door_count", 4) / 1.3
                    n_pass_max = max(1.0, t_boarding * door_rate)
                    delta_m_phys_max = n_pass_max * 75.0  # макс. физическое изменение массы (кг)

                    delta_m_raw = float(self.rng.normal(0.0, self.noise_params.m_sigma_stop))
                    delta_m = float(np.clip(delta_m_raw, -delta_m_phys_max, delta_m_phys_max))
                    new_m = float(np.clip(
                        self.model.params.mass_kg + delta_m,
                        self.noise_params.m_min,
                        self.noise_params.m_max
                    ))
                    self.model.set_mass(new_m)
                self.stopped_duration = 0.0
            else:
                # На остановке вагон удерживается стояночным механическим тормозом
                return

        # 2. Состояние движения (MOTION)
        v_lim = self.get_speed_limit(self.s, rad)

        # Расчет текущих ускорений методом предиктора-корректора (Heun / трапеция)
        a1, f1, w1 = self.model.calc_net_acceleration(cmd, self.v, slope, curv)

        # Мягкий регулятор тяги по скорости (Speed Governor)
        if cmd > 0 and self.v > 0.90 * v_lim:
            throttle = max(0.0, (v_lim - self.v) / (0.10 * v_lim))
            f1 *= throttle
            a1 = (f1 - w1) / self.model.effective_mass

        # Автоматическое удержание скорости на спусках при превышении лимита
        if self.v > v_lim + 0.3:
            f_hold = -self.model.effective_mass * min(1.0, (self.v - v_lim) / 0.5) * 0.8
            a1 += f_hold / self.model.effective_mass
        
        # Прогноз скорости
        v_pred = max(0.0, self.v + a1 * dt)
        a2, f2, w2 = self.model.calc_net_acceleration(cmd, v_pred, slope, curv)
        if cmd > 0 and v_pred > 0.90 * v_lim:
            throttle = max(0.0, (v_lim - v_pred) / (0.10 * v_lim))
            f2 *= throttle
            a2 = (f2 - w2) / self.model.effective_mass

        a_eff = 0.5 * (a1 + a2)
        if self.noise_params is not None and self.noise_params.enable_process_noise:
            a_eff += float(self.rng.normal(0.0, self.noise_params.accel_noise_sigma))

        # Ограничение рывка привода (Jerk limiter по ГОСТ 8802: |da/dt| <= 1.2 м/с³)
        max_delta_a = 1.20 * dt
        delta_a = a_eff - self.a
        if abs(delta_a) > max_delta_a:
            a_eff = self.a + np.sign(delta_a) * max_delta_a

        v_next = self.v + a_eff * dt

        # 3. Контроль торможения до нуля и переходов в STOPPED
        # Мягкий переход: в STOPPED переходим только если скорость погашена (v_next <= 0.05)
        # И (машинист держит активный тормоз cmd <= -4 ИЛИ стоянка без движения продолжается > 2.0 сек)
        if v_next <= 0.05 and (cmd <= -4 or self.stopped_duration > 2.0):
            # Трамвай затормозил до полной остановки
            v_avg = 0.5 * max(0.0, self.v)
            self.s += v_avg * dt
            self.v = 0.0
            self.a = 0.0
            self.f_prop = 0.0
            self.w_res = 0.0
            self.fsm_state = "STOPPED"
            self.stopped_duration = 0.0
        elif v_next <= 0.0:
            if slope < -2.0:
                # Скатывание под уклон по инерции
                self.v = max(0.1, self.v)
                self.s += self.v * dt
            else:
                self.v = 0.0
                self.a = 0.0
                self.stopped_duration += dt
        else:
            self.stopped_duration = 0.0
            # Ограничение физической скорости (не может быть отрицательной, ограничена v_lim)
            v_next_clamped = min(v_lim, max(0.0, v_next))
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
            "radius": np.array([s.radius_m for s in self.history]),
            "mass_kg": np.array([s.mass_kg for s in self.history]),
            "speed_limit_kmh": np.array([s.speed_limit_kmh for s in self.history]),
        }

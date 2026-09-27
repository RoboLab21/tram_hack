"""
Модуль фильтрованной резервной одометрии беспилотного трамвая с автокалибровкой
общего износа колес по цифровой карте пути pathgrath, фильтром торможения
и компенсацией коротких поворотов (v5, БЕЗ ИСПОЛЬЗОВАНИЯ GNSS).

Решение полностью автономно (не использует GNSS/IMU/лидар/камеру):
1. Фильтрует аппаратные сбои датчиков скорости (нули, выбросы, квадратичное расхождение dV^2).
2. Автоматически извлекает геометрию прямых участков и контрольных отрезков из цифровой карты пути (pathgrath).
3. Определяет общий масштабный коэффициент износа колес:
   k_scale = L_map / Delta_R_колес (где L_map — эталонная длина участка по карте пути).
4. Фильтр торможения при активном торможении (положение контроллера driver_cmd < 0):
   - Отсекает ползучий шум/дребезг колес перед и во время остановки (v < v_crawl).
   - Ограничивает физическое замедление при срыве в юз/блокировке колес (a < -a_brake_max).
   Положение контроллера водителя строго < 0 (при 0 — выбег/нейтраль без вмешательства).
5. Компенсация коротких / крутых поворотов по цифровой карте пути:
   - По накопленному расстоянию R извлекает локальную кривизну пути |kappa| из pathgrath.
   - В кривых малого радиуса (R <= 125 м, |kappa| > 0.008 м^-1) компенсирует забегание
     наружных колес жесткой колесной пары: V_corr = V * max(c_min, 1.0 - beta * (|kappa| - kappa_0)).
6. Интегрирует скорректированную скорость без систематического ухода по пробегу.
"""

import json
from dataclasses import dataclass
from typing import dict, Any, list, tuple
from collections import deque
from pathlib import Path
import numpy as np


@dataclass
class StraightSection:
    idx: int  # Номер контрольного прямого участка (1, 2, 3...)
    s_start: float  # Начальная путевая координата на карте (м)
    s_end: float  # Конечная путевая координата на карте (м)
    length: float  # Эталонная длина участка по карте (м)


@dataclass
class CalibratedOdometryState:
    timestamp: float  # Текущая метка времени (сек)
    v1_raw: float  # Исходная скорость передней тележки (м/с)
    v2_raw: float  # Исходная скорость задней тележки (м/с)
    v1_valid: bool  # Флаг исправности датчика 1
    v2_valid: bool  # Флаг исправности датчика 2
    v_est: float  # Отфильтрованная и масштабированная скорость с учетом износа (м/с)
    diff_sq: float  # Квадратичное отклонение между датчиками (v1 - v2)^2
    fault_type: (
        str  # "OK", "ZERO_V1", "ZERO_V2", "SPIKE_V1", "SPIKE_V2", "EXCESSIVE_DIFF"
    )
    delta_t: float  # Шаг времени (сек)
    delta_r: float  # Приращение скорректированного пути (м)
    distance: float  # Накопленный путь R (м)
    in_straight: bool  # Флаг нахождения на калибровочном участке
    straight_id: int  # Номер активного участка карты (0 - вне калибровки)
    k_scale: float  # Общий масштабный коэффициент износа колес (V_true / V_колес)
    wear_pct: float  # Оценочный износ бандажей в процентах: (k_scale - 1) * 100%
    driver_cmd: int  # Актуальное положение ручки контроллера водителя (-15 .. +15)
    is_braking: bool  # Флаг активного торможения (строго driver_cmd < 0)
    path_curvature: float  # Локальная кривизна пути |kappa| по карте (1/м)
    curve_factor: float  # Коэффициент компенсации кривизны (c_curve <= 1.0)


def load_path_geometry(
    path_file_or_data: Any,
    max_curvature: float = 0.0005,  # Радиус кривизны > 2000 м (|curv| < 0.0005 1/м)
    min_length_m: float = 70.0,  # Минимальная длина прямого отрезка для калибровки (м)
    top_n: int
    | None = None,  # None = все прямые участки на всем пути маршрута (Вариант 3)
) -> tuple[list[StraightSection], np.ndarray, np.ndarray]:
    """
    Загружает геометрию цифрового пути из pathgrath:
    Возвращает :
      - straight_sections: список контрольных прямых участков для автокалибровки износа колес на всем пути.
      - s_map: массив кумулятивного расстояния вдоль пути (м).
      - curv_map: массив модуля кривизны пути |kappa| (1/м).
    """
    if isinstance(path_file_or_data, (str, Path)):
        p = Path(path_file_or_data)
        d = json.loads(p.read_text(encoding="utf-8"))
    elif isinstance(path_file_or_data, dict):
        d = path_file_or_data
    else:
        raise TypeError("path_file_or_data должен быть путем к файлу или словарем")

    pts = d.get("points", [])
    if not pts:
        return [], np.array([]), np.array([])

    xs = np.array([pt["x"] for pt in pts], dtype=np.float64)
    ys = np.array([pt["y"] for pt in pts], dtype=np.float64)
    zs = np.array([pt.get("z", 0.0) for pt in pts], dtype=np.float64)
    curvs = np.array([abs(float(pt.get("curv", 0.0))) for pt in pts], dtype=np.float64)

    # 3D кумулятивное расстояние вдоль точек графа
    dists = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2 + np.diff(zs) ** 2)
    s_map = np.insert(np.cumsum(dists), 0, 0.0)

    # Поиск прямых участков
    is_straight = curvs < max_curvature
    sections: list[StraightSection] = []
    in_seg = False
    start_idx = 0

    for i in range(len(is_straight)):
        if is_straight[i] and not in_seg:
            in_seg = True
            start_idx = i
        elif not is_straight[i] and in_seg:
            in_seg = False
            length = float(s_map[i - 1] - s_map[start_idx])
            if length >= min_length_m:
                idx = len(sections) + 1
                sections.append(
                    StraightSection(
                        idx, float(s_map[start_idx]), float(s_map[i - 1]), length
                    )
                )
                if top_n is not None and len(sections) >= top_n:
                    break

    if in_seg and (top_n is None or len(sections) < top_n):
        length = float(s_map[-1] - s_map[start_idx])
        if length >= min_length_m:
            idx = len(sections) + 1
            sections.append(
                StraightSection(idx, float(s_map[start_idx]), float(s_map[-1]), length)
            )

    return sections, s_map, curvs


def extract_straight_sections(
    path_file_or_data: Any,
    max_curvature: float = 0.0005,
    min_length_m: float = 70.0,
    top_n: int | None = None,
) -> list[StraightSection]:
    """Возвращает список контрольных прямых участков вдоль всего пути (top_n=None по умолчанию)."""
    sections, _, _ = load_path_geometry(
        path_file_or_data, max_curvature, min_length_m, top_n
    )
    return sections


class PathCalibratedDeadReckoningNode:
    """
    Нода фильтрованной резервной одометрии беспилотного трамвая:
    - Квадратичная фильтрация аппаратных сбоев датчиков скорости (v1, v2).
    - Автокалибровка коэффициента общего износа бандажей (k_scale) по цифровой карте пути.
    - Фильтр торможения при активном торможении водителя/автопилота (driver_cmd < 0):
      отсечка шума остановки (v < v_crawl) и ограничение замедления при юзе.
    - Компенсация коротких / крутых поворотов пути по кривизне из цифровой карты пути.

    НЕ ИСПОЛЬЗУЕТ GNSS / IMU. Полностью соответствует регламенту хакатона (п. 3.2).
    """

    def __init__(
        self,
        path_file: Path | None = None,
        path_data: dict[str, Any | None] = None,
        straight_sections: list[StraightSection | None] = None,
        path_geometry: tuple[np.ndarray, np.ndarray | None] = None,
        k_scale_init: float = 1.0,
        input_in_kmh: bool = True,
        integration_method: str = "trapezoidal",
        diff_thresh_kmh: float = 3.6,
        a_max_ms2: float = 2.5,
        v_max_kmh: float = 75.0,
        zero_thresh_kmh: float = 1.0,
        window_size: int = 15,
        # Параметры сглаживания скорости:
        velocity_filter: str = "mean",  # "mean" (мгновенная), "sma" (Simple MA), "ema" (Exponential MA)
        window_size_speed: int = 5,  # Количество точек N для сглаживания скорости
        alpha_speed: float | None = None,  # Коэффициент EMA (по умолчанию 2 / (N + 1))
        # Параметры тормозного фильтра:
        enable_brake_filter: bool = True,
        brake_crawl_thresh_kmh: float = 0.8,  # Порог отсечения ползучего хода при торможении (м/с -> 0.22)
        brake_decel_limit_ms2: float = 2.5,  # Физический предел замедления сталь-рельс при торможении
        # Параметры компенсации кривизны поворотов:
        enable_curve_compensation: bool = True,
        curve_thresh_curv: float = 0.008,  # Порог кривизны для поворотов малого радиуса (R <= 125 м)
        curve_beta: float = 0.20,  # Фактор компенсации забегания колес в кривых
        curve_min_factor: float = 0.85,  # Максимальное снижение скорости в предельно резком повороте
    ):
        self.input_in_kmh = input_in_kmh
        self.scale_factor = (1.0 / 3.6) if input_in_kmh else 1.0
        self.integration_method = integration_method
        self.k_scale_init = float(k_scale_init)

        self.diff_sq_thresh = (diff_thresh_kmh / 3.6) ** 2
        self.a_max_sq = (a_max_ms2) ** 2
        self.v_max_ms = v_max_kmh / 3.6
        self.zero_thresh_ms = zero_thresh_kmh / 3.6
        self.window_size = window_size

        # Параметры тормозного фильтра
        self.enable_brake_filter = enable_brake_filter
        self.brake_crawl_thresh_ms = brake_crawl_thresh_kmh / 3.6
        self.brake_decel_limit_ms2 = brake_decel_limit_ms2

        # Параметры сглаживания скорости (MEAN / SMA / EMA)
        self.velocity_filter = velocity_filter.lower()
        self.window_size_speed = max(1, int(window_size_speed))
        if alpha_speed is not None:
            self.alpha_speed = float(alpha_speed)
        else:
            self.alpha_speed = 2.0 / (self.window_size_speed + 1.0)

        # Параметры компенсации поворотов
        self.enable_curve_compensation = enable_curve_compensation
        self.curve_thresh_curv = curve_thresh_curv
        self.curve_beta = curve_beta
        self.curve_min_factor = curve_min_factor

        # Загрузка карты пути (прямые участки и профиль кривизны)
        self.straight_sections: list[StraightSection] = []
        self.s_map: np.ndarray | None = None
        self.curv_map: np.ndarray | None = None

        if path_geometry is not None:
            self.s_map, self.curv_map = path_geometry
        if straight_sections is not None:
            self.straight_sections = straight_sections

        source = path_data if path_data is not None else path_file
        if source is not None:
            secs, s_arr, c_arr = load_path_geometry(source)
            if not self.straight_sections:
                self.straight_sections = secs
            if self.s_map is None:
                self.s_map = s_arr
                self.curv_map = c_arr

        self.reset()

    def set_path(self, path_file_or_data: Any):
        """Динамическая загрузка геометрии цифровой карты пути."""
        self.straight_sections, self.s_map, self.curv_map = load_path_geometry(
            path_file_or_data
        )

    def set_path_geometry(
        self,
        straight_sections: list[StraightSection],
        s_map: np.ndarray,
        curv_map: np.ndarray,
    ):
        """Прямая установка предрассчитанной геометрии пути."""
        self.straight_sections = straight_sections
        self.s_map = s_map
        self.curv_map = curv_map

    def reset(self):
        """Сброс состояния ноды одометрии."""
        self.R: float = 0.0  # Скорректированный пройденный путь (м)
        self.R_raw: float = 0.0  # Некалиброванный сырой путь колес (м)
        self.last_time: float | None = None
        self.v1: float = 0.0
        self.v2: float = 0.0
        self.v_est: float = 0.0
        self.last_v_est: float = 0.0

        # Контроллер водителя и тормозной контур
        self.driver_cmd: int = 0  # Текущее положение ручки (-15 .. +15)
        self.is_braking: bool = False  # Флаг торможения (строго driver_cmd < 0)

        # Кривизна и фактор поворота
        self.path_curvature: float = 0.0
        self.curve_factor: float = 1.0

        # Автокалибровка ОБЩЕГО износа колес по прямым участкам карты
        self.k_scale: float = self.k_scale_init
        self.is_calibrated: bool = self.k_scale_init != 1.0
        self.scale_samples: list[float] = []  # Оценки k_scale по участкам карты
        self.active_straight_id: int = 0
        self.section_entry_r: dict[int, float] = {}
        self.section_measurements: dict[int, dict[str, float]] = {}

        # Состояние сглаживания скорости (SMA / EMA)
        self.sma_speed_buffer: deque = deque(maxlen=self.window_size_speed)
        self.v_ema_speed: float | None = None

        self.history_v1 = deque(maxlen=self.window_size)
        self.history_v2 = deque(maxlen=self.window_size)
        self.history_vest = deque(maxlen=self.window_size)

    def update_driver_cmd(self, position: int):
        """
        Обновляет положение контроллера водителя (/vehicle/driver_position_cmd).
        Положение < 0: активное торможение.
        Положение == 0: выбег (нейтраль).
        Положение > 0: тяга.
        """
        self.driver_cmd = int(position)
        self.is_braking = self.driver_cmd < 0

    def update_front(
        self, timestamp: float, velocity_raw: float
    ) -> CalibratedOdometryState | None:
        """Обновление скорости передней тележки."""
        self.v1 = float(velocity_raw) * self.scale_factor
        return self._step(timestamp)

    def update_rear(
        self, timestamp: float, velocity_raw: float
    ) -> CalibratedOdometryState | None:
        """Обновление скорости задней тележки."""
        self.v2 = float(velocity_raw) * self.scale_factor
        return self._step(timestamp)

    def step(
        self,
        timestamp: float,
        v1_raw: float,
        v2_raw: float,
        driver_cmd: int | None = None,
    ) -> CalibratedOdometryState:
        """Синхронный шаг одометрии по двум скоростям и опциональной команде водителя."""
        if driver_cmd is not None:
            self.driver_cmd = int(driver_cmd)
            self.is_braking = self.driver_cmd < 0
        self.v1 = float(v1_raw) * self.scale_factor
        self.v2 = float(v2_raw) * self.scale_factor
        return self._step(timestamp)

    def get_path_curvature(self, distance_m: float) -> float:
        """Возвращает локальную кривизну пути |kappa| (1/м) по текущей путевой координате."""
        if self.s_map is not None and self.curv_map is not None and len(self.s_map) > 0:
            return float(np.interp(distance_m, self.s_map, self.curv_map))
        return 0.0

    def _check_straight_section(self, r_current: float) -> tuple[bool, int]:
        """Проверяет, попадает ли текущий пройденный путь в один из прямых участков карты."""
        for s in self.straight_sections:
            if s.s_start <= r_current <= s.s_end:
                return True, s.idx
        return False, 0

    def _step(self, timestamp: float) -> CalibratedOdometryState:
        self.is_braking = self.driver_cmd < 0
        in_straight, straight_id = self._check_straight_section(self.R_raw)

        if self.last_time is None:
            self.last_time = timestamp
            v_raw = (self.v1 + self.v2) / 2.0
            self.v_est = self.k_scale * v_raw
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
                k_scale=self.k_scale,
                wear_pct=(self.k_scale - 1.0) * 100.0,
                driver_cmd=self.driver_cmd,
                is_braking=self.is_braking,
                path_curvature=0.0,
                curve_factor=1.0,
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
            v1_is_zero = self.v1 < self.zero_thresh_ms
            v2_is_zero = self.v2 < self.zero_thresh_ms

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
        # 2. АВТОКАЛИБРОВКА ОБЩЕГО ИЗНОСА КОЛЕС ПО КАРТЕ ПУТИ (БЕЗ GNSS)
        # ---------------------------------------------------------------------
        if in_straight and self.active_straight_id == 0:
            self.active_straight_id = straight_id
            self.section_entry_r[straight_id] = self.R_raw

        if (not in_straight and self.active_straight_id != 0) or (
            in_straight and straight_id != self.active_straight_id
        ):
            finished_id = self.active_straight_id
            if finished_id in self.section_entry_r:
                r_in = self.section_entry_r[finished_id]
                delta_wheels = self.R_raw - r_in

                matched_sec = next(
                    (s for s in self.straight_sections if s.idx == finished_id), None
                )
                if (
                    matched_sec is not None
                    and matched_sec.length > 20.0
                    and delta_wheels > 10.0
                ):
                    l_map = matched_sec.length
                    k_meas = l_map / delta_wheels

                    if 0.97 <= k_meas <= 1.03:
                        self.scale_samples.append(k_meas)
                        self.section_measurements[finished_id] = {
                            "l_map": l_map,
                            "delta_wheels": delta_wheels,
                            "k_meas": k_meas,
                            "wear_pct": (k_meas - 1.0) * 100.0,
                        }
                        self.k_scale = float(np.median(self.scale_samples))
                        self.is_calibrated = True

            if in_straight:
                self.active_straight_id = straight_id
                self.section_entry_r[straight_id] = self.R_raw
            else:
                self.active_straight_id = 0

        # ---------------------------------------------------------------------
        # 3. СЫРАЯ БАЗОВАЯ СКОРОСТЬ
        # ---------------------------------------------------------------------
        if v1_valid and v2_valid:
            v_raw = (self.v1 + self.v2) / 2.0
        elif v1_valid and not v2_valid:
            v_raw = self.v1
        elif v2_valid and not v1_valid:
            v_raw = self.v2
        else:
            v_raw = self.last_v_est / max(self.k_scale, 0.01)

        v_filt = v_raw

        # ---------------------------------------------------------------------
        # 4. ФИЛЬТР ТОРМОЖЕНИЯ (СТРОГО driver_cmd < 0)
        # ---------------------------------------------------------------------
        # При driver_cmd == 0 (выбег) фильтр не вмешивается.
        # При driver_cmd < 0 (активное торможение):
        if self.enable_brake_filter and self.is_braking:
            # А. Отсечка ползучей скорости/дребезга перед и во время остановки:
            if v_filt < self.brake_crawl_thresh_ms:
                v_filt = 0.0
            # Б. Ограничение замедления при срыве колес в юз (блокировке под тормозом):
            if dt > 0.005:
                a_meas = (v_filt - self.last_v_est) / dt
                if a_meas < -self.brake_decel_limit_ms2:
                    v_filt = max(0.0, self.last_v_est - self.brake_decel_limit_ms2 * dt)

        # Сглаживание скорости (SMA / EMA)
        if self.velocity_filter == "sma":
            self.sma_speed_buffer.append(v_filt)
            v_filt = float(np.mean(self.sma_speed_buffer))
        elif self.velocity_filter == "ema":
            if self.v_ema_speed is None:
                self.v_ema_speed = v_filt
            else:
                self.v_ema_speed = (
                    self.alpha_speed * v_filt
                    + (1.0 - self.alpha_speed) * self.v_ema_speed
                )
            v_filt = self.v_ema_speed

        # Масштабирование скорости на коэффициент общего износа бандажей:
        v_est = self.k_scale * v_filt

        # ---------------------------------------------------------------------
        # 5. КОМПЕНСАЦИЯ КОРОТКИХ И КРУТЫХ ПОВОРОТОВ (ПО КАРТЕ PATHGRATH)
        # ---------------------------------------------------------------------
        curve_factor = 1.0
        curv_val = 0.0
        if (
            self.enable_curve_compensation
            and self.s_map is not None
            and self.curv_map is not None
        ):
            curv_val = self.get_path_curvature(self.R)
            if curv_val > self.curve_thresh_curv:
                # Масштабный фактор снижения забегания наружных колес в кривой
                curve_factor = max(
                    self.curve_min_factor,
                    1.0 - self.curve_beta * (curv_val - self.curve_thresh_curv),
                )
                v_est *= curve_factor

        self.path_curvature = curv_val
        self.curve_factor = curve_factor

        self.history_v1.append(self.v1)
        self.history_v2.append(self.v2)
        self.history_vest.append(v_est)

        # ---------------------------------------------------------------------
        # 6. ИНТЕГРИРОВАНИЕ ПУТИ
        # ---------------------------------------------------------------------
        if dt > 0.0:
            if self.integration_method == "trapezoidal":
                v_eff = (self.last_v_est + v_est) / 2.0
            else:
                v_eff = v_est
            delta_r = v_eff * dt
            delta_r_raw = v_raw * dt

            self.R += delta_r
            self.R_raw += delta_r_raw
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
            k_scale=self.k_scale,
            wear_pct=(self.k_scale - 1.0) * 100.0,
            driver_cmd=self.driver_cmd,
            is_braking=self.is_braking,
            path_curvature=curv_val,
            curve_factor=curve_factor,
        )
        return state

    def get_calibration_report(self) -> dict[str, Any]:
        """Возвращает отчет об автокалибровке общего износа колес по карте пути."""
        return {
            "is_calibrated": self.is_calibrated,
            "final_k_scale": self.k_scale,
            "wear_pct": (self.k_scale - 1.0) * 100.0,
            "delta_dia_mm_est": 620.0 * (self.k_scale - 1.0),
            "measured_sections_count": len(self.scale_samples),
            "measured_sections": self.section_measurements,
            "enable_brake_filter": self.enable_brake_filter,
            "enable_curve_compensation": self.enable_curve_compensation,
        }

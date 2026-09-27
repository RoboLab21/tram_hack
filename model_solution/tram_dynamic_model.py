"""
Аналитическая физико-математическая модель продольной динамики трамвая «Львёнок» (71-911ЕМ).

Модель построена на принципах теоретической механики и Правил тяговых расчетов (ПТР):
1. 2-й закон Ньютона с учетом момента инерции вращающихся масс (xi = 1 + gamma = 1.10).
2. Аналитическая тяговая характеристика 4 асинхронных ТЭД по 72 кВт (288 кВт) с двумя зонами:
   - постоянного момента (ограничение по пусковому току и комфорту пассажиров a_max);
   - постоянной мощности (ограничение гиперболой F = P / v).
3. Трехкаскадная ступенчатая характеристика торможения по позициям ручки (-1..-15):
   - электродинамическое торможение (ЕДТ / рекуперация) на позициях -1..-6;
   - электромеханические дисковые тормоза на позициях -7..-12 (+ автоматическое дотормаживание при затухании ЕДТ на малой скорости);
   - магниторельсовый тормоз (МРТ / башмаки) на ступенях -13..-15.
4. Силы сопротивления движению:
   - W_0: основное удельное сопротивление (качение колес, аэродинамика) по формуле ПТР;
   - W_i: продольный уклон пути (dz/ds);
   - W_r: сопротивление в кривых малого радиуса (1 / R).
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import math


@dataclass
class TramParameters:
    # Масса вагона (по умолчанию 23.0 т = 23000 кг, легко настраивается)
    mass_kg: float = 23000.0
    
    # Коэффициент учета вращающихся масс (колесные пары, тяговые редукторы, роторы ТЭД)
    gamma: float = 0.10             # xi = 1 + gamma = 1.10
    
    # Силовая установка (4 ТЭД по 72 кВт)
    p_nominal_w: float = 288000.0   # 288 кВт
    efficiency: float = 0.90        # КПД инвертора и редуктора
    
    # Нормативные ускорения (по ГОСТ 8802 / ПТР)
    a_accel_max: float = 1.35        # максимальное ускорение разгона при cmd = +15 (м/с²)
    a_service_brake_max: float = 1.50 # максимальное служебное торможение при cmd = -12 (м/с²)
    a_emerg_brake_max: float = 2.70   # максимальное экстренное торможение при cmd = -15 с МРТ (м/с²)
    
    # Скорость затухания рекуперации (м/с)
    v_fade_edt: float = 1.5          # ~5.4 км/ч
    
    # Максимальная конструкционная скорость (м/с)
    v_max: float = 20.83             # 75 км/ч
    
    # Ускорение свободного падения
    g: float = 9.80665

    # Паспортные геометрические габариты ПК ТС 71-911ЕМ («Львёнок»)
    length_m: float = 16.4          # Длина кузова: 16 400 мм
    width_m: float = 2.5            # Ширина кузова: 2 500 мм
    height_m: float = 3.3           # Высота вагона: 3 300 мм
    car_wheelbase_m: float = 7.5    # База вагона (между шкворнями тележек): 7 500 мм
    bogie_wheelbase_m: float = 1.8  # Жесткая база тележки: 1 800 мм
    track_gauge_m: float = 1.524    # Ширина трамвайной колеи: 1 524 мм
    door_width_m: float = 0.73      # Проём дверей: 730 мм
    door_count: int = 4             # Количество дверей
    
    # Аэродинамические параметры и трение реборд
    c_x: float = 0.45               # Коэффициент аэродинамического лобового сопротивления
    air_density: float = 1.225      # Плотность воздуха (кг/м³)
    mu_flange: float = 0.16         # Коэффициент трения гребня колеса о рельс в кривой


@dataclass
class MassNoiseParameters:
    """
    Параметры распределения шума массы и случайных возмущений,
    откалиброванные по 56 валидным заездам датасета Мосгортранса.
    """
    enable_stop_noise: bool = True       # включить стохастическое изменение массы на остановках
    enable_process_noise: bool = False   # включить непрерывный шум ускорения (для EKF/UKF)
    m_min: float = 22000.0               # мин. масса тары (22 т)
    m_max: float = 28000.0               # макс. масса полной загрузки (28 т)
    m_mean: float = 24970.0              # средняя масса по датасету (24.97 т)
    m_sigma_trip: float = 2615.0         # СКО априорной массы рейса (2.62 т)
    m_sigma_stop: float = 1150.0         # СКО изменения массы за одну остановку (1.15 т)
    accel_noise_sigma: float = 0.327     # СКО непрерывного шума ускорения (м/с²)
    min_stop_duration_s: float = 5.0     # минимальное время стоянки для фиксации посадки/высадки


@dataclass
class SpeedRegimeParameters:
    """
    Параметры скоростных режимов трамвая:
    - городское ограничение скорости: не более 60 км/ч (16.67 м/с);
    - ограничение скорости в кривых и поворотах: 5-10 км/ч при R <= 38 м (по ПТЭ);
    - плавное увеличение скорости по мере роста радиуса кривой;
    - учет продольных уклонов пути (подъемы и спуски).
    """
    enable_speed_regimes: bool = True     # включить контроль скоростных режимов
    enable_slope_dynamics: bool = True    # включить гравитационную динамику уклонов пути
    max_city_speed_kmh: float = 55.0      # реальный эксплуатационный максимум (городской лимит)
    cruise_speed_kmh: float = 40.0        # типичная крейсерская скорость перегона (км/ч)
    min_turn_speed_kmh: float = 8.0       # скорость в крутых поворотах и на кольцах (5-10 км/ч)
    moderate_curve_speed_kmh: float = 20.0# скорость в средних поворотах R ~ 70м (км/ч)
    wide_curve_speed_kmh: float = 38.0    # скорость в пологих кривых R ~ 140м (км/ч)
    turn_radius_tight_m: float = 38.0     # пороговый радиус крутых поворотов (м)
    turn_radius_moderate_m: float = 70.0  # радиус средних поворотов (м)
    turn_radius_wide_m: float = 140.0     # радиус пологих кривых (м)
    turn_radius_straight_m: float = 250.0 # радиус выхода на прямую (м)

    def calc_speed_limit(self, radius_m: float) -> float:
        """
        Возвращает допустимую скорость движения (м/с) в зависимости от точного радиуса кривизны R.
        """
        if not self.enable_speed_regimes:
            return float(self.max_city_speed_kmh / 3.6)
        
        r_pts = [
            0.0,
            self.turn_radius_tight_m,
            self.turn_radius_moderate_m,
            self.turn_radius_wide_m,
            self.turn_radius_straight_m
        ]
        v_pts = [
            self.min_turn_speed_kmh / 3.6,
            self.min_turn_speed_kmh / 3.6,
            self.moderate_curve_speed_kmh / 3.6,
            self.wide_curve_speed_kmh / 3.6,
            self.max_city_speed_kmh / 3.6
        ]
        import numpy as np
        return float(np.interp(radius_m, r_pts, v_pts))


class TramDynamicModel:
    """
    Класс расчета сил и мгновенного ускорения трамвая «Львёнок».
    """
    def __init__(self, params: Optional[TramParameters] = None):
        self.params = params if params is not None else TramParameters()

    def set_mass(self, mass_kg: float):
        """Динамическое изменение расчетной массы вагона."""
        self.params.mass_kg = max(18000.0, min(35000.0, float(mass_kg)))

    @property
    def effective_mass(self) -> float:
        """Эффективная инерционная масса с учетом вращающихся деталей (кг)."""
        return self.params.mass_kg * (1.0 + self.params.gamma)

    def calc_traction_force(self, cmd: int, v: float) -> float:
        """
        Сила тяги F_k(cmd, v) в Ньютонах при cmd in [1..15].
        """
        if cmd <= 0:
            return 0.0

        cmd_ratio = min(1.0, max(0.0, cmd / 15.0))
        
        # 1. Ограничение по пусковому току / максимальному моменту на ободе
        f_max_base = self.effective_mass * self.params.a_accel_max
        f_base = f_max_base * cmd_ratio

        # 2. Ограничение по располагаемой электрической мощности
        p_eff = self.params.p_nominal_w * self.params.efficiency
        p_available = p_eff * cmd_ratio
        
        v_eff = max(v, 0.5)
        f_power = p_available / v_eff

        # Результирующая сила тяги — минимум двух ограничений
        f_traction = min(f_base, f_power)
        return float(f_traction)

    def calc_braking_force(self, cmd: int, v: float, slope_permille: float = 0.0) -> float:
        """
        Тормозная сила B_t(cmd, v, slope) в Ньютонах при cmd in [-15..-1].
        Реализует трехкаскадную систему торможения с учетом режима ретардера (удержания под уклон):
        - 1..3: Микрорегулирование скорости / выборка зазора (ЕДТ);
        - 4..7: Служебное электродинамическое торможение (ЕДТ);
        - 8: Специальный режим ретардера 71-911ЕМ (удержание постоянной скорости на спусках и круизе);
        - 9..12: Глубокое служебное торможение ЕДТ + механические дисковые тормоза тележек;
        - 13..15: Экстренное торможение с подключением башмаков МРТ.
        """
        if cmd >= 0:
            return 0.0

        pos = abs(cmd)
        m_eff = self.effective_mass
        
        if pos == 1:
            target_a = 0.06
        elif pos == 2:
            target_a = 0.22
        elif pos == 3:
            target_a = 0.45
        elif pos <= 7:
            target_a = 0.45 + (0.80 - 0.45) * ((pos - 3) / 4.0)
        elif pos == 8:
            # Ступень 8 — ретардерное удержание скорости на уклонах и крейсере
            # На спуске (slope < -3.0‰) или при скорости выше 20 км/ч (5.5 м/с) работает как ретардер
            if slope_permille < -3.0 or v > (20.0 / 3.6):
                target_a = 0.30
            else:
                target_a = 0.85
        elif pos <= 12:
            fraction = (pos - 8.0) / 4.0
            target_a = 0.85 + (self.params.a_service_brake_max - 0.85) * fraction
        else:
            fraction = min(1.0, (pos - 12.0) / 3.0)
            target_a = self.params.a_service_brake_max + (self.params.a_emerg_brake_max - self.params.a_service_brake_max) * fraction

        # Режим ретардера на спусках: на уклоне вниз (slope < -3‰) ступени 1..8 не гасят скорость ниже безопасной крейсерской (25 км/ч)
        if slope_permille < -3.0 and pos <= 8:
            v_descent_target = 25.0 / 3.6
            if v < v_descent_target:
                w_grav = self.params.mass_kg * self.params.g * (abs(slope_permille) / 1000.0)
                a_grav = w_grav / m_eff
                # Ограничиваем тормозное усилие величиной гравитационного баланса уклона
                target_a = min(target_a, a_grav * max(0.0, v / v_descent_target))

        b_total = m_eff * target_a
        return float(b_total)

    def calc_resistance_force(self, v: float, slope_permille: float = 0.0, curvature: float = 0.0) -> float:
        """
        Суммарная сила сопротивления W_k(v, i, kappa) в Ньютонах с учетом паспортных габаритов вагона:
        1. Механическое трение качения колес и буксовых узлов (линейная часть ПТР).
        2. Аэродинамическое лобовое сопротивление по миделеву сечению вагона (Ширина 2.5м х Высота 3.3м).
        3. Продольный уклон пути i(s) (‰).
        4. Сопротивление в кривых с учетом жесткой базы тележки (1.8м) и ширины колеи (1.524м).
        """
        m = self.params.mass_kg
        g = self.params.g
        
        # 1. Механическое трение качения колес и подшипников
        v_kmh = max(0.0, v * 3.6)
        w_mech = 2.2 + 0.012 * v_kmh  # кгс/т
        w_mech_n = m * g * w_mech * 1e-3  # в Ньютонах
        
        # 2. Физическое аэродинамическое сопротивление по поперечному миделю
        # S_mid = Ширина * Высота * k_fill = 2.5 * 3.3 * 0.85 = 7.01 м²
        s_mid = self.params.width_m * self.params.height_m * 0.85
        f_aero_n = 0.5 * self.params.c_x * self.params.air_density * s_mid * (v ** 2)
        
        # 3. Сопротивление от продольного уклона W_i (Н)
        # При подъеме (i > 0) тормозит (+), при спуске (i < 0) тянет вперед (-)
        w_i_n = m * g * (slope_permille / 1000.0)
        
        # 4. Сопротивление в кривых по формуле ПТР через жесткую базу тележки b=1.8м и колею s=1.524м:
        # w_r = g * (b + s_gauge) / (2 * R) * mu_flange (кгс/т)
        abs_curv = abs(curvature)
        if abs_curv > 1e-5:
            radius_m = min(10000.0, 1.0 / abs_curv)
            geom_factor = (self.params.bogie_wheelbase_m + self.params.track_gauge_m) / (2.0 * radius_m)
            w_r_kgf = geom_factor * self.params.mu_flange * 1000.0  # кгс/т
            w_r_n = m * g * w_r_kgf * 1e-3
        else:
            w_r_n = 0.0
        
        return float(w_mech_n + f_aero_n + w_i_n + w_r_n)


    def calc_net_acceleration(
        self,
        cmd: int,
        v: float,
        slope_permille: float = 0.0,
        curvature: float = 0.0
    ) -> Tuple[float, float, float]:
        """
        Расчет мгновенного ускорения dv/dt (м/с²) по текущим условиям:
        Возвращает:
          (a_net, f_prop, w_res)
          где a_net — результирующее ускорение вагона,
              f_prop — сила тяги (>0) или торможения (<0),
              w_res  — сила сопротивления.
        """
        w_res = self.calc_resistance_force(v, slope_permille, curvature)
        
        if cmd > 0:
            f_prop = self.calc_traction_force(cmd, v)
        elif cmd < 0:
            f_prop = -self.calc_braking_force(cmd, v, slope_permille)
        else:
            f_prop = 0.0  # выбег (холостой ход)

        # 2-й закон Ньютона с вращающимися массами: a = (F - W) / m_eff
        a_net = (f_prop - w_res) / self.effective_mass
        return a_net, f_prop, w_res

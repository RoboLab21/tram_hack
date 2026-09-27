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

    def calc_braking_force(self, cmd: int, v: float) -> float:
        """
        Тормозная сила B_t(cmd, v) в Ньютонах при cmd in [-15..-1].
        Реализует трехкаскадную систему торможения:
        - 1..6: ЕДТ (рекуперация) с автоматическим переходом на механику на малой скорости;
        - 7..12: ЕДТ + механические дисковые тормоза тележек;
        - 13..15: ЕДТ + дисковые тормоза + магниторельсовый тормоз (башмаки).
        """
        if cmd >= 0:
            return 0.0

        pos = abs(cmd)
        m_eff = self.effective_mass
        
        if pos <= 6:
            # Ступени 1..6: Служебное электродинамическое торможение (замедление 0 .. 0.8 м/с²)
            target_a = 0.80 * (pos / 6.0)
            b_total = m_eff * target_a
            
        elif pos <= 12:
            # Ступени 7..12: Глубокое служебное торможение ЕДТ + механика (0.8 .. 1.5 м/с²)
            fraction = (pos - 6.0) / 6.0
            target_a = 0.80 + (self.params.a_service_brake_max - 0.80) * fraction
            b_total = m_eff * target_a
            
        else:
            # Ступени 13..15: Экстренное торможение с подключением башмаков МРТ (1.5 .. 2.7 м/с²)
            fraction = min(1.0, (pos - 12.0) / 3.0)
            target_a = self.params.a_service_brake_max + (self.params.a_emerg_brake_max - self.params.a_service_brake_max) * fraction
            b_total = m_eff * target_a

        return float(b_total)

    def calc_resistance_force(self, v: float, slope_permille: float = 0.0, curvature: float = 0.0) -> float:
        """
        Суммарная сила сопротивления W_k(v, i, kappa) в Ньютонах.
        :param v: скорость движения (м/с)
        :param slope_permille: продольный уклон пути в тысячных (‰ / промилле), i = dz/ds * 1000
        :param curvature: кривизна пути |kappa| = 1/R (1/м)
        """
        m = self.params.mass_kg
        g = self.params.g
        
        # 1. Основное удельное сопротивление w_0 по формуле ПТР для трамваев (кгс/т)
        v_kmh = max(0.0, v * 3.6)
        w_0 = 2.5 + 0.015 * v_kmh + 0.00035 * (v_kmh ** 2)
        w_0_n = m * g * w_0 * 1e-3  # перевод в Ньютоны
        
        # 2. Сопротивление от уклона W_i (Н)
        # При подъеме (i > 0) уклон сопротивляется движению (+), при спуске (i < 0) помогает (-)
        w_i_n = m * g * (slope_permille / 1000.0)
        
        # 3. Сопротивление в кривых W_r (Н)
        # w_r = 500 / R = 500 * |kappa| (кгс/т)
        w_r = 500.0 * abs(curvature)
        w_r_n = m * g * w_r * 1e-3
        
        return float(w_0_n + w_i_n + w_r_n)

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
            f_prop = -self.calc_braking_force(cmd, v)
        else:
            f_prop = 0.0  # выбег (холостой ход)

        # 2-й закон Ньютона с вращающимися массами: a = (F - W) / m_eff
        a_net = (f_prop - w_res) / self.effective_mass
        return a_net, f_prop, w_res

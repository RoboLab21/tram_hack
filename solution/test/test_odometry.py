import pytest
import numpy as np
from solution.odometry_node_path_calibrated import PathCalibratedDeadReckoningNode, load_path_geometry


def test_estimator_step():
    estimator = PathCalibratedDeadReckoningNode(
        input_in_kmh=True,
        enable_brake_filter=True,
        enable_curve_compensation=False,
        record_history=False
    )
    # Шаг с нулевой скоростью
    estimator.update_driver_cmd(100.0, 0)
    estimator.update_front(100.0, 0.0)
    s0 = estimator.update_rear(100.0, 0.0)
    assert s0.v_est == 0.0
    assert s0.distance == 0.0

    # Шаг со скоростью 36 км/ч = 10 м/с
    estimator.update_driver_cmd(101.0, 5)
    estimator.update_front(101.0, 36.0)
    s1 = estimator.update_rear(101.0, 36.0)
    assert np.isclose(s1.v_est, 10.0, atol=0.1)
    assert s1.distance > 0.0


def test_sensor_fault_rejection():
    estimator = PathCalibratedDeadReckoningNode(
        input_in_kmh=False,
        record_history=False
    )
    estimator.update_driver_cmd(1.0, 0)
    # Нормальная скорость 5 м/с
    estimator.update_front(1.0, 5.0)
    s = estimator.update_rear(1.0, 5.0)
    assert s.v_est == 5.0

    # Передняя тележка выдала 0.0 (аппаратный сбой)
    estimator.update_front(2.0, 0.0)
    s2 = estimator.update_rear(2.0, 5.0)
    assert s2.fault_type == "ZERO_V1"
    assert s2.v_est == 5.0


def test_brake_notch_adaptation():
    estimator = PathCalibratedDeadReckoningNode(
        input_in_kmh=True,
        enable_brake_filter=True,
        record_history=False,
    )

    # 1. При слабом торможении (-1) скорость 0.6 км/ч не глушится (порог ~0.4 км/ч)
    s1 = estimator.step(10.0, 0.6, 0.6, driver_cmd=-1)
    assert s1.v_est > 0.0

    # 2. При стояночном/экстренном торможении (-15) скорость 0.6 км/ч отсекается в 0.0 (ZUPT)
    s2 = estimator.step(10.1, 0.6, 0.6, driver_cmd=-15)
    assert s2.v_est == 0.0

    # 3. Защита от юза при резком срыве колес (просадка скорости со 20 км/ч до 0 за 0.1 с при cmd=-2)
    s3 = estimator.step(20.0, 20.0, 20.0, driver_cmd=-2)
    assert np.isclose(s3.v_est, 20.0 / 3.6, atol=0.01)

    # Резкий срыв обоих колес в 0 за dt = 0.1 с
    s_slide = estimator.step(20.1, 0.0, 0.0, driver_cmd=-2)
    # Скорость не должна мгновенно упасть в 0, а должна быть ограничена физическим замедлением ступени
    assert s_slide.v_est > 5.0  # 20 км/ч = 5.55 м/с, падение ограничено ~1.26 м/с² * 0.1 с ~ 0.13 м/с



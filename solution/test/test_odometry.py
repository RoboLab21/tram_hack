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

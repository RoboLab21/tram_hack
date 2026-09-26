"""
Пакет решений резервной одометрии беспилотного трамвая.

Содержит:
- DeadReckoningNode: Базовая нода резервной одометрии (v1)
- FilteredDeadReckoningNode: Нода с фильтрацией сбоев по квадратичному отклонению (v2)
- PathCalibratedDeadReckoningNode: Нода с автокалибровкой износа колес на первых трех прямых участках пути (v3)
- TramOdometryROS2Node: Нода ROS 2 Humble (v1)
- TramOdometryFilteredROS2Node: Нода ROS 2 Humble с фильтром (v2)
"""

from .odometry_node import DeadReckoningNode, OdometryState
from .odometry_node_filtered import FilteredDeadReckoningNode, FilteredOdometryState
from .odometry_node_path_calibrated import (
    PathCalibratedDeadReckoningNode,
    CalibratedOdometryState,
    StraightSection,
    extract_straight_sections
)

__all__ = [
    "DeadReckoningNode",
    "OdometryState",
    "FilteredDeadReckoningNode",
    "FilteredOdometryState",
    "PathCalibratedDeadReckoningNode",
    "CalibratedOdometryState",
    "StraightSection",
    "extract_straight_sections",
]

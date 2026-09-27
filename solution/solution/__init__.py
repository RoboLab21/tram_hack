"""
Пакет решений резервной одометрии беспилотного трамвая.

Содержит:
- PathCalibratedDeadReckoningNode: Нода с автокалибровкой износа колес по цифровой карте пути
"""

from .odometry_node_path_calibrated import (
    PathCalibratedDeadReckoningNode,
    CalibratedOdometryState,
    StraightSection,
    extract_straight_sections
)

__all__ = [
    "PathCalibratedDeadReckoningNode",
    "CalibratedOdometryState",
    "StraightSection",
    "extract_straight_sections",
]

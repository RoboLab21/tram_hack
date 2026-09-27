"""
Пакет solution: алгоритмы и ROS 2 ноды резервной одометрии беспилотного трамвая.
"""

from .odometry_node_path_calibrated import (
    PathCalibratedDeadReckoningNode,
    CalibratedOdometryState,
    StraightSection,
    load_path_geometry,
    extract_straight_sections,
)

__all__ = [
    "PathCalibratedDeadReckoningNode",
    "CalibratedOdometryState",
    "StraightSection",
    "load_path_geometry",
    "extract_straight_sections",
]

try:
    from .velocity_node import VelocityNode
    from .position_node import PositionNode
    from .recovery_odometry_node import RecoveryOdometryNode
    __all__.extend(["VelocityNode", "PositionNode", "RecoveryOdometryNode"])
except ImportError:
    pass

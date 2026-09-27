import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory("solution")
    default_params_file = os.path.join(pkg_dir, "config", "params.yaml")

    declare_params_file = DeclareLaunchArgument(
        "params_file",
        default_value=default_params_file,
        description="Путь к YAML-файлу параметров нод"
    )

    velocity_node = Node(
        package="solution",
        executable="velocity_node",
        name="velocity_node",
        output="screen",
        parameters=[LaunchConfiguration("params_file")]
    )

    position_node = Node(
        package="solution",
        executable="position_node",
        name="position_node",
        output="screen",
        parameters=[LaunchConfiguration("params_file")]
    )

    return LaunchDescription([
        declare_params_file,
        velocity_node,
        position_node,
    ])

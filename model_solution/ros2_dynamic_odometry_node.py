#!/usr/bin/env python3
"""
ROS 2 Humble нода резервной одометрии по динамической физической модели.
Работает исключительно по команде контроллера водителя (driver_cmd) и цифровой карте пути (pathgrath),
полностью игнорируя датчики скорости колес (тахометры).

Subscribe:
    /vehicle/driver_position_cmd  (tram_vehicle_msgs/msg/DriverControllerCommand)

Publish:
    /result/velocity (tram_vehicle_msgs/msg/VelocitySensor)
    /result/position (nav_msgs/msg/Odometry)
"""

import sys
from pathlib import Path

# Добавляем пути в sys.path
_current_dir = Path(__file__).resolve().parent
_repo_dir = _current_dir.parent
for p in [str(_repo_dir), str(_current_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy

from nav_msgs.msg import Odometry
from tram_vehicle_msgs.msg import VelocitySensor, DriverControllerCommand

from dynamic_odometry_node import DynamicOdometryNode
from tram_dynamic_model import TramParameters, SpeedRegimeParameters


class Ros2DynamicOdometryNode(Node):
    def __init__(self):
        super().__init__("dynamic_model_odometry_node")

        # 1. Параметры ноды
        self.declare_parameter("mass_kg", 24500.0)  # Масса вагона по умолчанию 24.5 т
        self.declare_parameter("path_file", "")
        self.declare_parameter("frame_id", "odom")
        self.declare_parameter("child_frame_id", "base_link")
        self.declare_parameter("max_city_speed_kmh", 60.0)
        self.declare_parameter("loop_speed_kmh", 15.0)
        
        mass_kg = self.get_parameter("mass_kg").get_parameter_value().double_value
        path_file = self.get_parameter("path_file").get_parameter_value().string_value
        self.frame_id = self.get_parameter("frame_id").get_parameter_value().string_value
        self.child_frame_id = self.get_parameter("child_frame_id").get_parameter_value().string_value
        max_city_spd = self.get_parameter("max_city_speed_kmh").get_parameter_value().double_value
        loop_spd = self.get_parameter("loop_speed_kmh").get_parameter_value().double_value

        # 2. Инициализация физико-математической модели со скоростными режимами
        params = TramParameters(mass_kg=mass_kg)
        speed_params = SpeedRegimeParameters(
            enable_speed_regimes=True,
            max_city_speed_kmh=max_city_spd,
            loop_speed_kmh=loop_spd
        )
        path_arg = Path(path_file) if path_file and Path(path_file).exists() else None
        
        self.estimator = DynamicOdometryNode(
            path_geometry_file_or_data=path_arg,
            tram_params=params,
            speed_params=speed_params
        )

        # 3. Публикаторы результатов
        qos_pub = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.pub_vel = self.create_publisher(VelocitySensor, "/result/velocity", qos_pub)
        self.pub_pos = self.create_publisher(Odometry, "/result/position", qos_pub)

        # 4. Подписка на топик контроллера машиниста
        self.sub_cmd = self.create_subscription(
            DriverControllerCommand,
            "/vehicle/driver_position_cmd",
            self.cb_driver_cmd,
            qos_profile_sensor_data
        )

        self.msg_count = 0
        self.get_logger().info(
            f"Dynamic Model Odometry Node запущена. Расчетная масса: {mass_kg / 1000.0:.1f} т, "
            f"Карта пути: {'загружена' if self.estimator.has_map else 'не задана'}"
        )

    def cb_driver_cmd(self, msg: DriverControllerCommand):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if t <= 0:
            return

        state = self.estimator.update_driver_cmd(t, msg.position)
        self.publish_results(msg.header.stamp, state)

    def publish_results(self, stamp, state):
        # 1. Публикация /result/velocity
        vel_msg = VelocitySensor()
        vel_msg.header.stamp = stamp
        vel_msg.header.frame_id = self.child_frame_id
        vel_msg.velocity = float(state.v_est)
        self.pub_vel.publish(vel_msg)

        # 2. Публикация /result/position
        odom_msg = Odometry()
        odom_msg.header.stamp = stamp
        odom_msg.header.frame_id = self.frame_id
        odom_msg.child_frame_id = self.child_frame_id

        # Положение (3D по карте, либо продольное x = distance при отсутствии карты)
        if self.estimator.has_map:
            odom_msg.pose.pose.position.x = float(state.x)
            odom_msg.pose.pose.position.y = float(state.y)
            odom_msg.pose.pose.position.z = float(state.z)
        else:
            odom_msg.pose.pose.position.x = float(state.distance)
            odom_msg.pose.pose.position.y = 0.0
            odom_msg.pose.pose.position.z = 0.0

        # Оцененная продольная скорость в Twist
        odom_msg.twist.twist.linear.x = float(state.v_est)
        self.pub_pos.publish(odom_msg)

        self.msg_count += 1
        if self.msg_count % 100 == 0:
            self.get_logger().info(
                f"Матмодель: t={state.timestamp:.1f}s | cmd={state.driver_cmd:+2d} | "
                f"V_расч={state.v_kmh:.1f} км/ч ({state.v_est:.2f} м/с) | S={state.distance:.1f} м | FSM={state.fsm_state}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = Ros2DynamicOdometryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

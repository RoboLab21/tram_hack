#!/usr/bin/env python3
"""
ROS 2 Нода оценки скорости беспилотного трамвая (VelocityNode).

Входные топики (Subscribe):
    /vehicle/front_bogie_velocity (tram_vehicle_msgs/msg/VelocitySensor) - скорость передней тележки (км/ч)
    /vehicle/rear_bogie_velocity  (tram_vehicle_msgs/msg/VelocitySensor) - скорость задней тележки (км/ч)
    /vehicle/driver_position_cmd  (tram_vehicle_msgs/msg/DriverControllerCommand) - положение контроллера (-15..+15)

Выходные топики (Publish):
    /result/velocity (tram_vehicle_msgs/msg/VelocitySensor) - оцененная продольная скорость (м/с)
"""

import sys
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy

from tram_vehicle_msgs.msg import VelocitySensor, DriverControllerCommand

try:
    from .odometry_node_path_calibrated import PathCalibratedDeadReckoningNode
except ImportError:
    from odometry_node_path_calibrated import PathCalibratedDeadReckoningNode


class VelocityNode(Node):
    def __init__(self):
        super().__init__("velocity_node")

        # ---------------------------------------------------------------------
        # Параметры ноды
        # ---------------------------------------------------------------------
        self.declare_parameter("input_in_kmh", True)
        self.declare_parameter("integration_method", "trapezoidal")
        self.declare_parameter("velocity_filter", "mean")
        self.declare_parameter("enable_brake_filter", True)
        self.declare_parameter("enable_curve_compensation", True)
        self.declare_parameter("child_frame_id", "base_link")

        input_in_kmh = self.get_parameter("input_in_kmh").get_parameter_value().bool_value
        integration_method = self.get_parameter("integration_method").get_parameter_value().string_value
        velocity_filter = self.get_parameter("velocity_filter").get_parameter_value().string_value
        enable_brake_filter = self.get_parameter("enable_brake_filter").get_parameter_value().bool_value
        enable_curve_comp = self.get_parameter("enable_curve_compensation").get_parameter_value().bool_value
        self.child_frame_id = self.get_parameter("child_frame_id").get_parameter_value().string_value

        # ---------------------------------------------------------------------
        # Алгоритмическое ядро оценки скорости и фильтрации сбоев
        # ---------------------------------------------------------------------
        self.estimator = PathCalibratedDeadReckoningNode(
            input_in_kmh=input_in_kmh,
            integration_method=integration_method,
            velocity_filter=velocity_filter,
            enable_brake_filter=enable_brake_filter,
            enable_curve_compensation=enable_curve_comp,
            record_history=False
        )

        # ---------------------------------------------------------------------
        # Издатель /result/velocity
        # ---------------------------------------------------------------------
        qos_pub = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.pub_vel = self.create_publisher(VelocitySensor, "/result/velocity", qos_pub)

        # ---------------------------------------------------------------------
        # Подписчики входной телеметрии
        # ---------------------------------------------------------------------
        self.sub_front = self.create_subscription(
            VelocitySensor,
            "/vehicle/front_bogie_velocity",
            self.cb_front_velocity,
            qos_profile_sensor_data
        )
        self.sub_rear = self.create_subscription(
            VelocitySensor,
            "/vehicle/rear_bogie_velocity",
            self.cb_rear_velocity,
            qos_profile_sensor_data
        )
        self.sub_cmd = self.create_subscription(
            DriverControllerCommand,
            "/vehicle/driver_position_cmd",
            self.cb_driver_cmd,
            qos_profile_sensor_data
        )

        self.msg_count = 0
        self.get_logger().info("VelocityNode успешно инициализирована.")

    def cb_driver_cmd(self, msg: DriverControllerCommand):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.estimator.update_driver_cmd(t, msg.position)

    def cb_front_velocity(self, msg: VelocitySensor):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        state = self.estimator.update_front(t, msg.velocity)
        if state:
            self._publish_velocity(msg.header.stamp, state.v_est)

    def cb_rear_velocity(self, msg: VelocitySensor):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        state = self.estimator.update_rear(t, msg.velocity)
        if state:
            self._publish_velocity(msg.header.stamp, state.v_est)

    def _publish_velocity(self, stamp, v_est: float):
        vel_msg = VelocitySensor()
        vel_msg.header.stamp = stamp
        vel_msg.header.frame_id = self.child_frame_id
        vel_msg.velocity = float(v_est)
        self.pub_vel.publish(vel_msg)

        self.msg_count += 1
        if self.msg_count % 500 == 0:
            self.get_logger().info(f"V_est={v_est * 3.6:.1f} км/ч ({v_est:.2f} м/с)")


def main(args=None):
    rclpy.init(args=args)
    node = VelocityNode()
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

#!/usr/bin/env python3
"""
ROS 2 нода резервной одометрии (v2) с квадратичной фильтрацией ошибок датчиков.

Использует FilteredDeadReckoningNode:
- Детектирует выпадения в ноль и аномальные скачки через (v1 - v2)^2 и (dv/dt)^2
- Публикует /result/velocity и /result/position согласно регламенту хакатона.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy

from nav_msgs.msg import Odometry
from tram_vehicle_msgs.msg import VelocitySensor, DriverControllerCommand

from odometry_node_filtered import FilteredDeadReckoningNode


class TramOdometryFilteredNode(Node):
    def __init__(self):
        super().__init__("tram_odometry_filtered_node")

        self.declare_parameter("input_in_kmh", True)
        self.declare_parameter("integration_method", "trapezoidal")
        self.declare_parameter("diff_thresh_kmh", 3.6)
        self.declare_parameter("a_max_ms2", 2.5)
        self.declare_parameter("frame_id", "odom")
        self.declare_parameter("child_frame_id", "base_link")

        input_in_kmh = self.get_parameter("input_in_kmh").get_parameter_value().bool_value
        method = self.get_parameter("integration_method").get_parameter_value().string_value
        diff_th = self.get_parameter("diff_thresh_kmh").get_parameter_value().double_value
        a_max = self.get_parameter("a_max_ms2").get_parameter_value().double_value
        self.frame_id = self.get_parameter("frame_id").get_parameter_value().string_value
        self.child_frame_id = self.get_parameter("child_frame_id").get_parameter_value().string_value

        # Нода с квадратичной фильтрацией
        self.estimator = FilteredDeadReckoningNode(
            input_in_kmh=input_in_kmh,
            integration_method=method,
            diff_thresh_kmh=diff_th,
            a_max_ms2=a_max
        )

        qos_pub = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.pub_vel = self.create_publisher(VelocitySensor, "/result/velocity", qos_pub)
        self.pub_pos = self.create_publisher(Odometry, "/result/position", qos_pub)

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
        self.anomaly_count = 0
        self.get_logger().info("TramOdometryFilteredNode (v2) инициализирована.")

    def cb_front_velocity(self, msg: VelocitySensor):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        state = self.estimator.update_front(t, msg.velocity)
        if state:
            self.publish_results(msg.header.stamp, state)

    def cb_rear_velocity(self, msg: VelocitySensor):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        state = self.estimator.update_rear(t, msg.velocity)
        if state:
            self.publish_results(msg.header.stamp, state)

    def cb_driver_cmd(self, msg: DriverControllerCommand):
        pass

    def publish_results(self, stamp, state):
        if state.fault_type != "OK":
            self.anomaly_count += 1
            if self.anomaly_count % 20 == 1:
                self.get_logger().warn(
                    f"Аномалия датчиков [{state.fault_type}]: v1={state.v1_raw*3.6:.1f} км/ч, "
                    f"v2={state.v2_raw*3.6:.1f} км/ч, фильтр={state.v_est*3.6:.1f} км/ч"
                )

        # 1. /result/velocity
        vel_msg = VelocitySensor()
        vel_msg.header.stamp = stamp
        vel_msg.header.frame_id = self.child_frame_id
        vel_msg.velocity = float(state.v_est)
        self.pub_vel.publish(vel_msg)

        # 2. /result/position
        odom_msg = Odometry()
        odom_msg.header.stamp = stamp
        odom_msg.header.frame_id = self.frame_id
        odom_msg.child_frame_id = self.child_frame_id
        odom_msg.pose.pose.position.x = float(state.distance)
        odom_msg.pose.pose.position.y = 0.0
        odom_msg.pose.pose.position.z = 0.0
        odom_msg.twist.twist.linear.x = float(state.v_est)
        self.pub_pos.publish(odom_msg)

        self.msg_count += 1
        if self.msg_count % 100 == 0:
            self.get_logger().info(
                f"Фильтрованная одометрия: t={state.timestamp:.1f}s | "
                f"V={state.v_est*3.6:.1f} км/ч | R={state.distance:.2f} м | Состояние={state.fault_type}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = TramOdometryFilteredNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

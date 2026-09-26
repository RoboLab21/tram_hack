#!/usr/bin/env python3
"""
Subscribe:
    /vehicle/front_bogie_velocity (tram_vehicle_msgs/msg/VelocitySensor)
    /vehicle/rear_bogie_velocity  (tram_vehicle_msgs/msg/VelocitySensor)
    /vehicle/driver_position_cmd  (tram_vehicle_msgs/msg/DriverControllerCommand)

Publish:
    /result/velocity (tram_vehicle_msgs/msg/VelocitySensor)
    /result/position (nav_msgs/msg/Odometry)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy

from nav_msgs.msg import Odometry
from tram_vehicle_msgs.msg import VelocitySensor, DriverControllerCommand

# нужно схлопнуть с этой штукой
from odometry_node import DeadReckoningNode


class RecoveryOdometryNode(Node):
    def __init__(self):
        super().__init__("recovery_odometry_node")

        # непонятное
        
        self.declare_parameter("input_in_kmh", True)
        self.declare_parameter("integration_method", "trapezoidal")
        self.declare_parameter("frame_id", "odom")
        self.declare_parameter("child_frame_id", "base_link")
        
        input_in_kmh = self.get_parameter("input_in_kmh").get_parameter_value().bool_value
        method = self.get_parameter("integration_method").get_parameter_value().string_value
        self.frame_id = self.get_parameter("frame_id").get_parameter_value().string_value
        self.child_frame_id = self.get_parameter("child_frame_id").get_parameter_value().string_value
        
        # Ядро одометрии
        self.estimator = DeadReckoningNode(input_in_kmh=input_in_kmh, integration_method=method)

        # перепроверить что нужно публиковать
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
        self.get_logger().info("Recovery Odometry Node started")

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

    # нет прямого использования :(
    def cb_driver_cmd(self, msg: DriverControllerCommand):
        # Позиция ручки водителя может использоваться для дополнительной фильтрации
        pass

    # перепроверить
    def publish_results(self, stamp, state):
        # 1. Публикация /result/velocity
        vel_msg = VelocitySensor()
        vel_msg.header.stamp = stamp
        vel_msg.header.frame_id = self.child_frame_id
        vel_msg.velocity = float(state.v_avg)
        self.pub_vel.publish(vel_msg)

        # 2. Публикация /result/position
        odom_msg = Odometry()
        odom_msg.header.stamp = stamp
        odom_msg.header.frame_id = self.frame_id
        odom_msg.child_frame_id = self.child_frame_id
        
        # Продольное положение трамвая вдоль пути: x = R
        odom_msg.pose.pose.position.x = float(state.distance)
        odom_msg.pose.pose.position.y = 0.0
        odom_msg.pose.pose.position.z = 0.0
        
        # Скорость в Twist
        odom_msg.twist.twist.linear.x = float(state.v_avg)
        
        self.pub_pos.publish(odom_msg)
        
        self.msg_count += 1
        if self.msg_count % 100 == 0:
            self.get_logger().info(
                f"Одометрия онлайн: t={state.timestamp:.1f}s | "
                f"V_ср={state.v_avg * 3.6:.1f} км/ч ({state.v_avg:.2f} м/с) | R={state.distance:.2f} м"
            )

    # добавить обработку через модель

def main(args=None):
    rclpy.init(args=args)
    node = RecoveryOdometryNode()
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

#!/usr/bin/env python3
"""Relay solution topics with original timestamps intact.

  /localization/kinematic_state  ->  /result/position  (nav_msgs/msg/Odometry)
  /vehicle/front_bogie_velocity  ->  /result/velocity  (tram_vehicle_msgs/msg/VelocitySensor)

Messages are republished as-is, immediately on receipt, keeping header.stamp.
"""

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from tram_vehicle_msgs.msg import VelocitySensor


class Relay(Node):
    def __init__(self):
        super().__init__("result_relay")
        # Subscribe best-effort: compatible with both reliable and best-effort publishers.
        self.create_subscription(
            Odometry, "/localization/kinematic_state", self.cb_position, qos_profile_sensor_data
        )
        self.create_subscription(
            VelocitySensor,
            "/vehicle/front_bogie_velocity",
            self.cb_velocity,
            qos_profile_sensor_data,
        )
        self.pub_position = self.create_publisher(Odometry, "/result/position", 10)
        self.pub_velocity = self.create_publisher(VelocitySensor, "/result/velocity", 10)

    def cb_position(self, msg: Odometry):
        self.pub_position.publish(msg)
        msg_velocity = VelocitySensor()
        msg_velocity.header = msg.header
        msg_velocity.velocity = msg.twist.twist.linear.x
        self.pub_velocity.publish(msg_velocity)

    def cb_velocity(self, msg: VelocitySensor):
        # msg.velocity = msg.velocity / 3.6
        # self.pub_velocity.publish(msg)
        pass


def main():
    rclpy.init()
    node = Relay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

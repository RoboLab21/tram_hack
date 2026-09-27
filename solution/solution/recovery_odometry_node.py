"""
Основная ROS 2 нода резервной одометрии беспилотного трамвая (RecoveryOdometryNode).

Входные топики (Subscribe):
    /vehicle/front_bogie_velocity (tram_vehicle_msgs/msg/VelocitySensor) - скорость передней тележки (км/ч)
    /vehicle/rear_bogie_velocity  (tram_vehicle_msgs/msg/VelocitySensor) - скорость задней тележки (км/ч)
    /vehicle/driver_position_cmd  (tram_vehicle_msgs/msg/DriverControllerCommand) - положение контроллера (-15..+15)

Выходные топики (Publish):
    /result/velocity (tram_vehicle_msgs/msg/VelocitySensor) - оцененная продольная скорость (м/с)
    /result/position (nav_msgs/msg/Odometry)                - оцененное положение (м) вдоль пути / 3D

Алгоритм:
1. Квадратичный фильтр аппаратных сбоев колесных датчиков (нули, выбросы, расхождение dV^2).
2. Автокалибровка коэффициента износа колес k_scale на ВСЕМ ПУТИ МАРШРУТА (на каждом контрольном прямом участке).
3. Фильтр активного торможения (строго при driver_cmd < 0): отсечка ползучего хода остановки (v < 0.8 км/ч)
   и ограничение замедления при срыве в юз (a < -2.5 м/с^2).
4. Геометрическая компенсация кривизны коротких / крутых поворотов (R <= 125 м) по цифровой карте pathgrath.
"""

import sys
import json
from pathlib import Path
from typing import list

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy

from nav_msgs.msg import Odometry
from tram_vehicle_msgs.msg import VelocitySensor, DriverControllerCommand

# Подключение модулей решения
_dir = Path(__file__).resolve().parent
if str(_dir) not in sys.path:
    sys.path.insert(0, str(_dir))
if str(_dir.parent) not in sys.path:
    sys.path.insert(0, str(_dir.parent))

try:
    from solution.odometry_node_path_calibrated import (
        PathCalibratedDeadReckoningNode,
        load_path_geometry,
        StraightSection,
    )
except ImportError:
    from odometry_node_path_calibrated import (
        PathCalibratedDeadReckoningNode,
        load_path_geometry,
        StraightSection,
    )

import numpy as np


class RecoveryOdometryNode(Node):
    def __init__(self):
        super().__init__("recovery_odometry_node")

        # ---------------------------------------------------------------------
        # Параметры ноды
        # ---------------------------------------------------------------------
        self.declare_parameter("route", "щук-талл")  # "щук-талл" или "талл-щук"
        self.declare_parameter("path_file", "")  # Явный путь к файлу pathgrath JSON
        self.declare_parameter("input_in_kmh", True)  # Входные датчики тележек в км/ч
        self.declare_parameter(
            "integration_method", "trapezoidal"
        )  # "trapezoidal" или "rectangular"
        self.declare_parameter("velocity_filter", "mean")  # "mean", "ema", "sma"
        self.declare_parameter("frame_id", "odom")
        self.declare_parameter("child_frame_id", "base_link")
        self.declare_parameter(
            "publish_3d_pose", False
        )  # True = x,y,z по карте; False = x=R вдоль пути
        self.declare_parameter("enable_brake_filter", True)
        self.declare_parameter("enable_curve_compensation", True)

        self.route = self.get_parameter("route").get_parameter_value().string_value
        path_file_param = (
            self.get_parameter("path_file").get_parameter_value().string_value
        )
        input_in_kmh = (
            self.get_parameter("input_in_kmh").get_parameter_value().bool_value
        )
        integration_method = (
            self.get_parameter("integration_method").get_parameter_value().string_value
        )
        velocity_filter = (
            self.get_parameter("velocity_filter").get_parameter_value().string_value
        )
        self.frame_id = (
            self.get_parameter("frame_id").get_parameter_value().string_value
        )
        self.child_frame_id = (
            self.get_parameter("child_frame_id").get_parameter_value().string_value
        )
        self.publish_3d_pose = (
            self.get_parameter("publish_3d_pose").get_parameter_value().bool_value
        )
        enable_brake_filter = (
            self.get_parameter("enable_brake_filter").get_parameter_value().bool_value
        )
        enable_curve_comp = (
            self.get_parameter("enable_curve_compensation")
            .get_parameter_value()
            .bool_value
        )

        # ---------------------------------------------------------------------
        # Поиск и загрузка цифровой карты пути (pathgrath)
        # ---------------------------------------------------------------------
        path_file_path = self._locate_path_file(path_file_param, self.route)
        self.straight_sections: list[StraightSection] = []
        self.s_map: np.ndarray | None = None
        self.curv_map: np.ndarray | None = None
        self.pts_x: np.ndarray | None = None
        self.pts_y: np.ndarray | None = None
        self.pts_z: np.ndarray | None = None

        if path_file_path and path_file_path.exists():
            self.get_logger().info(
                f"Загрузка цифровой карты пути: {path_file_path.name}"
            )
            # Вариант 3: загружаем ВСЕ прямые участки на всем протяжении маршрута (top_n=None)
            self.straight_sections, self.s_map, self.curv_map = load_path_geometry(
                path_file_path, top_n=None
            )
            # Извлекаем 3D координаты точек карты для интерполяции положения
            try:
                d = json.loads(path_file_path.read_text(encoding="utf-8"))
                pts = d.get("points", [])
                if pts:
                    self.pts_x = np.array([pt["x"] for pt in pts], dtype=np.float64)
                    self.pts_y = np.array([pt["y"] for pt in pts], dtype=np.float64)
                    self.pts_z = np.array(
                        [pt.get("z", 0.0) for pt in pts], dtype=np.float64
                    )
            except Exception as e:
                self.get_logger().warn(
                    f"Не удалось распарсить 3D координаты точек: {e}"
                )

            self.get_logger().info(
                f"Карта успешно загружена: длина {self.s_map[-1]:.1f}м, "
                f"контрольных прямых участков на всем пути: {len(self.straight_sections)}"
            )
        else:
            self.get_logger().warn(
                "Цифровая карта пути не найдена. Работаем без автокалибровки по карте."
            )

        # ---------------------------------------------------------------------
        # Ядро резервной одометрии (с калибровкой k_scale на всем пути)
        # ---------------------------------------------------------------------
        self.estimator = PathCalibratedDeadReckoningNode(
            straight_sections=self.straight_sections,
            path_geometry=(self.s_map, self.curv_map)
            if self.s_map is not None
            else None,
            input_in_kmh=input_in_kmh,
            integration_method=integration_method,
            velocity_filter=velocity_filter,
            enable_brake_filter=enable_brake_filter,
            enable_curve_compensation=enable_curve_comp,
        )

        # ---------------------------------------------------------------------
        # Издатели (Publishers) согласно контракту хакатона
        # ---------------------------------------------------------------------
        qos_pub = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.pub_vel = self.create_publisher(
            VelocitySensor, "/result/velocity", qos_pub
        )
        self.pub_pos = self.create_publisher(Odometry, "/result/position", qos_pub)

        # ---------------------------------------------------------------------
        # Подписчики (Subscribers)
        # ---------------------------------------------------------------------
        self.sub_front = self.create_subscription(
            VelocitySensor,
            "/vehicle/front_bogie_velocity",
            self.cb_front_velocity,
            qos_profile_sensor_data,
        )
        self.sub_rear = self.create_subscription(
            VelocitySensor,
            "/vehicle/rear_bogie_velocity",
            self.cb_rear_velocity,
            qos_profile_sensor_data,
        )
        self.sub_cmd = self.create_subscription(
            DriverControllerCommand,
            "/vehicle/driver_position_cmd",
            self.cb_driver_cmd,
            qos_profile_sensor_data,
        )

        self.msg_count = 0
        self.last_k_scale = 1.0
        self.get_logger().info(
            "RecoveryOdometryNode (Основное решение, Вариант 3: k_scale на всем пути) инициализирована."
        )

    def _locate_path_file(self, explicit_path: str, route: str) -> Path | None:
        """Определяет путь к JSON файлу цифровой карты."""
        if explicit_path:
            p = Path(explicit_path)
            if p.exists():
                return p

        # Варианты имен файлов для маршрутов
        candidates_map = {
            "щук-талл": ["щукинская - таллинская.json", "shchuk_tall.json"],
            "талл-щук": ["таллинская - щукинская.json", "tall_shchuk.json"],
        }
        cand_files = candidates_map.get(route.lower(), ["щукинская - таллинская.json"])

        search_dirs = [
            Path.cwd() / "pathgrath",
            Path(__file__).resolve().parent.parent / "pathgrath",
            Path(__file__).resolve().parent / "pathgrath",
            Path("/pathgrath"),
        ]

        for s_dir in search_dirs:
            if s_dir.exists():
                for fname in cand_files:
                    target = s_dir / fname
                    if target.exists():
                        return target

        return None

    def cb_driver_cmd(self, msg: DriverControllerCommand):
        """Обработка команд контроллера водителя (/vehicle/driver_position_cmd)."""
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.estimator.update_driver_cmd(t, msg.position)

    def cb_front_velocity(self, msg: VelocitySensor):
        """Обработка скорости передней тележки."""
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        state = self.estimator.update_front(t, msg.velocity)
        if state:
            self.publish_results(msg.header.stamp, state)

    def cb_rear_velocity(self, msg: VelocitySensor):
        """Обработка скорости задней тележки."""
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        state = self.estimator.update_rear(t, msg.velocity)
        if state:
            self.publish_results(msg.header.stamp, state)

    def publish_results(self, stamp, state):
        """Публикация оцененной скорости и положения в выходные топики."""
        # 1. Топик /result/velocity (м/с)
        vel_msg = VelocitySensor()
        vel_msg.header.stamp = stamp
        vel_msg.header.frame_id = self.child_frame_id
        vel_msg.velocity = float(state.v_est)
        self.pub_vel.publish(vel_msg)

        # 2. Топик /result/position (м)
        odom_msg = Odometry()
        odom_msg.header.stamp = stamp
        odom_msg.header.frame_id = self.frame_id
        odom_msg.child_frame_id = self.child_frame_id

        # Формирование координат положения
        if self.publish_3d_pose and self.pts_x is not None and self.s_map is not None:
            # 3D интерполяция положения вдоль карты pathgrath
            s_cur = max(0.0, min(float(state.distance), float(self.s_map[-1])))
            odom_msg.pose.pose.position.x = float(
                np.interp(s_cur, self.s_map, self.pts_x)
            )
            odom_msg.pose.pose.position.y = float(
                np.interp(s_cur, self.s_map, self.pts_y)
            )
            odom_msg.pose.pose.position.z = float(
                np.interp(s_cur, self.s_map, self.pts_z)
            )
        else:
            # Продольная координата по оси пути (x = R, y = 0, z = 0)
            odom_msg.pose.pose.position.x = float(state.distance)
            odom_msg.pose.pose.position.y = 0.0
            odom_msg.pose.pose.position.z = 0.0

        # Скорость в Twist
        odom_msg.twist.twist.linear.x = float(state.v_est)
        self.pub_pos.publish(odom_msg)

        # Логирование калибровки и хода движения
        self.msg_count += 1
        if state.k_scale != self.last_k_scale:
            self.get_logger().info(
                f"[Автокалибровка k_scale]: Обновлен коэффициент износа: k={state.k_scale:.5f} "
                f"(износ {state.wear_pct:+.2f}%, откалибровано участков: {len(self.estimator.scale_samples)})"
            )
            self.last_k_scale = state.k_scale

        if self.msg_count % 200 == 0:
            self.get_logger().info(
                f"t={state.timestamp:.1f}s | V_est={state.v_est * 3.6:4.1f} км/ч ({state.v_est:4.2f} м/с) | "
                f"R={state.distance:6.1f} м | k={state.k_scale:.5f} | Участок={state.straight_id} | "
                f"Тормоз={'ДА' if state.is_braking else 'НЕТ'}"
            )


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

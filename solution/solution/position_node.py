#!/usr/bin/env python3
"""
ROS 2 Нода оценки 3D положения беспилотного трамвая (PositionNode).

Особенности:
1. Автоматический выбор цифровой карты пути (pathgraph) по стартовой GNSS-позиции.
2. Полная отвязка от ручного указания маршрута или файла карты (нулевая конфигурация).
3. Высокоточная 3D проекция (x, y, z, ориентация) вдоль рельсового пути.

Входные топики (Subscribe):
    /result/velocity         (tram_vehicle_msgs/msg/VelocitySensor) - оцененная скорость вагона
    /vehicle/front_bogie_velocity (tram_vehicle_msgs/msg/VelocitySensor) - резервный канал скорости
    /sensing/gnss/master/fix (sensor_msgs/msg/NavSatFix)           - GNSS master антенна
    /sensing/gnss/rover/fix  (sensor_msgs/msg/NavSatFix)           - GNSS rover антенна

Выходной топик (Publish):
    /result/position         (nav_msgs/msg/Odometry)                - 3D координаты (x, y, z) base_link
"""

import os
import sys
import json
import math
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import NavSatFix
    from tram_vehicle_msgs.msg import VelocitySensor
    HAS_RCLPY = True
except ImportError:
    HAS_RCLPY = False
    Node = object
    qos_profile_sensor_data = QoSProfile = ReliabilityPolicy = None
    Odometry = NavSatFix = VelocitySensor = None

try:
    from ament_index_python.packages import get_package_share_directory
    HAS_AMENT = True
except ImportError:
    HAS_AMENT = False

import numpy as np


# Полиномиальные коэффициенты преобразования WGS-84 (lat, lon) в координаты карты Москвы (X, Y)
_PROJ_CX = np.array([99701.77473554, 2616.15267403, 62690.94975708, -93308.16929609, 24136.64597943, -2001.52042217])
_PROJ_CY = np.array([84978.82007164, 111270.23819383, -1441.24027607, 40917.43731783, -10440.75503033, 1053.07056341])


def wgs84_to_map(lat: float, lon: float) -> Tuple[float, float]:
    """Переводит координаты WGS-84 в плоские координаты карты (метры)."""
    dlat = lat - 55.80
    dlon = lon - 37.40
    feat = np.array([1.0, dlat, dlon, dlat ** 2, dlat * dlon, dlon ** 2])
    return float(feat @ _PROJ_CX), float(feat @ _PROJ_CY)


class RouteGeometry:
    """Хранилище геометрии одного направления движения."""
    def __init__(self, name: str, file_path: Path):
        self.name = name
        self.file_path = file_path
        d = json.loads(file_path.read_text(encoding="utf-8"))
        pts = d.get("points", [])
        self.pts_x = np.array([p["x"] for p in pts], dtype=np.float64)
        self.pts_y = np.array([p["y"] for p in pts], dtype=np.float64)
        self.pts_z = np.array([p.get("z", 0.0) for p in pts], dtype=np.float64)
        self.pts_tang = np.array([p.get("tang", 0.0) for p in pts], dtype=np.float64)

        dists = np.sqrt(np.diff(self.pts_x)**2 + np.diff(self.pts_y)**2 + np.diff(self.pts_z)**2)
        self.s_map = np.cumsum(np.insert(dists, 0, 0.0))
        self.total_length = float(self.s_map[-1])


class PositionNode(Node):
    def __init__(self):
        super().__init__("position_node")

        # ---------------------------------------------------------------------
        # Параметры ноды (только необходимые фреймы)
        # ---------------------------------------------------------------------
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("child_frame_id", "base_link")
        self.frame_id = self.get_parameter("frame_id").get_parameter_value().string_value
        self.child_frame_id = self.get_parameter("child_frame_id").get_parameter_value().string_value

        # ---------------------------------------------------------------------
        # Автоматический поиск и загрузка всех доступных карт пути
        # ---------------------------------------------------------------------
        self.routes: Dict[str, RouteGeometry] = self._load_all_routes()
        self.active_route: Optional[RouteGeometry] = None

        # По умолчанию выбираем прямой маршрут "щук-талл" (при отсутствии GNSS)
        for r_name in ["shchuk_tall", "щукинская - таллинская"]:
            if r_name in self.routes:
                self.active_route = self.routes[r_name]
                break
        if self.active_route is None and self.routes:
            self.active_route = next(iter(self.routes.values()))

        # ---------------------------------------------------------------------
        # Состояние одометрии
        # ---------------------------------------------------------------------
        self.s_0: float = 0.0
        self.accumulated_distance: float = 0.0
        self.last_vel_time: Optional[float] = None
        self.last_v_est: float = 0.0
        self.gnss_initialized: bool = False
        self.last_master_fix: Optional[Tuple[float, float, float, float]] = None
        self.last_rover_fix: Optional[Tuple[float, float, float, float]] = None

        # ---------------------------------------------------------------------
        # Параметры выравнивания дрейфа по GNSS
        # ---------------------------------------------------------------------
        self.declare_parameter("enable_gnss_drift_correction", True)
        self.declare_parameter("gnss_corr_max_lateral_dev_m", 3.5)
        self.declare_parameter("gnss_corr_max_longitudinal_dev_m", 25.0)
        self.declare_parameter("gnss_corr_gain", 0.15)
        self.declare_parameter("gnss_corr_gain_stopped", 0.50)
        self.declare_parameter("gnss_min_interval_sec", 0.5)

        self.enable_gnss_drift_correction = self.get_parameter("enable_gnss_drift_correction").get_parameter_value().bool_value
        self.gnss_corr_max_lateral_dev_m = self.get_parameter("gnss_corr_max_lateral_dev_m").get_parameter_value().double_value
        self.gnss_corr_max_longitudinal_dev_m = self.get_parameter("gnss_corr_max_longitudinal_dev_m").get_parameter_value().double_value
        self.gnss_corr_gain = self.get_parameter("gnss_corr_gain").get_parameter_value().double_value
        self.gnss_corr_gain_stopped = self.get_parameter("gnss_corr_gain_stopped").get_parameter_value().double_value
        self.gnss_min_interval_sec = self.get_parameter("gnss_min_interval_sec").get_parameter_value().double_value

        self.last_gnss_corr_time: Optional[float] = None
        self.gnss_corr_count: int = 0
        self.gnss_total_correction_m: float = 0.0

        # ---------------------------------------------------------------------
        # Издатель /result/position
        # ---------------------------------------------------------------------
        qos_pub = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.pub_pos = self.create_publisher(Odometry, "/result/position", qos_pub)

        # ---------------------------------------------------------------------
        # Подписчики
        # ---------------------------------------------------------------------
        # Основной источник скорости — топик /result/velocity от VelocityNode
        self.sub_vel = self.create_subscription(
            VelocitySensor,
            "/result/velocity",
            self.cb_velocity,
            qos_profile_sensor_data
        )

        # Резервный источник скорости (на случай запуска без VelocityNode)
        self.sub_front_raw = self.create_subscription(
            VelocitySensor,
            "/vehicle/front_bogie_velocity",
            self.cb_front_raw_fallback,
            qos_profile_sensor_data
        )

        # Топики GNSS для автоматического выбора карты и стартовой привязки
        self.sub_gnss_master = self.create_subscription(
            NavSatFix,
            "/sensing/gnss/master/fix",
            self.cb_gnss_master,
            qos_profile_sensor_data
        )
        self.sub_gnss_rover = self.create_subscription(
            NavSatFix,
            "/sensing/gnss/rover/fix",
            self.cb_gnss_rover,
            qos_profile_sensor_data
        )

        self.received_result_vel = False
        self.msg_count = 0
        self.get_logger().info(
            f"PositionNode успешно инициализирована. Загружено маршрутов: {len(self.routes)}. "
            f"Ожидание GNSS для автоматического выбора направления..."
        )

    def _load_all_routes(self) -> Dict[str, RouteGeometry]:
        """Ищет и загружает все цифровые карты путей из известных папок."""
        search_dirs = [
            Path("/opt/pathgrath"),
            Path.cwd() / "pathgrath",
            Path(__file__).resolve().parent.parent / "pathgrath",
            Path(__file__).resolve().parent.parent.parent / "pathgrath",
        ]
        if HAS_AMENT:
            try:
                pkg_share = Path(get_package_share_directory("solution"))
                search_dirs.insert(0, pkg_share / "pathgrath")
            except Exception:
                pass

        routes = {}
        for s_dir in search_dirs:
            if not s_dir.exists():
                continue
            for jf in s_dir.glob("*.json"):
                # Пропускаем резервные копии .orig.json
                if jf.name.endswith(".orig.json"):
                    continue
                name_key = jf.stem.lower()
                if name_key not in routes:
                    try:
                        rg = RouteGeometry(jf.stem, jf)
                        routes[name_key] = rg
                        self.get_logger().info(f"Загружена карта пути: '{jf.name}' ({rg.total_length:.1f} м, {len(rg.pts_x)} точек)")
                    except Exception as e:
                        self.get_logger().warn(f"Не удалось загрузить {jf}: {e}")
        return routes

    def cb_gnss_master(self, msg: NavSatFix):
        if not math.isfinite(msg.latitude) or not math.isfinite(msg.longitude):
            return
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.last_master_fix = (t, msg.latitude, msg.longitude, msg.altitude)

        if not self.gnss_initialized:
            self._try_auto_select_route_and_init()
        elif self.enable_gnss_drift_correction:
            self._correct_position_drift_from_gnss(t, msg.latitude, msg.longitude, msg.altitude)

    def cb_gnss_rover(self, msg: NavSatFix):
        if not math.isfinite(msg.latitude) or not math.isfinite(msg.longitude):
            return
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.last_rover_fix = (t, msg.latitude, msg.longitude, msg.altitude)

        if not self.gnss_initialized:
            self._try_auto_select_route_and_init()

    def _try_auto_select_route_and_init(self):
        """Автоматический выбор карты пути и привязка начальной путевой координаты."""
        if self.gnss_initialized or self.last_master_fix is None:
            return

        tm, lat_m, lon_m, alt_m = self.last_master_fix
        has_valid_rover = False
        ux, uy = 1.0, 0.0

        try:
            xm, ym = wgs84_to_map(lat_m, lon_m)

            if self.last_rover_fix is not None:
                tr, lat_r, lon_r, _ = self.last_rover_fix
                if abs(tm - tr) <= 2.0:
                    xr, yr = wgs84_to_map(lat_r, lon_r)
                    dx = xr - xm
                    dy = yr - ym
                    dist = math.hypot(dx, dy)
                    if dist >= 2.0:
                        ux = dx / dist
                        uy = dy / dist
                        has_valid_rover = True

            # Если rover доступен, рассчитываем точный base_link по вектору антенн
            if has_valid_rover:
                x_base = xm + 9.873 * ux
                y_base = ym + 9.873 * uy
            else:
                # При отсутствии rover ориентируемся по координатам master антенны
                x_base = xm
                y_base = ym

            z_base = alt_m - 3.0 if math.isfinite(alt_m) else 0.0

            # -----------------------------------------------------------------
            # Автоматический выбор наилучшей карты из загруженных маршрутов
            # -----------------------------------------------------------------
            best_route = None
            best_dev_sq = float("inf")
            best_s0 = 0.0

            for route in self.routes.values():
                dists_sq = (route.pts_x - x_base) ** 2 + (route.pts_y - y_base) ** 2
                min_idx = int(np.argmin(dists_sq))
                dev_sq = dists_sq[min_idx]

                if dev_sq < best_dev_sq:
                    best_dev_sq = dev_sq
                    best_route = route
                    best_s0 = float(route.s_map[min_idx])

            if best_route is not None:
                # Если rover не было, уточняем проекцию base_link по касательной рельса
                if not has_valid_rover:
                    tang_val = float(np.interp(best_s0, best_route.s_map, best_route.pts_tang))
                    x_base = xm + 9.873 * math.cos(tang_val)
                    y_base = ym + 9.873 * math.sin(tang_val)
                    dists_sq = (best_route.pts_x - x_base) ** 2 + (best_route.pts_y - y_base) ** 2
                    min_idx = int(np.argmin(dists_sq))
                    best_s0 = float(best_route.s_map[min_idx])
                    best_dev_sq = dists_sq[min_idx]

                self.active_route = best_route
                self.s_0 = best_s0
                dev_m = math.sqrt(best_dev_sq)
                self.gnss_initialized = True

                self.get_logger().info(
                    f"✓ [АВТОВЫБОР ПАФГРАФА]: Маршрут автоматически определен: '{best_route.name}'. "
                    f"Начальная позиция base_link: ({x_base:.1f}, {y_base:.1f}, {z_base:.1f}), "
                    f"s0 = {self.s_0:.1f} м (отклонение от рельс: {dev_m:.2f} м, rover={'ДА' if has_valid_rover else 'НЕТ'})"
                )
        except Exception as e:
            self.get_logger().warn(f"Ошибка при автовыборе карты: {e}")

    def _correct_position_drift_from_gnss(self, tm: float, lat_m: float, lon_m: float, alt_m: float):
        """
        Выравнивание накопленного продольного дрейфа позиции по спутниковым данным.
        Вызывается при наличии промежуточных сообщений GNSS на маршруте.
        """
        if self.active_route is None:
            return

        # Ограничение частоты коррекции
        if self.last_gnss_corr_time is not None and (tm - self.last_gnss_corr_time) < self.gnss_min_interval_sec:
            return

        rg = self.active_route
        s_cur = self.s_0 + self.accumulated_distance
        if s_cur < 5.0 or s_cur > (rg.total_length - 5.0):
            return

        try:
            xm, ym = wgs84_to_map(lat_m, lon_m)

            # Вычисление координат base_link:
            used_rover = False
            if self.last_rover_fix is not None:
                tr, lat_r, lon_r, _ = self.last_rover_fix
                if abs(tm - tr) <= 1.0:
                    xr, yr = wgs84_to_map(lat_r, lon_r)
                    dx = xr - xm
                    dy = yr - ym
                    dist = math.hypot(dx, dy)
                    if dist >= 2.0:
                        ux = dx / dist
                        uy = dy / dist
                        x_base = xm + 9.873 * ux
                        y_base = ym + 9.873 * uy
                        used_rover = True

            if not used_rover:
                tang_val = float(np.interp(s_cur, rg.s_map, rg.pts_tang))
                x_base = xm + 9.873 * math.cos(tang_val)
                y_base = ym + 9.873 * math.sin(tang_val)

            # Локальный поиск ближайшей точки пути в окне вокруг текущей координаты s_cur
            idx_cur = int(np.searchsorted(rg.s_map, s_cur))
            w_pts = 60  # ~60 метров вдоль пути
            i_lo = max(0, idx_cur - w_pts)
            i_hi = min(len(rg.s_map), idx_cur + w_pts)
            if i_hi <= i_lo:
                return

            sub_x = rg.pts_x[i_lo:i_hi]
            sub_y = rg.pts_y[i_lo:i_hi]
            dists_sq = (sub_x - x_base) ** 2 + (sub_y - y_base) ** 2
            min_k = int(np.argmin(dists_sq))
            lat_dev = math.sqrt(dists_sq[min_k])

            # 1. Отсечение выбросов по боковому расстоянию до рельсов
            if lat_dev > self.gnss_corr_max_lateral_dev_m:
                return

            s_gnss = float(rg.s_map[i_lo + min_k])
            ds = s_gnss - s_cur

            # 2. Отсечение нереалистичных продольных скачков
            if abs(ds) > self.gnss_corr_max_longitudinal_dev_m:
                return

            # 3. Мягкая коррекция: на остановках сходимся быстрее, в движении — плавно (без рывков)
            gain = self.gnss_corr_gain_stopped if abs(self.last_v_est) < 0.1 else self.gnss_corr_gain
            corr = gain * ds

            self.s_0 += corr
            self.last_gnss_corr_time = tm
            self.gnss_corr_count += 1
            self.gnss_total_correction_m += corr

            if self.gnss_corr_count % 10 == 1:
                self.get_logger().info(
                    f"[GNSS Drift Alignment] Невязка Δs={ds:+.2f}м | Правка={corr:+.2f}м | "
                    f"s={self.s_0 + self.accumulated_distance:.1f}м | Откл. от рельс={lat_dev:.2f}м "
                    f"(всего правок: {self.gnss_corr_count})"
                )
        except Exception as e:
            self.get_logger().warn(f"Ошибка выравнивания дрейфа по GNSS: {e}")


    def cb_velocity(self, msg: VelocitySensor):
        """Основной обработчик оцененной скорости из /result/velocity."""
        self.received_result_vel = True
        self._process_speed_sample(msg.header.stamp, msg.velocity)

    def cb_front_raw_fallback(self, msg: VelocitySensor):
        """Резервный обработчик при отсутствии VelocityNode."""
        if not self.received_result_vel:
            v_ms = msg.velocity / 3.6
            self._process_speed_sample(msg.header.stamp, v_ms)

    def _process_speed_sample(self, stamp, v_est: float):
        t = stamp.sec + stamp.nanosec * 1e-9

        if self.last_vel_time is None:
            self.last_vel_time = t
            self.last_v_est = v_est
            self._publish_position(stamp, v_est)
            return

        dt = t - self.last_vel_time
        if dt < 0.0:
            dt = 0.0
        elif dt > 2.0:
            dt = 0.0
            self.last_vel_time = t

        if dt > 0.0:
            # Численное интегрирование перемещения методом трапеций
            v_eff = 0.5 * (self.last_v_est + v_est)
            self.accumulated_distance += v_eff * dt
            self.last_vel_time = t
            self.last_v_est = v_est

        self._publish_position(stamp, v_est)

    def _publish_position(self, stamp, v_est: float):
        odom_msg = Odometry()
        odom_msg.header.stamp = stamp
        odom_msg.header.frame_id = self.frame_id
        odom_msg.child_frame_id = self.child_frame_id

        if self.active_route is not None:
            rg = self.active_route
            s_cur = max(0.0, min(self.s_0 + self.accumulated_distance, rg.total_length))

            odom_msg.pose.pose.position.x = float(np.interp(s_cur, rg.s_map, rg.pts_x))
            odom_msg.pose.pose.position.y = float(np.interp(s_cur, rg.s_map, rg.pts_y))
            odom_msg.pose.pose.position.z = float(np.interp(s_cur, rg.s_map, rg.pts_z))

            tang_val = float(np.interp(s_cur, rg.s_map, rg.pts_tang))
            odom_msg.pose.pose.orientation.z = math.sin(tang_val / 2.0)
            odom_msg.pose.pose.orientation.w = math.cos(tang_val / 2.0)
        else:
            odom_msg.pose.pose.position.x = float(self.accumulated_distance)
            odom_msg.pose.pose.orientation.w = 1.0

        odom_msg.twist.twist.linear.x = float(v_est)
        self.pub_pos.publish(odom_msg)

        self.msg_count += 1
        if self.msg_count % 500 == 0:
            route_label = self.active_route.name if self.active_route else "UNKNOWN"
            cur_s = self.s_0 + self.accumulated_distance
            pos = odom_msg.pose.pose.position
            self.get_logger().info(
                f"[{route_label}] s={cur_s:.1f} м | XYZ=({pos.x:.1f}, {pos.y:.1f}, {pos.z:.1f}) | V={v_est*3.6:.1f} км/ч"
            )


def main(args=None):
    rclpy.init(args=args)
    node = PositionNode()
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

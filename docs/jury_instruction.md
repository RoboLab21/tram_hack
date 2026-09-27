# Инструкция для жюри по проверке решения

---

## Быстрый старт (Docker Compose)

Все компоненты уже настроены и готовы к запуску.

### 1. Запуск ноды одометрии
```bash
cd tram_hack
docker compose up -d solution
```
*Нода автоматически загружает цифровую карту рельсового пути (`/opt/pathgrath`) и начинает слушать топики телеметрии.*

### 2. Запуск проверки метрик (Checker)
В отдельном терминале запустите автоматический валидатор:
```bash
docker compose --profile metrics up checker
```

### 3. Воспроизведение тестового rosbag
Положите нужный rosbag в папку `tram_hack/bags/` (или используйте уже имеющиеся в `check-code/bags/`):
```bash
docker compose run --rm bag_player bash -c "
  source /opt/ws/install/setup.bash &&
  ros2 bag play /opt/bags/30618_88aea4d9
"
```

---

## Остановка и очистка
```bash
docker compose --profile metrics --profile test down
```

---

## Интерфейс топиков ROS 2

### Входные топики (из rosbag / бортовой сети вагона)
| Топик | Тип сообщения | Описание |
|---|---|---|
| `/vehicle/front_bogie_velocity` | `tram_vehicle_msgs/msg/VelocitySensor` | Линейная скорость передней тележки |
| `/vehicle/rear_bogie_velocity` | `tram_vehicle_msgs/msg/VelocitySensor` | Линейная скорость задней тележки |
| `/vehicle/driver_position_cmd` | `tram_vehicle_msgs/msg/DriverControllerCommand` | Положение контроллера водителя (тяга / тормоз) |
| `/sensing/gnss/master/fix` | `sensor_msgs/msg/NavSatFix` | Антенна GNSS Master (автовыбор маршрута и сброс дрейфа) |
| `/sensing/gnss/rover/fix` | `sensor_msgs/msg/NavSatFix` | Антенна GNSS Rover (определение курсового угла) |

### Выходные топики (публикуются решением)
| Топик | Тип сообщения | Частота | Описание |
|---|---|---|---|
| `/result/velocity` | `tram_vehicle_msgs/msg/VelocitySensor` | 20 Гц | Оцененная фильтрованная скорость вагона (`base_link`) |
| `/result/position` | `nav_msgs/msg/Odometry` | 20 Гц | 3D координаты `x, y, z`, ориентация (кватернион) и скорость |

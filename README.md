# Резервная одометрия беспилотного трамвая (ROS 2 Humble)

Программный комплекс высокоточной оценки скорости и 3D-позиционирования трамвая при деградации или отсутствии спутниковой навигации (GNSS).

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
В отдельном терминале запустите автоматический валидатор организаторов:
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
*Чекер автоматически рассчитает и выведет в консоль значения RMSE и максимальные ошибки по скорости и положению.*

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

---

## Настройка параметров (`solution/config/params.yaml`)

Основные параметры конфигурируются в файле [`params.yaml`](solution/config/params.yaml):

```yaml
velocity_node:
  ros__parameters:
    input_in_kmh: true                  # Входные скорости датчиков в км/ч (true)
    integration_method: "trapezoidal"   # Численное интегрирование ("trapezoidal" / "rectangular")
    velocity_filter: "mean"             # Фильтрация скорости ("mean", "ema", "sma")
    enable_brake_filter: true           # Защита от юза и ZUPT-отсечка ползучего шума при торможении
    enable_curve_compensation: true     # Геометрическая компенсация забегания колес в кривых

position_node:
  ros__parameters:
    frame_id: "map"                     # Глобальная СК
    child_frame_id: "base_link"         # Базовый фрейм трамвая
    enable_gnss_drift_correction: true  # Мягкое выравнивание продольного дрейфа по GNSS (true по умолчанию)
```

---

## Дополнительная документация
- ⚙️ **[Архитектура и алгоритмическое ядро](solution/README.md)** — математические формулы фильтров, геометрическая модель трамвая, калибровка.

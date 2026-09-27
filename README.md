# Docker Setup for Tram Hackathon

## Prerequisites

1. Build the checker image first (from `check-code/` directory):
   ```bash
   cd ../check-code
   ./scripts/build.sh
   ```

2. Place your bag files in `tram_hack/bags/` directory.

## Quick Start

### Build and run solution node:
```bash
docker compose build
docker compose up solution
```

### Run with metrics checker:
```bash
docker compose --profile metrics up
```

### Run with bag player for testing:
```bash
docker compose --profile test up

# In another terminal, play a bag:
docker exec -it tram-bag-player bash
source /opt/ws/install/setup.bash
ros2 bag play /opt/bags/<bag_name>
```

## Configuration

Edit `.env` to set `ROS_DOMAIN_ID` (default: 0).

## Services

| Service | Profile | Description |
|---------|---------|-------------|
| `solution` | default | Main odometry node (auto-start) |
| `solution-2` | multi | Second instance with different route |
| `checker` | metrics | Hackathon solution checker |
| `bag_player` | test | Manual bag playback container |

## Entering Containers

```bash
# Enter solution container
docker compose exec solution bash

# Enter checker container
docker compose --profile metrics exec checker bash
```

## ROS 2 Topics

- **Input** (from bag):
  - `/vehicle/front_bogie_velocity`
  - `/vehicle/rear_bogie_velocity`
  - `/vehicle/driver_position_cmd`

- **Output** (from solution):
  - `/result/velocity` (tram_vehicle_msgs/msg/VelocitySensor)
  - `/result/position` (nav_msgs/msg/Odometry)

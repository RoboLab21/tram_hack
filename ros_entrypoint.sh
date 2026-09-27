#!/bin/bash
set -e

source /opt/ros/humble/setup.bash

if [ -f /opt/ros2_ws/install/setup.bash ]; then
    source /opt/ros2_ws/install/setup.bash
fi

exec "$@"

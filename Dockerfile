FROM ros:humble-ros-base

SHELL ["/bin/bash", "-c"]

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3-pip \
        python3-numpy \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ros2_ws

COPY tram_vehicle_msgs/ src/tram_vehicle_msgs/
COPY solution/ src/solution/

RUN . /opt/ros/humble/setup.sh \
    && colcon build --packages-select tram_vehicle_msgs solution \
    && rm -rf build log

COPY pathgrath/ /opt/pathgrath/

COPY ros_entrypoint.sh /
RUN chmod +x /ros_entrypoint.sh

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["ros2", "run", "solution", "recovery_odometry_node"]

# Cartographer and Foxglove bridge for mapping WarpTracer runs. CPU only.
# Build from the repository root:
#   docker build -f docker/mapping/Dockerfile -t race-stack-mapping .
FROM ros:humble-ros-base

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ros-humble-cartographer-ros \
        ros-humble-foxglove-bridge \
        ros-humble-nav2-map-server \
        ros-humble-tf2-ros \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ws
COPY roboracer_slam src/roboracer_slam
COPY warptracer_bridge src/warptracer_bridge
RUN bash -c "source /opt/ros/humble/setup.bash && colcon build"

ENTRYPOINT ["/bin/bash", "-c", "source /ws/install/setup.bash && exec \"$@\"", "--"]
CMD ["bash"]

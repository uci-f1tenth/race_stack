# Offline cartographer tuning on a laptop: same cartographer as the car, plus
# Python bindings for its .pbstream format (apt doesn't install the .proto files).
FROM ros:humble-ros-base
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-humble-cartographer-ros python3-numpy python3-protobuf protobuf-compiler git \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch ros2 https://github.com/ros2/cartographer /tmp/carto \
    && mkdir -p /opt/carto_py && cd /tmp/carto \
    && protoc --python_out=/opt/carto_py $(find cartographer -name "*.proto" ! -path "*/cloud/*") \
    && rm -rf /tmp/carto
ENV PYTHONPATH=/opt/carto_py
# car_model.py: the drive commands' message type, and scipy for the fit
RUN apt-get update && apt-get install -y --no-install-recommends ros-humble-ackermann-msgs python3-scipy \
    && rm -rf /var/lib/apt/lists/*

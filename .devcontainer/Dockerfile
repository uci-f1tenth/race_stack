FROM --platform=linux/amd64 autodriveecosystem/autodrive_roboracer_api:2026-icra-practice

RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    ros-$ROS_DISTRO-foxglove-bridge \
    ros-$ROS_DISTRO-slam-toolbox \
    ros-$ROS_DISTRO-cartographer \
    ros-$ROS_DISTRO-cartographer-ros \
    ros-$ROS_DISTRO-nav2-amcl \
    ros-$ROS_DISTRO-nav2-lifecycle-manager

RUN pip3 install torch scipy scikit-image typer numpy==1.22.2

RUN echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> ~/.bashrc
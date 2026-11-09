from setuptools import find_packages, setup

package_name = "race_bringup"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/bringup_launch.py"]),
        (
            "share/" + package_name + "/config",
            ["config/vesc.yaml", "config/sensors.yaml", "config/mux.yaml"],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="root",
    maintainer_email="root@todo.todo",
    description="Car bringup (replaces f1tenth_stack's) and IMU-fused odometry.",
    license="TODO: License declaration",
    entry_points={
        "console_scripts": [
            "imu_odom = race_bringup.imu_odom:main",
            "low_voltage_shutdown = race_bringup.low_voltage_shutdown:main",
        ],
    },
)

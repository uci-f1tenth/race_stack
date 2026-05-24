from setuptools import setup

package_name = "roboracer_slam"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            "share/" + package_name + "/launch",
            [
                "launch/roboracer_cartographer.launch.py",
                "launch/roboracer_cartographer_localization.launch.py",
            ],
        ),
        (
            "share/" + package_name + "/config",
            [
                "config/roboracer_cartographer.lua",
                "config/roboracer_cartographer_localization.lua",
            ],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Your Name",
    maintainer_email="you@example.com",
    description="Cartographer SLAM bringup for the RoboRacer / AutoDRIVE platform.",
    license="MIT",
    entry_points={"console_scripts": []},
)

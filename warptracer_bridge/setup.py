from glob import glob

from setuptools import setup

package_name = "warptracer_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Your Name",
    maintainer_email="you@example.com",
    description="Publishes WarpTracer simulator recordings as ROS 2 topics.",
    license="MIT",
    entry_points={"console_scripts": ["replay = warptracer_bridge.replay:main",
                                      "layer_mapper = warptracer_bridge.layer_mapper:main",
                                      "export_maps = warptracer_bridge.export_maps:main"]},
)

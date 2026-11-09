from setuptools import find_packages, setup

package_name = "deadman_switch"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "deadman_switch.html"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="root",
    maintainer_email="root@todo.todo",
    description="Websocket deadman switch gating /drive into the ackermann mux.",
    license="TODO: License declaration",
    entry_points={
        "console_scripts": ["deadman_switch = deadman_switch.deadman_switch:main"],
    },
)

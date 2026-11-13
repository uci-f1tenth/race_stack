from glob import glob

from setuptools import find_packages, setup

package_name = "warporacer_node"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        # The policy checkpoint is not in git; install it if it's been copied here.
        ("share/" + package_name, ["package.xml"] + glob("agent_final.pt")),
        ("share/" + package_name + "/launch", ["launch/warporacer_node.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="root",
    maintainer_email="root@todo.todo",
    description="Inference node for the warporacer PPO policy.",
    license="TODO: License declaration",
    entry_points={
        "console_scripts": ["warporacer_node = warporacer_node.warporacer_node:main"],
    },
)

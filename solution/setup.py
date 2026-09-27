import os
from glob import glob
from setuptools import setup

package_name = "solution"

# Карты пути (pathgrath) для установки в share
map_files = glob("../pathgrath/*.json")
if not map_files:
    map_files = glob("pathgrath/*.json")

setup(
    name=package_name,
    version="1.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "pathgrath"), map_files),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="Команда хакатона Московского Транспорта",
    maintainer_email="hackathon@moscow.transport",
    description="Резервная одометрия и счисление пути беспилотного трамвая по математической модели и колесным датчикам",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "velocity_node = solution.velocity_node:main",
            "position_node = solution.position_node:main",
            "recovery_odometry_node = solution.recovery_odometry_node:main",
        ],
    },

)

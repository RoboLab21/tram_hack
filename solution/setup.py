from setuptools import setup

package_name = 'solution'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='User',
    maintainer_email='user@example.com',
    description='Tram odometry solution',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'recovery_odometry_node = solution.recovery_odometry_node:main',
        ],
    },
)

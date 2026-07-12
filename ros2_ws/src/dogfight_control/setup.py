from setuptools import find_packages, setup

package_name = 'dogfight_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Yunus Emre Tom',
    maintainer_email='yunusemretom@gmail.com',
    description='Flight control nodes for DogFight project (attitude, velocity, position, visual offboard)',
    license='BSD-3-Clause',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'attitude_setpoint_node = dogfight_control.attitude_setpoint_node:main',
            'trajectory_velocity_node = dogfight_control.trajectory_velocity_node:main',
            'trajectory_position_node = dogfight_control.trajectory_position_node:main',
            'visual_offboard_node = dogfight_control.visual_offboard_node:main',
            'px4_status_monitor = dogfight_control.px4_status_monitor:main',
        ],
    },
)

from setuptools import find_packages, setup

package_name = 'dogfight_tracking'

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
    description='Target tracking nodes for DogFight project (GPS tracking, visual tracking)',
    license='BSD-3-Clause',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'gps_tracker_node = dogfight_tracking.gps_tracker_node:main',
            'visual_tracker_node = dogfight_tracking.visual_tracker_node:main',
        ],
    },
)

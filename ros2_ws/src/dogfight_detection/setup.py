from setuptools import find_packages, setup

package_name = 'dogfight_detection'

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
    description='Object detection nodes for DogFight project (YOLO, RF-DETR)',
    license='BSD-3-Clause',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'yolo_detection_node = dogfight_detection.yolo_detection_node:main',
            'rfdetr_detection_node = dogfight_detection.rfdetr_detection_node:main',
        ],
    },
)

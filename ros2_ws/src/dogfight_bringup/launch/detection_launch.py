"""
Detection pipeline launch file.
Starts YOLO or RF-DETR detection node.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    detector_arg = DeclareLaunchArgument(
        'detector',
        default_value='yolo',
        description='Detection model to use: yolo or rfdetr'
    )

    yolo_node = Node(
        package='dogfight_detection',
        executable='yolo_detection_node',
        name='yolo_detection_node',
        output='screen',
        parameters=[],
        condition=None,  # TODO: add condition based on 'detector' arg
    )

    return LaunchDescription([
        detector_arg,
        yolo_node,
    ])

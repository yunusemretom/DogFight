"""
Full system launch file.
Starts detection, tracking, and control nodes together.
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    yolo_detection = Node(
        package='dogfight_detection',
        executable='yolo_detection_node',
        name='yolo_detection_node',
        output='screen',
    )

    gps_tracker = Node(
        package='dogfight_tracking',
        executable='gps_tracker_node',
        name='gps_tracker_node',
        output='screen',
    )

    visual_offboard = Node(
        package='dogfight_control',
        executable='visual_offboard_node',
        name='visual_offboard_node',
        output='screen',
    )

    return LaunchDescription([
        yolo_detection,
        gps_tracker,
        visual_offboard,
    ])

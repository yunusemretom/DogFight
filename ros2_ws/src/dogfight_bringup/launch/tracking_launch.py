"""
Tracking + Control pipeline launch file.
Starts GPS tracker and attitude/visual offboard controller.
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    gps_tracker = Node(
        package='dogfight_tracking',
        executable='gps_tracker_node',
        name='gps_tracker_node',
        output='screen',
    )

    attitude_controller = Node(
        package='dogfight_control',
        executable='attitude_controller_node',
        name='attitude_controller_node',
        output='screen',
    )

    return LaunchDescription([
        gps_tracker,
        attitude_controller,
    ])

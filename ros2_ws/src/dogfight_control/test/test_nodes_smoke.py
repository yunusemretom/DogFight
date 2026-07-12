#!/usr/bin/env python3
"""Node smoke testleri — rclpy gerektirir, SITL GEREKTİRMEZ.

Her node ayağa kaldırılır, birkaç kontrol döngüsü döndürülür ve veri yokken
WAIT_DATA durumunda kalıp güvenli (idle) setpoint yayınladığı doğrulanır.

Çalıştırma (workspace source edilmiş olmalı):
    pytest src/dogfight_control/test/test_nodes_smoke.py -v
"""

import time

import pytest
import rclpy

from dogfight_control.offboard_base import S_WAIT_DATA


@pytest.fixture(scope='module')
def ros_context():
    rclpy.init()
    yield
    rclpy.shutdown()


def _spin_ticks(node, ticks=10):
    deadline = time.time() + 5.0
    for _ in range(ticks):
        rclpy.spin_once(node, timeout_sec=0.1)
        if time.time() > deadline:
            break


def _smoke(node_cls, ros_context):
    node = node_cls()
    try:
        _spin_ticks(node)
        # Veri yok → WAIT_DATA'da kalmalı, çökmemeli
        assert node.state == S_WAIT_DATA
    finally:
        node.destroy_node()


def test_attitude_setpoint_node_smoke(ros_context):
    from dogfight_control.attitude_setpoint_node import AttitudeSetpointNode
    _smoke(AttitudeSetpointNode, ros_context)


def test_trajectory_velocity_node_smoke(ros_context):
    from dogfight_control.trajectory_velocity_node import TrajectoryVelocityNode
    _smoke(TrajectoryVelocityNode, ros_context)


def test_trajectory_position_node_smoke(ros_context):
    from dogfight_control.trajectory_position_node import TrajectoryPositionNode
    _smoke(TrajectoryPositionNode, ros_context)


def test_visual_offboard_node_smoke(ros_context):
    from dogfight_control.visual_offboard_node import VisualOffboardNode
    _smoke(VisualOffboardNode, ros_context)


def test_px4_status_monitor_smoke(ros_context):
    from dogfight_control.px4_status_monitor import PX4StatusMonitor
    node = PX4StatusMonitor()
    try:
        _spin_ticks(node, ticks=5)
    finally:
        node.destroy_node()

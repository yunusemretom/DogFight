#!/usr/bin/env python
############################################################################
#
#   Copyright (C) 2022 PX4 Development Team. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
# 3. Neither the name PX4 nor the names of its contributors may be
#    used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
# AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
############################################################################

__author__ = "Jaeyoung Lim"
__contact__ = "jalim@ethz.ch"

import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint, ActuatorMotors
from px4_msgs.msg import VehicleStatus
from px4_msgs.msg import VehicleLocalPosition
import math

class OffboardControl(Node):

    def __init__(self):
        super().__init__('minimal_publisher')

                # QoS profiles
        qos_profile_pub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        qos_profile_sub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.status_sub = self.create_subscription(
            VehicleStatus,
            '/fmu/out/vehicle_status_v4',
            self.vehicle_status_callback,
            qos_profile_sub)
        # Hedef aracın (px4_2) lokal pozisyonu
        self.tgt_pos_sub = self.create_subscription(
            VehicleLocalPosition,
            '/px4_2/fmu/out/vehicle_local_position_v1',
            self.tgt_local_pos_callback,
            qos_profile_sub)
        self.publisher_offboard_mode = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile_pub)
        self.publisher_trajectory = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile_pub)
        self.publisher_actuator = self.create_publisher(ActuatorMotors, '/fmu/in/actuator_motors', qos_profile_pub)
        
        
        timer_period = 0.02  # seconds
        self.timer = self.create_timer(timer_period, self.cmdloop_callback)
        self.dt = timer_period
        self.declare_parameter('radius', 10.0)
        self.declare_parameter('omega', 5.0)
        self.declare_parameter('altitude', 20.0)
        self.declare_parameter('yaw_deg', 30.0)
        self.declare_parameter('target_x', 0.0)
        self.declare_parameter('target_y', 0.0)
        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX
        self.arming_state = VehicleStatus.ARMING_STATE_DISARMED
        self.theta = 0.0
        self.radius = self.get_parameter('radius').value
        self.omega = self.get_parameter('omega').value
        self.altitude = self.get_parameter('altitude').value
        self.yaw_deg = self.get_parameter('yaw_deg').value
        self.target_x = self.get_parameter('target_x').value
        self.target_y = self.get_parameter('target_y').value
        # Hedef (px4_2) NED lokal pozisyon — None = henüz veri gelmedi
        self.tgt_x  = None
        self.tgt_y  = None
        self.tgt_z  = None
        self.tgt_vx = 0.0
        self.tgt_vy = 0.0

    def vehicle_status_callback(self, msg):
        # TODO: handle NED->ENU transformation
        print("NAV_STATUS: ", msg.nav_state)
        print("  - offboard status: ", VehicleStatus.NAVIGATION_STATE_OFFBOARD)
        self.nav_state = msg.nav_state
        self.arming_state = msg.arming_state

    def tgt_local_pos_callback(self, msg: VehicleLocalPosition):
        """Hedef aracın (px4_2) NED lokal pozisyon ve hız verisini güncelle."""
        self.tgt_x  = msg.x
        self.tgt_y  = msg.y
        self.tgt_z  = msg.z
        self.tgt_vx = msg.vx
        self.tgt_vy = msg.vy

    def cmdloop_callback(self):
        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp    = int(self.get_clock().now().nanoseconds / 1000)
        offboard_msg.position     = False
        offboard_msg.velocity     = True
        offboard_msg.acceleration = False
        offboard_msg.attitude     = False
        offboard_msg.body_rate    = False   # eksikti
        self.publisher_offboard_mode.publish(offboard_msg)

        if (self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD
                and self.arming_state == VehicleStatus.ARMING_STATE_ARMED):

            msg = TrajectorySetpoint()
            msg.timestamp    = int(self.get_clock().now().nanoseconds / 1000)
            if (self.tgt_x is None or self.tgt_y is None or self.tgt_z is None):
                print("No target position available")
                return
            # NED koordinatları: z negatif = yukarı
            print("TGT_POS: ", self.tgt_x, self.tgt_y, self.tgt_z)
            msg.position     = [float('nan'), float('nan'), float('nan')]
            # Position modunda kullanılmayan alanlar nan olmalı
            msg.velocity     = [self.tgt_x, self.tgt_y, self.tgt_z]
            msg.acceleration = [float('nan'), float('nan'), float('nan')]
            msg.jerk         = [float('nan'), float('nan'), float('nan')]
            # yaw=nan → PX4 heading'i position error yönünden otomatik hesaplar
            msg.yaw          = float('nan')

            self.publisher_trajectory.publish(msg)

            self.theta += self.omega * self.dt


def main(args=None):
    rclpy.init(args=args)

    offboard_control = OffboardControl()

    rclpy.spin(offboard_control)

    offboard_control.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
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
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import VehicleStatus
from px4_msgs.msg import VehicleLocalPosition
from px4_msgs.msg import FixedWingLongitudinalSetpoint
import math

class OffboardControl(Node):

    def __init__(self):
        super().__init__('minimal_publisher')

        # QoS profiles
        qos_profile_pub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=0
        )

        qos_profile_sub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=0
        )

        self.status_sub = self.create_subscription(
            VehicleStatus,
            '/px4_1/fmu/out/vehicle_status',
            self.vehicle_status_callback,
            qos_profile_sub)
        self.status_sub = self.create_subscription(
            VehicleStatus,
            '/px4_1/fmu/out/vehicle_status_v1',
            self.vehicle_status_callback,
            qos_profile_sub)
        self.status_sub = self.create_subscription(
            VehicleLocalPosition,
            '/px4_3/fmu/out/vehicle_local_position_v1',
            self.konum_verisi,
            qos_profile_sub)
        self.publisher_fw_longitudinal = self.create_publisher(
            FixedWingLongitudinalSetpoint,
            '/px4_1/fmu/in/fixed_wing_longitudinal_setpoint',
            qos_profile_pub
        )
        self.publisher_offboard_mode = self.create_publisher(OffboardControlMode, '/px4_1/fmu/in/offboard_control_mode', qos_profile_pub,)
        self.publisher_trajectory = self.create_publisher(TrajectorySetpoint, '/px4_1/fmu/in/trajectory_setpoint', qos_profile_pub)
        timer_period = 0.02  # seconds
        self.timer = self.create_timer(timer_period, self.cmdloop_callback)
        self.dt = timer_period
        self.declare_parameter('radius', 10.0)
        self.declare_parameter('omega', 5.0)
        self.declare_parameter('altitude', 5.0)
        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX
        self.arming_state = VehicleStatus.ARMING_STATE_DISARMED
        # Note: no parameter callbacks are used to prevent sudden inflight changes of radii and omega
        # which would result in large discontinuities in setpoints
        self.theta = 0.0
        self.radius = self.get_parameter('radius').value
        self.omega = self.get_parameter('omega').value
        self.altitude = self.get_parameter('altitude').value
    
    def vehicle_status_callback(self, msg):        
        # TODO: handle NED->ENU transformation
        print("NAV_STATUS: ", msg.nav_state)
        print("NAV_STATUS: ", msg.nav_state)
        print("  - offboard status: ", VehicleStatus.NAVIGATION_STATE_OFFBOARD)
        self.nav_state = msg.nav_state
        self.arming_state = msg.arming_state

    def konum_verisi(self, msg_or_x, y=None, z=None):
        """Konum bilgisini yazdır.

        - Subscription callback olarak çağrıldığında: msg_or_x bir VehicleLocalPosition mesajıdır.
        - Manuel çağrıldığında: (x, y, z) sayısal değerleri verilebilir.
        """
        if y is None and z is None:
            msg = msg_or_x
            self.x_val, self.y_val, self.z_val = msg.x, msg.y, msg.z
        else:
            self.x_val, self.y_val, self.z_val = float(msg_or_x), float(y), float(z)

        print("X konumu: ", self.x_val)
        print("Y konumu: ", self.y_val)
        print("Z konumu: ", self.z_val)
        
    def cmdloop_callback(self):
        # Publish offboard control modes
        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        offboard_msg.position=True
        offboard_msg.velocity=True
        offboard_msg.acceleration=False
        self.publisher_offboard_mode.publish(offboard_msg)
        if (self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD and self.arming_state == VehicleStatus.ARMING_STATE_ARMED):
            
            now_us = int(self.get_clock().now().nanoseconds / 1000)
            
            trajectory_msg = TrajectorySetpoint()
            
            trajectory_msg.position[0] = self.x_val
            trajectory_msg.position[1] = self.y_val
            trajectory_msg.position[2] = self.z_val
            
            trajectory_msg.velocity[0] = math.nan
            trajectory_msg.velocity[1] = 3
            trajectory_msg.velocity[2] = math.nan

            trajectory_msg.acceleration[0] = math.nan
            trajectory_msg.acceleration[1] = math.nan
            trajectory_msg.acceleration[2] = math.nan
            
            self.publisher_trajectory.publish(trajectory_msg)

            self.theta = self.theta + self.omega * self.dt


def main(args=None):
    rclpy.init(args=args)

    offboard_control = OffboardControl()

    rclpy.spin(offboard_control)

    offboard_control.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

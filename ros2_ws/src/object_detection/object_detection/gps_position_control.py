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

__author__ = "Jaeyoung Lim"#!/usr/bin/env python
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
from dataclasses import dataclass


@dataclass
class LocalPosition:
    x: float
    y: float
    z: float

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
        self.status_sub_v1 = self.create_subscription(
            VehicleStatus,
            '/px4_1/fmu/out/vehicle_status_v1',
            self.vehicle_status_callback,
            qos_profile_sub)

        # İki uçak için konum abonelikleri
        self.pos_sub_px4_2 = self.create_subscription(
            VehicleLocalPosition,
            '/px4_2/fmu/out/vehicle_local_position_v1',
            lambda msg: self.konum_verisi('px4_2', msg),
            qos_profile_sub)
        self.pos_sub_px4_1_v1 = self.create_subscription(
            VehicleLocalPosition,
            '/px4_1/fmu/out/vehicle_local_position_v1',
            lambda msg: self.konum_verisi('px4_1', msg),
            qos_profile_sub)

        # Bazı kurulumlarda namespace'siz topic yayınlanıyor olabilir; px4_1 için yedek abonelik
        self.pos_sub_fallback = self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position',
            lambda msg: self.konum_verisi('px4_1', msg),
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

        # Son bilinen konumlar
        self.px4_1_pos: LocalPosition | None = None
        self.px4_2_pos: LocalPosition | None = None
        self.last_distance_m: float | None = None
    
    def vehicle_status_callback(self, msg):        
        # TODO: handle NED->ENU transformation
        print("NAV_STATUS: ", msg.nav_state)
        print("NAV_STATUS: ", msg.nav_state)
        print("  - offboard status: ", VehicleStatus.NAVIGATION_STATE_OFFBOARD)
        self.nav_state = msg.nav_state
        self.arming_state = msg.arming_state

    def konum_verisi(self, vehicle: str, msg_or_x, y=None, z=None):
        """İki uçağın konumunu güncelle ve aralarındaki mesafeyi hesapla.

        - Subscription callback: msg_or_x bir VehicleLocalPosition mesajıdır.
        - Manuel çağrı: (x, y, z) sayısal değerleri verilebilir.
        """
        if y is None and z is None:
            msg = msg_or_x
            pos = LocalPosition(float(msg.x), float(msg.y), float(msg.z))
        else:
            pos = LocalPosition(float(msg_or_x), float(y), float(z))

        if vehicle == 'px4_1':
            self.px4_1_pos = pos
        elif vehicle == 'px4_2':
            self.px4_2_pos = pos
        else:
            return

        # İki konum da mevcutsa mesafe hesapla
        self._hesapla_ve_yazdir_mesafe()

    def _hesapla_ve_yazdir_mesafe(self):
        if self.px4_1_pos is None or self.px4_2_pos is None:
            return

        dx = self.px4_1_pos.x - self.px4_2_pos.x
        dy = self.px4_1_pos.y - self.px4_2_pos.y
        dz = self.px4_1_pos.z - self.px4_2_pos.z
        self.last_distance_m = math.sqrt(dx * dx + dy * dy + dz * dz)

        print(f"PX4_1 <-> PX4_2 mesafe (m): {self.last_distance_m:.2f}")
        
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

            # Takip hedefi: PX4_2 konumu (mevcutsa)
            if self.px4_2_pos is not None:
                trajectory_msg.position[0] = self.px4_2_pos.x
                trajectory_msg.position[1] = self.px4_2_pos.y
                trajectory_msg.position[2] = self.px4_2_pos.z
            else:
                trajectory_msg.position[0] = math.nan
                trajectory_msg.position[1] = math.nan
                trajectory_msg.position[2] = math.nan
            
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

__contact__ = "jalim@ethz.ch"

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import VehicleStatus
from px4_msgs.msg import VehicleLocalPosition
from px4_msgs.msg import FixedWingLongitudinalSetpoint
import math
from dataclasses import dataclass


@dataclass
class LocalPosition:
    x: float
    y: float
    z: float

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
        self.status_sub_v1 = self.create_subscription(
            VehicleStatus,
            '/px4_1/fmu/out/vehicle_status_v1',
            self.vehicle_status_callback,
            qos_profile_sub)

        # İki uçak için konum abonelikleri
        self.pos_sub_px4_2 = self.create_subscription(
            VehicleLocalPosition,
            '/px4_2/fmu/out/vehicle_local_position_v1',
            lambda msg: self.konum_verisi('px4_2', msg),
            qos_profile_sub)
        self.pos_sub_px4_1_v1 = self.create_subscription(
            VehicleLocalPosition,
            '/px4_1/fmu/out/vehicle_local_position_v1',
            lambda msg: self.konum_verisi('px4_1', msg),
            qos_profile_sub)

        # Bazı kurulumlarda namespace'siz topic yayınlanıyor olabilir; px4_1 için yedek abonelik
        self.pos_sub_fallback = self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position',
            lambda msg: self.konum_verisi('px4_1', msg),
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

        # Son bilinen konumlar
        self.px4_1_pos: LocalPosition | None = None
        self.px4_2_pos: LocalPosition | None = None
        self.last_distance_m: float | None = None
    
    def vehicle_status_callback(self, msg):        
        # Not: NED/ENU dönüşümü gerektiren kurulumlarda eksen dönüşümü eklenmelidir.
        print("NAV_STATUS: ", msg.nav_state)
        print("NAV_STATUS: ", msg.nav_state)
        print("  - offboard status: ", VehicleStatus.NAVIGATION_STATE_OFFBOARD)
        self.nav_state = msg.nav_state
        self.arming_state = msg.arming_state

    def konum_verisi(self, vehicle: str, msg_or_x, y=None, z=None):
        """İki uçağın konumunu güncelle ve aralarındaki mesafeyi hesapla.

        - Subscription callback: msg_or_x bir VehicleLocalPosition mesajıdır.
        - Manuel çağrı: (x, y, z) sayısal değerleri verilebilir.
        """
        if y is None and z is None:
            msg = msg_or_x
            pos = LocalPosition(float(msg.x), float(msg.y), float(msg.z))
        else:
            pos = LocalPosition(float(msg_or_x), float(y), float(z))

        if vehicle == 'px4_1':
            self.px4_1_pos = pos
        elif vehicle == 'px4_2':
            self.px4_2_pos = pos
        else:
            return

        # İki konum da mevcutsa mesafe hesapla
        self._hesapla_ve_yazdir_mesafe()

    def _hesapla_ve_yazdir_mesafe(self):
        if self.px4_1_pos is None or self.px4_2_pos is None:
            return

        dx = self.px4_1_pos.x - self.px4_2_pos.x
        dy = self.px4_1_pos.y - self.px4_2_pos.y
        dz = self.px4_1_pos.z - self.px4_2_pos.z
        self.last_distance_m = math.sqrt(dx * dx + dy * dy + dz * dz)

        print(f"PX4_1 <-> PX4_2 mesafe (m): {self.last_distance_m:.2f}")
        
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
            trajectory_msg.timestamp = now_us

            # Takip hedefi: PX4_2 konumu (mevcutsa)
            if self.px4_2_pos is not None:
                trajectory_msg.position[0] = self.px4_2_pos.x
                trajectory_msg.position[1] = self.px4_2_pos.y
                trajectory_msg.position[2] = self.px4_2_pos.z
            else:
                trajectory_msg.position[0] = math.nan
                trajectory_msg.position[1] = math.nan
                trajectory_msg.position[2] = math.nan
            
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

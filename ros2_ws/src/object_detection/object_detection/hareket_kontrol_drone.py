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
from geometry_msgs.msg import Point

import math
import time
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

        # YOLO distance verilerini subscribe et
        self.distance_sub = self.create_subscription(
            Point,
            '/yolo/target_distance',
            self.distance_callback,
            qos_profile_sub)
        
        self.status_sub = self.create_subscription(
            VehicleStatus,
            'fmu/out/vehicle_status',
            self.vehicle_status_callback,
            qos_profile_sub)
        self.status_sub = self.create_subscription(
            VehicleStatus,
            'fmu/out/vehicle_status_v1',
            self.vehicle_status_callback,
            qos_profile_sub)
        self.publisher_offboard_mode = self.create_publisher(OffboardControlMode, 'fmu/in/offboard_control_mode', qos_profile_pub)
        self.publisher_trajectory = self.create_publisher(TrajectorySetpoint, 'fmu/in/trajectory_setpoint', qos_profile_pub)
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
        
        # YOLO'dan gelen veriler
        self.distance_x = 0.0
        self.distance_y = 0.0
        self.last_detection_time = time.time()
        self.detection_timeout = 0.5  # 0.5 saniye veri gelmezse sıfırla
        
    def distance_callback(self, msg):
        """YOLO'dan gelen merkeze uzaklık verilerini al"""
        self.distance_x = msg.x
        self.distance_y = msg.y
        self.last_detection_time = time.time()
        self.get_logger().info(f'YOLO verisi alındı: dx={msg.x:.1f}, dy={msg.y:.1f}, conf={msg.z:.2f}')

    def vehicle_status_callback(self, msg):
        # TODO: handle NED->ENU transformation
        print("NAV_STATUS: ", msg.nav_state)
        print("  - offboard status: ", VehicleStatus.NAVIGATION_STATE_OFFBOARD)
        self.nav_state = msg.nav_state
        self.arming_state = msg.arming_state

    def cmdloop_callback(self):
        # Publish offboard control modes
        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        offboard_msg.position=False
        offboard_msg.velocity=True
        offboard_msg.acceleration=False
        self.publisher_offboard_mode.publish(offboard_msg)
        
        # Veri gelme süresini kontrol et
        time_since_detection = time.time() - self.last_detection_time
        
        if (self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD and self.arming_state == VehicleStatus.ARMING_STATE_ARMED):

            trajectory_msg = TrajectorySetpoint()
            # Position KULLANMIYORUZ
            trajectory_msg.position[0] = math.nan
            trajectory_msg.position[1] = math.nan
            trajectory_msg.position[2] = math.nan

            # Velocity KULLANIYORUZ - YOLO verilerine göre ayarla
            if time_since_detection > self.detection_timeout:
                # Veri gelmiyorsa tüm velocity değerleri 0
                trajectory_msg.velocity[0] = 0.0
                trajectory_msg.velocity[1] = 0.0
                trajectory_msg.velocity[2] = 0.0
                self.get_logger().warn('YOLO verisi gelmiyor - Duruluyor')
            else:
                # YOLO verilerine göre velocity hesapla
                # distance_x: pozitif = hedef sağda, negatif = hedef solda
                # distance_y: pozitif = hedef aşağıda, negatif = hedef yukarıda
                
                # Gain değerleri (hız kontrolü için)
                kp_x = 0.005  # X ekseni kazancı
                kp_y = 0.005  # Y ekseni kazancı
                
                # Velocity hesapla (basit orantılı kontrol)
                vel_x = -kp_x * self.distance_y # Sabit ileri hız
                vel_y = kp_y * self.distance_x  # Yatay (sağ/sol) hareket - NED koordinat sistemi
                vel_z = 0  # Dikey (yukarı/aşağı) hareket - NED koordinat sistemi
                
                # Hız limitleri
                max_vel_lateral = 10.0
                max_vel_vertical = 10.0
                
                vel_y = max(-max_vel_lateral, min(max_vel_lateral, vel_y))
                vel_z = max(-max_vel_vertical, min(max_vel_vertical, vel_x))
                
                trajectory_msg.velocity[0] = vel_x    # İleri
                trajectory_msg.velocity[1] = vel_y    # Sağ/Sol
                trajectory_msg.velocity[2] = vel_z    # Yukarı/Aşağı
                
                self.get_logger().info(f'Velocity: vx={vel_x:.2f}, vy={vel_y:.2f}, vz={vel_z:.2f}')

            
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
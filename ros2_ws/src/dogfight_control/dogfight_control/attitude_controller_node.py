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
from px4_msgs.msg import VehicleAttitudeSetpoint
from px4_msgs.msg import VehicleLocalPosition, VehicleGlobalPosition
from px4_msgs.msg import GotoSetpoint
import math

class OffboardControl(Node):
    
    def __init__(self):
        super().__init__('minimal_publisher')

                # QoS profiles
        qos_profile_pub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        qos_profile_sub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.my_lat = 0.0
        self.my_lon = 0.0
        self.my_alt = 0.0
        self.my_alt_rel = 0.0
        self.target_lat = 47.401628  
        self.target_lon = 8.545730
        self.target_alt = None   # None → px4_2 verisi henüz gelmedi
        self.target_alt_rel = None
        self.my_heading = 0.0
        # --- Yaw PID (dönüş) ---
        self.yaw_pid_integral  = 0.0
        self.yaw_pid_prev_error = 0.0
        self.YAW_KP = 0.12
        self.YAW_KI = 0.003
        self.YAW_KD = 0.01
        self.MAX_ROLL = math.radians(60)  # Maksimum bank açısı ±60°

        # --- Yükseklik PID (pitch kontrolü) ---
        self.alt_pid_integral   = 0.0
        self.alt_pid_prev_error = 0.0
        self.ALT_KP = 0.04    # pitch derece / metre hata
        self.ALT_KI = 0.002
        self.ALT_KD = 0.01
        self.MAX_PITCH_DEG = 20.0   # ±20° pitch sınırı
        # --- Mesafe PID (itki kontrolu) ---
        self.dist_pid_integral = 0.0
        self.dist_pid_prev_error = 0.0
        self.DIST_KP = 0.009
        self.DIST_KI = 0.00005
        self.DIST_KD = 0.03
        self.DIST_TARGET_M = 5.0
        self.MIN_THRUST = 0.30
        self.MAX_THRUST = 0.69
        self.NOMINAL_THRUST = 0.55
        self.THRUST_SMOOTH_ALPHA = 0.22
        self.last_thrust = self.NOMINAL_THRUST
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
        self.pos_sub_px4_2 = self.create_subscription(
            VehicleLocalPosition,
            '/px4_1/fmu/out/vehicle_local_position_v1',
            self.vehicle_local_pos_callback,
            qos_profile_sub)
        self.target_local_pos_sub = self.create_subscription(
            VehicleLocalPosition,
            '/px4_2/fmu/out/vehicle_local_position_v1',
            self.target_local_pos_callback,
            qos_profile_sub)
        self.pos_sub_px4_1 = self.create_subscription(
            VehicleGlobalPosition,
            '/px4_1/fmu/out/vehicle_global_position',
            self.vehicle_global_pos_callback,
            qos_profile_sub)
        self.target_pos_sub = self.create_subscription(
            VehicleGlobalPosition,
            '/px4_2/fmu/out/vehicle_global_position',
            self.target_global_pos_callback,
            qos_profile_sub)
        self.attitude_pub = self.create_publisher(VehicleAttitudeSetpoint, '/px4_1/fmu/in/vehicle_attitude_setpoint_v1', qos_profile_pub)
        self.publisher_offboard_mode = self.create_publisher(OffboardControlMode, '/px4_1/fmu/in/offboard_control_mode', qos_profile_pub)
        self.publisher_trajectory = self.create_publisher(TrajectorySetpoint, '/px4_1/fmu/in/trajectory_setpoint', qos_profile_pub)
        self.publisher_goto = self.create_publisher(GotoSetpoint, '/px4_1/fmu/in/goto_setpoint', qos_profile_pub)
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
        self.hiz=0.5
    def vehicle_global_pos_callback(self, msg):
        self.my_lat = msg.lat
        self.my_lon = msg.lon
        self.my_alt = msg.alt
        #print(f"Current position: lat={self.my_lat:.6f}, lon={self.my_lon:.6f}")
    def target_global_pos_callback(self, msg):
        self.target_lat = msg.lat
        self.target_lon = msg.lon
        self.target_alt = msg.alt
        #print(f"Target position: lat={self.target_lat:.6f}, lon={self.target_lon:.6f}")
    def vehicle_local_pos_callback(self, msg):
        self.my_alt_rel = msg.z
        self.my_heading = math.degrees(msg.heading)
        if self.my_heading < 0:
            self.my_heading += 360

    def target_local_pos_callback(self, msg):
        self.target_alt_rel = msg.z
       
    def pid_compute(self, error, integral, prev_error, kp, ki, kd, dt,
                     integral_limit=None, output_limit=None):
        """
        Genel PID hesaplayıcı.
        Döner: (output, yeni_integral, yeni_prev_error)
        """
        integral += error * dt
        if integral_limit is not None:
            integral = max(-integral_limit, min(integral_limit, integral))
        derivative = (error - prev_error) / dt
        output = kp * error + ki * integral + kd * derivative
        if output_limit is not None:
            output = max(-output_limit, min(output_limit, output))
        return output, integral, error

    def hesapla(self, lat1, lon1, lat2, lon2):
        # Dereceyi radyana çevir
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

        # --- MESAFe (Haversine) ---
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        mesafe_km = 6371 * 2 * math.asin(math.sqrt(a))

        # --- BEARING (hangi yöne uçmalısın) ---
        x = math.sin(dlon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        bearing = math.degrees(math.atan2(x, y))
        bearing = (bearing + 360) % 360  # 0-360 arası normalize et

        return mesafe_km, bearing   
    
    def vehicle_status_callback(self, msg):
        # TODO: handle NED->ENU transformation
        print("NAV_STATUS: ", msg.nav_state)
        print("  - offboard status: ", VehicleStatus.NAVIGATION_STATE_OFFBOARD)
        self.nav_state = msg.nav_state
        self.arming_state = msg.arming_state

    def euler_to_quaternion(self, roll, pitch, yaw):
        """
        roll, pitch, yaw -> radian
        PX4 quaternion sırası: [w, x, y, z]
        """
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy


        return [w, x, y, z]
    
    def cmdloop_callback(self):
        # Publish offboard control modes
        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        offboard_msg.position=False
        offboard_msg.velocity=False
        offboard_msg.acceleration=False
        offboard_msg.attitude=True
        offboard_msg.direct_actuator=False
        self.publisher_offboard_mode.publish(offboard_msg)
        if (self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD and self.arming_state == VehicleStatus.ARMING_STATE_ARMED):
            
            
        
            msg = VehicleAttitudeSetpoint()
            mesafe_km, target_bearing_deg = self.hesapla(
                self.my_lat, self.my_lon, self.target_lat, self.target_lon)

            # --- YAW PID: hedefe dön ---
            hedef_yaw = target_bearing_deg - self.my_heading
            hedef_yaw = (hedef_yaw + 180) % 360 - 180  # shortest path [-180, 180]

            """roll_deg, self.yaw_pid_integral, self.yaw_pid_prev_error = self.pid_compute(
                error          = hedef_yaw,
                integral       = self.yaw_pid_integral,
                prev_error     = self.yaw_pid_prev_error,
                kp             = self.YAW_KP,
                ki             = self.YAW_KI,
                kd             = self.YAW_KD,
                dt             = self.dt,
                integral_limit = 30.0,
                output_limit   = None,
            )""" # pid ile yaw kontrolü --- IGNORE ---
            roll_cmd = max(-self.MAX_ROLL, min(self.MAX_ROLL, math.radians(hedef_yaw)))

            roll  = roll_cmd

            # --- YÜKSEKLİK PID: pitch ile yüksekliği eşitle ---
            if self.target_alt_rel is not None:
                # NED z ekseni: pozitif aşağı. Hata pozitifse hedeften alçaktayız.
                alt_error = self.my_alt_rel - self.target_alt_rel
                pitch_deg, self.alt_pid_integral, self.alt_pid_prev_error = self.pid_compute(
                    error          = alt_error,
                    integral       = self.alt_pid_integral,
                    prev_error     = self.alt_pid_prev_error,
                    kp             = self.ALT_KP,
                    ki             = self.ALT_KI,
                    kd             = self.ALT_KD,
                    dt             = self.dt,
                    integral_limit = 20.0,
                    output_limit   = self.MAX_PITCH_DEG,
                )
                pitch = math.radians(pitch_deg)
            else:
                alt_error = 0.0
                pitch = math.radians(0.0)  # hedef irtifası bilinene kadar düz uç

            
            # Throttle: mesafe PID ile 8m hedefe kilitlen
            distance_error = mesafe_km * 1000.0 - self.DIST_TARGET_M
            thrust_delta, self.dist_pid_integral, self.dist_pid_prev_error = self.pid_compute(
                error          = distance_error,
                integral       = self.dist_pid_integral,
                prev_error     = self.dist_pid_prev_error,
                kp             = self.DIST_KP,
                ki             = self.DIST_KI,
                kd             = self.DIST_KD,
                dt             = self.dt,
                integral_limit = 100.0,
                output_limit   = 1.0,
            )
            thrust_cmd = max(self.MIN_THRUST, min(self.MAX_THRUST, self.NOMINAL_THRUST + thrust_delta))
            # Low-pass filter to reduce thrust jitter
            thrust = (self.THRUST_SMOOTH_ALPHA * thrust_cmd
                      + (1.0 - self.THRUST_SMOOTH_ALPHA) * self.last_thrust)
            self.last_thrust = thrust
            
            
            
            # Yaw: mevcut heading ile quaternion oluştur — aksi halde roll+yaw
            # çarpımı sahte pitch-up kuplajı yaratır
            yaw   = math.radians(0.0)

            tgt_alt_str = f"{self.target_alt_rel:.1f}m" if self.target_alt_rel is not None else "?"
            print(f"Mesafe: {mesafe_km*1000:.0f}m  "
                f"Hata yön: {hedef_yaw:.1f}°  Roll: {math.degrees(roll_cmd):.1f}°  "
                f"İrtifa(z): bende={self.my_alt_rel:.1f}m hedef={tgt_alt_str} hata={alt_error:+.1f}m  "
                f"Pitch: {math.degrees(pitch):.1f}°"
                f"  Thrust: {thrust:.2f}")
            


            q = self.euler_to_quaternion(roll, pitch*10, yaw)
            msg.q_d = q
            msg.thrust_body = [thrust, 0.0, 0.0]
            msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
            self.attitude_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    offboard_control = OffboardControl()

    rclpy.spin(offboard_control)

    offboard_control.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
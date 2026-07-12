#!/usr/bin/env python3
############################################################################
#
#   px4_status_monitor — PX4 durum monitörü (eski deneme.py'nin düzenlenmişi).
#
#   Konum, hız, yön, açısal hız, hava hızı ve arm/nav durumunu loglar.
#   Namespace parametrelidir; v1.16 versiyonlu topic varyantlarına da
#   abone olur. Yalnızca gözlem — hiçbir şey yayınlamaz.
#
#   Kullanım:
#     ros2 run dogfight_control px4_status_monitor              # takipçi (/fmu)
#     ros2 run dogfight_control px4_status_monitor --ros-args -p vehicle_ns:=/px4_1
#
############################################################################

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from px4_msgs.msg import (VehicleLocalPosition, VehicleGlobalPosition,
                          VehicleAttitude, VehicleAngularVelocity,
                          AirspeedValidated, VehicleStatus)


class PX4StatusMonitor(Node):

    def __init__(self):
        super().__init__('px4_status_monitor')

        self.declare_parameter('vehicle_ns', '')     # '' = /fmu
        self.declare_parameter('log_period', 1.0)    # her veri türü için [s]
        ns = str(self.get_parameter('vehicle_ns').value).rstrip('/')
        self.log_period = float(self.get_parameter('log_period').value)

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)

        def sub_multi(msg_type, base, cb):
            for topic in (base, base + '_v1', base + '_v2', base + '_v4'):
                self.create_subscription(msg_type, topic, cb, qos)

        sub_multi(VehicleLocalPosition, ns + '/fmu/out/vehicle_local_position',
                  self.local_pos_cb)
        sub_multi(VehicleGlobalPosition, ns + '/fmu/out/vehicle_global_position',
                  self.global_pos_cb)
        sub_multi(VehicleAttitude, ns + '/fmu/out/vehicle_attitude',
                  self.attitude_cb)
        sub_multi(VehicleAngularVelocity, ns + '/fmu/out/vehicle_angular_velocity',
                  self.ang_vel_cb)
        sub_multi(AirspeedValidated, ns + '/fmu/out/airspeed_validated',
                  self.airspeed_cb)
        sub_multi(VehicleStatus, ns + '/fmu/out/vehicle_status', self.status_cb)

        self.get_logger().info(f"PX4 monitör başladı | araç='{ns or '/fmu'}'")

    def local_pos_cb(self, msg):
        altitude = -msg.z  # NED'de z negatif = yukarı
        speed_2d = math.hypot(msg.vx, msg.vy)
        self.get_logger().info(
            f'Konum NED: x={msg.x:.1f}m  y={msg.y:.1f}m  alt={altitude:.1f}m | '
            f'Hız: vx={msg.vx:.1f}  vy={msg.vy:.1f}  vz={msg.vz:.1f} m/s | '
            f'Yatay hız: {speed_2d:.1f} m/s',
            throttle_duration_sec=self.log_period)

    def global_pos_cb(self, msg):
        self.get_logger().info(
            f'GPS: lat={msg.lat:.6f}°  lon={msg.lon:.6f}°  alt={msg.alt:.1f}m MSL',
            throttle_duration_sec=self.log_period)

    def attitude_cb(self, msg):
        roll, pitch, yaw = self._q_to_euler(msg.q)
        self.get_logger().info(
            f'Attitude: roll={math.degrees(roll):.1f}°  '
            f'pitch={math.degrees(pitch):.1f}°  '
            f'yaw={math.degrees(yaw):.1f}°',
            throttle_duration_sec=self.log_period)

    def ang_vel_cb(self, msg):
        self.get_logger().info(
            f'Angular vel: p={math.degrees(msg.xyz[0]):.1f}°/s  '
            f'q={math.degrees(msg.xyz[1]):.1f}°/s  '
            f'r={math.degrees(msg.xyz[2]):.1f}°/s',
            throttle_duration_sec=self.log_period)

    def airspeed_cb(self, msg):
        self.get_logger().info(
            f'Hava hızı: CAS={msg.calibrated_airspeed_m_s:.1f} m/s  '
            f'TAS={msg.true_airspeed_m_s:.1f} m/s',
            throttle_duration_sec=self.log_period)

    def status_cb(self, msg):
        armed = (msg.arming_state == VehicleStatus.ARMING_STATE_ARMED)
        self.get_logger().info(
            f'Durum: armed={armed}  nav_state={msg.nav_state}  '
            f'failsafe={msg.failsafe}',
            throttle_duration_sec=self.log_period)

    @staticmethod
    def _q_to_euler(q):
        w, x, y, z = q[0], q[1], q[2], q[3]
        roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return roll, pitch, yaw


def main(args=None):
    rclpy.init(args=args)
    node = PX4StatusMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

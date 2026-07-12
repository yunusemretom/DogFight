#!/usr/bin/env python3
"""
PX4 Fixed-Wing Durum Monitörü
Konum, hız, yön ve hava hızı verilerini okur.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import math

from px4_msgs.msg import (
    VehicleLocalPosition,
    VehicleGlobalPosition,
    VehicleAttitude,
    VehicleAngularVelocity,
    AirspeedValidated,
    VehicleStatus,
)

class FWStatusMonitor(Node):
    def __init__(self):
        super().__init__('fw_status_monitor')

        # PX4 best-effort QoS
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscriber'lar
        self.create_subscription(VehicleLocalPosition,
            '/fmu/out/vehicle_local_position',
            self.local_pos_cb, qos)

        self.create_subscription(VehicleGlobalPosition,
            '/fmu/out/vehicle_global_position',
            self.global_pos_cb, qos)

        self.create_subscription(VehicleAttitude,
            '/fmu/out/vehicle_attitude',
            self.attitude_cb, qos)

        self.create_subscription(VehicleAngularVelocity,
            '/fmu/out/vehicle_angular_velocity',
            self.ang_vel_cb, qos)

        self.create_subscription(AirspeedValidated,
            '/fmu/out/airspeed_validated',
            self.airspeed_cb, qos)

        self.create_subscription(VehicleStatus,
            '/fmu/out/vehicle_status',
            self.status_cb, qos)

    def local_pos_cb(self, msg):
        altitude = -msg.z  # NED'de z negatif = yukarı
        speed_2d = math.sqrt(msg.vx**2 + msg.vy**2)
        self.get_logger().info(
            f'Konum NED: x={msg.x:.1f}m  y={msg.y:.1f}m  alt={altitude:.1f}m | '
            f'Hız: vx={msg.vx:.1f}  vy={msg.vy:.1f}  vz={msg.vz:.1f} m/s | '
            f'Yatay hız: {speed_2d:.1f} m/s'
        )

    def global_pos_cb(self, msg):
        self.get_logger().info(
            f'GPS: lat={msg.lat:.6f}°  lon={msg.lon:.6f}°  alt={msg.alt:.1f}m MSL'
        )

    def attitude_cb(self, msg):
        q = msg.q
        roll, pitch, yaw = self._q_to_euler(q)
        self.get_logger().info(
            f'Attitude: roll={math.degrees(roll):.1f}°  '
            f'pitch={math.degrees(pitch):.1f}°  '
            f'yaw={math.degrees(yaw):.1f}°'
        )

    def ang_vel_cb(self, msg):
        self.get_logger().info(
            f'Angular vel: p={math.degrees(msg.xyz[0]):.1f}°/s  '
            f'q={math.degrees(msg.xyz[1]):.1f}°/s  '
            f'r={math.degrees(msg.xyz[2]):.1f}°/s'
        )

    def airspeed_cb(self, msg):
        self.get_logger().info(
            f'Hava hızı: CAS={msg.calibrated_airspeed_m_s:.1f} m/s  '
            f'TAS={msg.true_airspeed_m_s:.1f} m/s  '
            f'Geçerli={msg.airspeed_sensor_measurement_valid}'
        )

    def status_cb(self, msg):
        armed = (msg.arming_state == 2)
        self.get_logger().info(
            f'Durum: armed={armed}  nav_state={msg.nav_state}  '
            f'vehicle_type={msg.vehicle_type}  failsafe={msg.failsafe}'
        )

    @staticmethod
    def _q_to_euler(q):
        w, x, y, z = q[0], q[1], q[2], q[3]
        roll  = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        pitch = math.asin(max(-1, min(1, 2*(w*y - z*x))))
        yaw   = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        return roll, pitch, yaw


def main():
    rclpy.init()
    node = FWStatusMonitor()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
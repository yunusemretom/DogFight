#!/usr/bin/env python3
############################################################################
#
#   attitude_setpoint_node — VehicleAttitudeSetpoint offboard test node'u.
#
#   Amaç: PX4 v1.16'da sabit kanat için attitude offboard arayüzünü
#   (OffboardControlMode.attitude=true + VehicleAttitudeSetpoint) hedef
#   takibiyle test etmek.
#
#   Kontrol yasası:
#     roll   = kp_roll * heading_hatası          (hedefe dön)
#     pitch  = PID(irtifa_hatası)                (hedef irtifasına eşitlen;
#              irtifalar GLOBAL pozisyondan — local z'ler çapraz-çerçeve!)
#     thrust = nominal + PID(mesafe - standoff)  (mesafeyi koru), low-pass'lı
#     yaw    = mevcut heading                    (yaw=0 sabitlemek roll ile
#              sahte kuplaj yaratır — eski node'daki hata)
#
#   ⚠ UYARI: attitude offboard TECS'i devre dışı bırakır — stall/irtifa
#   koruması YOKTUR. Yalnızca SITL'de ve irtifada test edin. Üretim takibi
#   için l1_pursuit_node kullanılır.
#
############################################################################

import math

import rclpy
from px4_msgs.msg import VehicleAttitudeSetpoint

from dogfight_control.offboard_base import OffboardTestBase
from dogfight_control.control_math import (PID, clamp, euler_to_quaternion,
                                           wrap_pi)


class AttitudeSetpointNode(OffboardTestBase):

    def __init__(self):
        super().__init__('attitude_setpoint_node', ocm_flags={'attitude': True})

        p = self.declare_parameter
        p('standoff_distance', 20.0)   # korunacak takip mesafesi [m]
        p('altitude_offset', 0.0)      # hedef irtifasına eklenecek fark [m]
        p('kp_roll', 1.0)              # heading hatası → roll kazancı [rad/rad]
        p('max_roll_deg', 45.0)
        p('kp_pitch', 0.02)            # irtifa hatası → pitch [rad/m]
        p('ki_pitch', 0.002)
        p('kd_pitch', 0.01)
        p('max_pitch_deg', 20.0)
        p('kp_thrust', 0.01)           # mesafe hatası → thrust [1/m]
        p('ki_thrust', 0.0005)
        p('kd_thrust', 0.02)
        p('nominal_thrust', 0.55)
        p('min_thrust', 0.25)
        p('max_thrust', 0.85)
        p('thrust_lp_alpha', 0.2)      # thrust low-pass (1 = filtresiz)

        gp = lambda n: self.get_parameter(n).value
        self.standoff = float(gp('standoff_distance'))
        self.alt_offset = float(gp('altitude_offset'))
        self.kp_roll = float(gp('kp_roll'))
        self.max_roll = math.radians(float(gp('max_roll_deg')))
        max_pitch = math.radians(float(gp('max_pitch_deg')))
        self.pid_pitch = PID(float(gp('kp_pitch')), float(gp('ki_pitch')),
                             float(gp('kd_pitch')),
                             i_limit=20.0, out_limit=max_pitch)
        self.pid_thrust = PID(float(gp('kp_thrust')), float(gp('ki_thrust')),
                              float(gp('kd_thrust')),
                              i_limit=100.0, out_limit=0.4)
        self.nominal_thrust = float(gp('nominal_thrust'))
        self.min_thrust = float(gp('min_thrust'))
        self.max_thrust = float(gp('max_thrust'))
        self.lp_alpha = float(gp('thrust_lp_alpha'))
        self.last_thrust = self.nominal_thrust

        # v1.16'da topic versiyonlu; her iki varyanta da yayınla
        f = self.follower_ns
        self.pub_att = [
            self.create_publisher(VehicleAttitudeSetpoint,
                                  f + '/fmu/in/vehicle_attitude_setpoint',
                                  self.qos),
            self.create_publisher(VehicleAttitudeSetpoint,
                                  f + '/fmu/in/vehicle_attitude_setpoint_v1',
                                  self.qos)]

        self.get_logger().info(
            f"AttitudeSetpoint testi | takipçi='{f or '/'}' "
            f"hedef='{self.target_ns}' standoff={self.standoff}m")

    # ── Setpoint yayını ───────────────────────────────────────────────── #

    def _publish_attitude(self, ts, roll, pitch, yaw, thrust):
        msg = VehicleAttitudeSetpoint()
        msg.timestamp = ts
        msg.q_d = [float(v) for v in euler_to_quaternion(roll, pitch, yaw)]
        # Sabit kanat: thrust_body[0] = itki, diğerleri 0
        msg.thrust_body = [float(thrust), 0.0, 0.0]
        for pub in self.pub_att:
            pub.publish(msg)

    def publish_idle(self, ts):
        # Kanatlar düz, mevcut heading, nominal itki — güvenli bekleme
        self.pid_pitch.reset()
        self.pid_thrust.reset()
        self._publish_attitude(ts, 0.0, 0.0, self.my_heading(),
                               self.nominal_thrust)

    def publish_active(self, ts):
        rel_n, rel_e, rel_d = self.rel_ned()
        dist_xy = math.hypot(rel_n, rel_e)
        heading = self.my_heading()

        # Roll: hedefe dön
        bearing = math.atan2(rel_e, rel_n)
        hdg_err = wrap_pi(bearing - heading)
        roll = clamp(self.kp_roll * hdg_err, -self.max_roll, self.max_roll)

        # Pitch: irtifa hatası (global alt, yukarı pozitif hata = tırman)
        alt_err = (float(self.tgt_glob.alt) + self.alt_offset
                   - float(self.my_glob.alt))
        pitch = self.pid_pitch.update(alt_err, self.dt)

        # Thrust: mesafeyi koru (uzaksa hızlan)
        dist_err = dist_xy - self.standoff
        thrust_cmd = clamp(self.nominal_thrust
                           + self.pid_thrust.update(dist_err, self.dt),
                           self.min_thrust, self.max_thrust)
        thrust = (self.lp_alpha * thrust_cmd
                  + (1.0 - self.lp_alpha) * self.last_thrust)
        self.last_thrust = thrust

        self._publish_attitude(ts, roll, pitch, heading, thrust)

        self.get_logger().info(
            f'[ATT] mesafe={dist_xy:.1f}m (hedef {self.standoff:.0f}m)  '
            f'hdg_err={math.degrees(hdg_err):+.0f}°  '
            f'roll={math.degrees(roll):+.1f}°  '
            f'pitch={math.degrees(pitch):+.1f}° (alt_err={alt_err:+.1f}m)  '
            f'thrust={thrust:.2f}',
            throttle_duration_sec=1.0)


def main(args=None):
    rclpy.init(args=args)
    node = AttitudeSetpointNode()
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

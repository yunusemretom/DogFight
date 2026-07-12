#!/usr/bin/env python3
############################################################################
#
#   trajectory_velocity_node — TrajectorySetpoint.velocity offboard testi.
#
#   Amaç: PX4 v1.16'da sabit kanatta velocity trajectory setpoint'in
#   davranışını kontrollü gözlemlemek.
#
#   ⚠ SINIRLAMA (PROJE_OZETI tuzak #6): TrajectorySetpoint yayınlanınca
#   PX4'ün FixedWingModeManager'ı devreye girer ve kendi guidance'ını
#   çalıştırır — sabit kanatta velocity setpoint desteği sınırlıdır ve
#   davranış multicopter'daki gibi doğrudan değildir. Bu node üretim için
#   DEĞİL, arayüzün ne yaptığını görmek içindir. Üretim: l1_pursuit_node.
#
#   Kontrol yasası (takipçi merkezli NED):
#     fp        = follow_point(hedef, standoff)     [yay-üzerinde]
#     v_xy      = hedef_hızı + kp_approach * fp     → [min,max] hıza kırpılır
#     vz        = -kp_alt * irtifa_hatası(yukarı+)  → ±max_vz
#     yaw       = hız vektörünün yönü
#
############################################################################

import math

import rclpy
from px4_msgs.msg import TrajectorySetpoint

from dogfight_control.offboard_base import OffboardTestBase
from dogfight_control.control_math import clamp, clamp_speed_xy, follow_point

NAN = float('nan')


class TrajectoryVelocityNode(OffboardTestBase):

    def __init__(self):
        super().__init__('trajectory_velocity_node',
                         ocm_flags={'velocity': True})

        p = self.declare_parameter
        p('standoff_distance', 30.0)  # korunacak takip mesafesi [m]
        p('altitude_offset', 0.0)     # hedef irtifasına eklenecek fark [m]
        p('kp_approach', 0.4)         # takip noktası hatası → hız [1/s]
        p('min_speed', 10.0)          # min XY hız (stall koruması) [m/s]
        p('max_speed', 25.0)          # max XY hız [m/s]
        p('kp_alt', 0.5)              # irtifa hatası → dikey hız [1/s]
        p('max_vz', 3.0)              # max dikey hız [m/s]

        gp = lambda n: self.get_parameter(n).value
        self.standoff = float(gp('standoff_distance'))
        self.alt_offset = float(gp('altitude_offset'))
        self.kp_approach = float(gp('kp_approach'))
        self.min_speed = float(gp('min_speed'))
        self.max_speed = float(gp('max_speed'))
        self.kp_alt = float(gp('kp_alt'))
        self.max_vz = float(gp('max_vz'))

        f = self.follower_ns
        self.pub_tsp = self.create_publisher(
            TrajectorySetpoint, f + '/fmu/in/trajectory_setpoint', self.qos)

        self.get_logger().info(
            f"TrajectoryVelocity testi | takipçi='{f or '/'}' "
            f"hedef='{self.target_ns}' standoff={self.standoff}m — "
            'DİKKAT: FW mode manager devrede, davranış gözlem amaçlı')

    def _publish_tsp(self, ts, vn, ve, vd, yaw):
        msg = TrajectorySetpoint()
        msg.timestamp = ts
        msg.position = [NAN, NAN, NAN]
        msg.velocity = [float(vn), float(ve), float(vd)]
        msg.acceleration = [NAN, NAN, NAN]
        msg.jerk = [NAN, NAN, NAN]
        msg.yaw = float(yaw)
        msg.yawspeed = NAN
        self.pub_tsp.publish(msg)

    def publish_idle(self, ts):
        # Sabit kanat duramaz: mevcut heading'de min hızda düz uçuş
        hdg = self.my_heading()
        self._publish_tsp(ts, self.min_speed * math.cos(hdg),
                          self.min_speed * math.sin(hdg), 0.0, hdg)

    def publish_active(self, ts):
        rel_n, rel_e, _ = self.rel_ned()
        dist_xy = math.hypot(rel_n, rel_e)
        tgt_vn, tgt_ve = float(self.tgt_loc.vx), float(self.tgt_loc.vy)

        fp_n, fp_e = follow_point(rel_n, rel_e, tgt_vn, tgt_ve,
                                  self.standoff, self.tgt_omega)

        raw_vn = tgt_vn + self.kp_approach * fp_n
        raw_ve = tgt_ve + self.kp_approach * fp_e
        vn, ve = clamp_speed_xy(raw_vn, raw_ve, self.min_speed, self.max_speed)

        # İrtifa: global alt farkından (yukarı pozitif hata → vz negatif/yukarı)
        alt_err = (float(self.tgt_glob.alt) + self.alt_offset
                   - float(self.my_glob.alt))
        vd = clamp(-self.kp_alt * alt_err, -self.max_vz, self.max_vz)

        yaw = math.atan2(ve, vn)
        self._publish_tsp(ts, vn, ve, vd, yaw)

        self.get_logger().info(
            f'[VEL] mesafe={dist_xy:.1f}m (hedef {self.standoff:.0f}m)  '
            f'cmd=({vn:+.1f}N,{ve:+.1f}E,{vd:+.1f}D)m/s  '
            f'alt_err={alt_err:+.1f}m  yaw={math.degrees(yaw):.0f}°',
            throttle_duration_sec=1.0)


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryVelocityNode()
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

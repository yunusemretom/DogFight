#!/usr/bin/env python3
############################################################################
#
#   trajectory_position_node — TrajectorySetpoint.position offboard testi.
#
#   Amaç: PX4 v1.16'da sabit kanatta position trajectory setpoint'in
#   davranışını kontrollü gözlemlemek.
#
#   ⚠ SINIRLAMA (PROJE_OZETI tuzak #6): TrajectorySetpoint yayınlanınca
#   PX4'ün FixedWingModeManager'ı devreye girer; position setpoint sabit
#   kanatta PX4'ün KENDİ guidance'ı ile (loiter benzeri) izlenir — harici
#   hassas takip mümkün değildir. Bu node arayüz davranışını görmek içindir.
#
#   Setpoint üretimi:
#     - Hedefin konumu global pozisyon farkından takipçinin LOCAL NED
#       çerçevesine taşınır (eski node'daki çapraz-çerçeve hatasının
#       düzeltmesi: başka aracın local x/y/z'si asla doğrudan kullanılmaz).
#     - Nokta, hedefin yolunun standoff kadar gerisine konur.
#     - z, irtifa farkından türetilir; yaw/velocity NaN bırakılır.
#
############################################################################

import math

import rclpy
from px4_msgs.msg import TrajectorySetpoint

from dogfight_control.offboard_base import OffboardTestBase
from dogfight_control.control_math import follow_point

NAN = float('nan')


class TrajectoryPositionNode(OffboardTestBase):

    def __init__(self):
        super().__init__('trajectory_position_node',
                         ocm_flags={'position': True})

        p = self.declare_parameter
        p('standoff_distance', 30.0)  # hedefin gerisindeki nokta [m]
        p('altitude_offset', 0.0)     # hedef irtifasına eklenecek fark [m]

        gp = lambda n: self.get_parameter(n).value
        self.standoff = float(gp('standoff_distance'))
        self.alt_offset = float(gp('altitude_offset'))

        f = self.follower_ns
        self.pub_tsp = self.create_publisher(
            TrajectorySetpoint, f + '/fmu/in/trajectory_setpoint', self.qos)

        self.get_logger().info(
            f"TrajectoryPosition testi | takipçi='{f or '/'}' "
            f"hedef='{self.target_ns}' standoff={self.standoff}m — "
            'DİKKAT: FW mode manager devrede, davranış gözlem amaçlı')

    def _publish_tsp(self, ts, pn, pe, pd):
        msg = TrajectorySetpoint()
        msg.timestamp = ts
        msg.position = [float(pn), float(pe), float(pd)]
        msg.velocity = [NAN, NAN, NAN]
        msg.acceleration = [NAN, NAN, NAN]
        msg.jerk = [NAN, NAN, NAN]
        msg.yaw = NAN
        msg.yawspeed = NAN
        self.pub_tsp.publish(msg)

    def publish_idle(self, ts):
        # Mevcut konumu (yoksa origin'i) tut — offboard kabulü için akış şart
        if self.my_loc is not None:
            self._publish_tsp(ts, self.my_loc.x, self.my_loc.y, self.my_loc.z)
        else:
            self._publish_tsp(ts, 0.0, 0.0, -30.0)

    def publish_active(self, ts):
        rel_n, rel_e, _ = self.rel_ned()
        dist_xy = math.hypot(rel_n, rel_e)
        tgt_vn, tgt_ve = float(self.tgt_loc.vx), float(self.tgt_loc.vy)

        fp_n, fp_e = follow_point(rel_n, rel_e, tgt_vn, tgt_ve,
                                  self.standoff, self.tgt_omega)

        # Takipçi local NED'ine taşı: setpoint = kendi konumum + bağıl vektör
        pn = float(self.my_loc.x) + fp_n
        pe = float(self.my_loc.y) + fp_e
        # z: irtifa hatası kadar in/çık (yukarı pozitif hata → z azalır)
        alt_err = (float(self.tgt_glob.alt) + self.alt_offset
                   - float(self.my_glob.alt))
        pd = float(self.my_loc.z) - alt_err

        self._publish_tsp(ts, pn, pe, pd)

        self.get_logger().info(
            f'[POS] mesafe={dist_xy:.1f}m (hedef {self.standoff:.0f}m)  '
            f'sp=({pn:.0f}N,{pe:.0f}E,{pd:.0f}D)  alt_err={alt_err:+.1f}m',
            throttle_duration_sec=1.0)


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryPositionNode()
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

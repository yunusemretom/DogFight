#!/usr/bin/env python3
############################################################################
#
#   visual_offboard_node — görsel tespit → offboard köprü test node'u.
#
#   Girdi: /yolo/target_distance (geometry_msgs/Point)
#     x = hedefin görüntü merkezine yatay uzaklığı [px]  (+ = sağda)
#     y = hedefin görüntü merkezine dikey uzaklığı [px]  (+ = aşağıda)
#     z = tespit güveni (confidence)
#
#   Çıkış: PX4 v1.16 sabit kanat offboard reçetesi (l1_pursuit_node ile aynı):
#     - OffboardControlMode.velocity=true, TrajectorySetpoint YAYINLANMAZ
#       → FixedWingModeManager pasif, setpoint'ler FwLateralLongitudinalControl'e.
#     - FixedWingLateralSetpoint.lateral_acceleration = kp_lateral * px_x
#       (hedef sağda → sağa yanal ivme)
#     - FixedWingLongitudinalSetpoint.height_rate = -kp_height * px_y
#       (hedef aşağıda → alçal), airspeed sabit cruise.
#
#   Eski node'daki düzeltilen hatalar:
#     - vel_z'ye yanlışlıkla vel_x yazılıyordu (kopyala-yapıştır).
#     - Dikey görüntü hatası ileri hıza bağlanmıştı (eksen karışıklığı).
#     - Timeout'ta (0,0,0) hız komutu = sabit kanatta stall; şimdi timeout'ta
#       kanat düz + irtifa koru + cruise hızla uçuş sürer (base HOLD durumu).
#     - NED hız komutu FW'de mode manager'ı tetikliyordu (tuzak #6).
#
############################################################################

import rclpy
from geometry_msgs.msg import Point
from px4_msgs.msg import FixedWingLateralSetpoint, FixedWingLongitudinalSetpoint

from dogfight_control.offboard_base import OffboardTestBase
from dogfight_control.control_math import clamp

NAN = float('nan')


class VisualOffboardNode(OffboardTestBase):

    def __init__(self):
        # Hedef başka araç değil görsel tespit — araç aboneliği yok
        super().__init__('visual_offboard_node',
                         ocm_flags={'velocity': True},
                         subscribe_target=False)

        p = self.declare_parameter
        p('detection_topic', '/yolo/target_distance')
        p('min_confidence', 0.3)       # bu güvenin altındaki tespitler atılır
        p('kp_lateral', 0.05)          # px → yanal ivme [m/s² / px]
        p('max_lateral_accel', 8.0)    # yanal ivme limiti [m/s²] (~39° roll)
        p('kp_height_rate', 0.02)      # px → tırmanma hızı [m/s / px]
        p('max_height_rate', 3.0)      # tırmanma/alçalma limiti [m/s]
        p('cruise_airspeed', 15.0)     # sabit airspeed komutu [m/s]

        gp = lambda n: self.get_parameter(n).value
        self.min_conf = float(gp('min_confidence'))
        self.kp_lat = float(gp('kp_lateral'))
        self.max_lat_acc = float(gp('max_lateral_accel'))
        self.kp_hr = float(gp('kp_height_rate'))
        self.max_hr = float(gp('max_height_rate'))
        self.cruise_aspd = float(gp('cruise_airspeed'))

        self.px_x = 0.0
        self.px_y = 0.0
        self._det_t = -1.0e9

        self.create_subscription(Point, str(gp('detection_topic')),
                                 self._detection_cb, 10)

        f = self.follower_ns
        self.pub_lat = self.create_publisher(
            FixedWingLateralSetpoint,
            f + '/fmu/in/fixed_wing_lateral_setpoint', self.qos)
        self.pub_lon = self.create_publisher(
            FixedWingLongitudinalSetpoint,
            f + '/fmu/in/fixed_wing_longitudinal_setpoint', self.qos)

        self.get_logger().info(
            f"VisualOffboard testi | takipçi='{f or '/'}' "
            f"tespit='{gp('detection_topic')}' aspd={self.cruise_aspd}m/s")

    # Hedef tazeliği görsel tespit zamanından gelir
    def target_fresh(self, now: float) -> bool:
        return (now - self._det_t) < self.tgt_timeout

    def _detection_cb(self, msg: Point):
        if msg.z < self.min_conf:
            return
        self.px_x = float(msg.x)
        self.px_y = float(msg.y)
        self._det_t = self._now()

    def _publish_fw(self, ts, lateral_accel, height_rate):
        lat = FixedWingLateralSetpoint()
        lat.timestamp = ts
        lat.course = NAN
        lat.airspeed_direction = NAN
        lat.lateral_acceleration = float(lateral_accel)
        self.pub_lat.publish(lat)

        lon = FixedWingLongitudinalSetpoint()
        lon.timestamp = ts
        lon.altitude = NAN            # NaN → irtifa height_rate ile sürülür
        lon.height_rate = float(height_rate)
        lon.equivalent_airspeed = float(self.cruise_aspd)
        lon.pitch_direct = NAN
        lon.throttle_direct = NAN
        self.pub_lon.publish(lon)

    def publish_idle(self, ts):
        # Tespit yok: kanat düz, irtifayı koru, cruise hızda uç (stall YOK)
        self._publish_fw(ts, 0.0, 0.0)

    def publish_active(self, ts):
        a_lat = clamp(self.kp_lat * self.px_x,
                      -self.max_lat_acc, self.max_lat_acc)
        # Hedef görüntüde aşağıda (+y) → alçal (height_rate yukarı +)
        h_rate = clamp(-self.kp_hr * self.px_y, -self.max_hr, self.max_hr)
        self._publish_fw(ts, a_lat, h_rate)

        self.get_logger().info(
            f'[VIS] px=({self.px_x:+.0f},{self.px_y:+.0f})  '
            f'a_lat={a_lat:+.2f}m/s²  h_rate={h_rate:+.2f}m/s',
            throttle_duration_sec=1.0)


def main(args=None):
    rclpy.init(args=args)
    node = VisualOffboardNode()
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

#!/usr/bin/env python3
############################################################################
#
#   Fixed-Wing Pure Pursuit Controller — Koordinat Çerçevesi Düzeltmesi
#
#   SORUN: Her araç kendi HOME konumuna göre NED kullanır.
#   /fmu/out/vehicle_local_position  → takipçi NED (kendi home'u)
#   /px4_2/fmu/out/vehicle_local_position → hedef NED (hedefin home'u)
#   Bunları direkt çıkarmak YANLIŞ yön verir!
#
#   ÇÖZÜM: Her araçtan ref_lat/ref_lon alıp ORTAK bir referans çerçevesine
#   (WGS84 → ENU metre) dönüştür. Böylece her iki konumu aynı uzayda
#   karşılaştırabiliriz.
#
#   Koordinat sistemi: NED (PX4 standardı)
#     x = kuzey, y = doğu, z = aşağı (negatif = yukarı)
#
############################################################################

import rclpy
import math
import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint
from px4_msgs.msg import VehicleStatus, VehicleCommand
from px4_msgs.msg import VehicleLocalPosition

# ── WGS84 sabitleri ────────────────────────────────────────────────────── #
_WGS84_A = 6378137.0          # yarı büyük eksen [m]
_WGS84_E2 = 6.6943799901e-3   # eccentricity²


def _lla_to_ecef(lat_deg: float, lon_deg: float, alt_m: float):
    """Lat/Lon/Alt → ECEF (metre)."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    N = _WGS84_A / math.sqrt(1 - _WGS84_E2 * math.sin(lat) ** 2)
    x = (N + alt_m) * math.cos(lat) * math.cos(lon)
    y = (N + alt_m) * math.cos(lat) * math.sin(lon)
    z = (N * (1 - _WGS84_E2) + alt_m) * math.sin(lat)
    return x, y, z


def _ecef_to_ned(ecef_ref, ecef_pt, lat_ref_deg: float, lon_ref_deg: float):
    """Bir noktanın ECEF koordinatını, referans noktaya göre NED'e çevir."""
    lat = math.radians(lat_ref_deg)
    lon = math.radians(lon_ref_deg)
    dx = ecef_pt[0] - ecef_ref[0]
    dy = ecef_pt[1] - ecef_ref[1]
    dz = ecef_pt[2] - ecef_ref[2]
    n = (-math.sin(lat) * math.cos(lon) * dx
         - math.sin(lat) * math.sin(lon) * dy
         + math.cos(lat) * dz)
    e = (-math.sin(lon) * dx
         + math.cos(lon) * dy)
    d = (-math.cos(lat) * math.cos(lon) * dx
         - math.cos(lat) * math.sin(lon) * dy
         - math.sin(lat) * dz)
    return n, e, d  # NED


def ned_relative(
    my_ref_lat, my_ref_lon, my_ref_alt,
    my_x, my_y, my_z,
    tgt_ref_lat, tgt_ref_lon, tgt_ref_alt,
    tgt_x, tgt_y, tgt_z
):
    """
    Hedefin konumunu takipçinin NED çerçevesine dönüştür.
    Dönüş: (rel_n, rel_e, rel_d) — takipçiye göre hedef pozisyonu [m].
    """
    # Takipçinin GPS konumu
    my_lat_deg  = math.degrees(my_ref_lat)  + my_x / _WGS84_A * (180 / math.pi)
    my_lon_deg  = math.degrees(my_ref_lon)  + my_y / (_WGS84_A * math.cos(my_ref_lat)) * (180 / math.pi)
    my_alt      = my_ref_alt - my_z  # NED: z aşağı

    # Hedefin GPS konumu
    tgt_lat_deg = math.degrees(tgt_ref_lat) + tgt_x / _WGS84_A * (180 / math.pi)
    tgt_lon_deg = math.degrees(tgt_ref_lon) + tgt_y / (_WGS84_A * math.cos(tgt_ref_lat)) * (180 / math.pi)
    tgt_alt     = tgt_ref_alt - tgt_z

    # Her ikisini ECEF'e çevir
    my_ecef  = _lla_to_ecef(my_lat_deg,  my_lon_deg,  my_alt)
    tgt_ecef = _lla_to_ecef(tgt_lat_deg, tgt_lon_deg, tgt_alt)

    # ECEF farkını takipçinin NED çerçevesine çevir
    rel_n, rel_e, rel_d = _ecef_to_ned(my_ecef, tgt_ecef, my_lat_deg, my_lon_deg)
    return rel_n, rel_e, rel_d


# ─────────────────────────────────────────────────────────────────────────── #


class FWPursuitController(Node):
    """
    Fixed-wing pure pursuit — koordinat çerçevesi dönüşümlü.

    Her araç kendi NED home'una sahip olduğundan, hedefin pozisyonu
    GPS/ECEF üzerinden takipçinin NED çerçevesine dönüştürülür.

    Kontrol formülü (takipçi NED'inde):
        follow_pt = tgt_pos_ned - normalize(tgt_vel_ned) * FOLLOW_DIST
        err       = follow_pt - (0, 0)  [takipçi her zaman orijinde]
        v_cmd_xy  = clamp(
                        FF * tgt_vel_ned_xy
                      + Kp * err_xy
                      + Kd * (tgt_vel - my_vel)_xy,
                      MIN_SPEED, MAX_SPEED)
        vz_cmd    = clamp(Kp_alt * rel_d, -MAX_VZ, MAX_VZ)
    """

    # ── Parametreler ──────────────────────────────────────────────────── #
    FOLLOW_DIST  = 30.0    # Takip mesafesi [m]
    MIN_SPEED    = 10.0    # Min fixed-wing hızı (stall önleme) [m/s]
    MAX_SPEED    = 20.0    # Max komut hızı [m/s]
    KP           = 0.4     # Yaklaşma kazancı [1/s]
    KD_VEL       = 0.5     # Hız eşleşme kazancı
    FF_WEIGHT    = 1.0     # Feedforward ağırlığı
    KP_ALT       = 0.5     # İrtifa kazancı [m/s per m]
    MAX_VZ       = 3.0     # Max dikey hız [m/s]
    LP_ALPHA     = 0.5     # Low-pass alpha (1=filtresiz)
    DEADBAND_M   = 3.0     # Yaklaşma kesme mesafesi [m]

    # Otomatik arm + offboard (simülatörde True, gerçek uçuşta False)
    AUTO_ARM_OFFBOARD       = True
    OFFBOARD_SETPOINT_COUNT = 30   # Kaç heartbeat sonra arm/offboard

    def __init__(self):
        super().__init__('fw_pursuit_controller')

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # ── Durum ─────────────────────────────────────────────────────── #
        self.nav_state    = VehicleStatus.NAVIGATION_STATE_MAX
        self.arming_state = VehicleStatus.ARMING_STATE_DISARMED

        # Takipçi — NED + ref
        self.my_x = 0.0; self.my_y = 0.0; self.my_z = 0.0
        self.my_vx = 0.0; self.my_vy = 0.0
        self.my_ref_lat = None   # radyan
        self.my_ref_lon = None
        self.my_ref_alt = 0.0
        self.my_ok = False

        # Hedef — NED + ref (px4_2)
        self.tgt_x = None; self.tgt_y = None; self.tgt_z = None
        self.tgt_vx = 0.0; self.tgt_vy = 0.0
        self.tgt_ref_lat = None
        self.tgt_ref_lon = None
        self.tgt_ref_alt = 0.0
        self.tgt_ok = False

        # Offboard sayacı
        self._sp_count = 0

        self.dt = 0.02  # 50 Hz

        # ── Abonelikler ───────────────────────────────────────────────── #
        # Durum
        for topic in ['/fmu/out/vehicle_status', '/fmu/out/vehicle_status_v4']:
            self.create_subscription(VehicleStatus, topic, self.status_cb, qos)

        # Takipçi pozisyonu
        for topic in ['/fmu/out/vehicle_local_position',
                      '/fmu/out/vehicle_local_position_v1']:
            self.create_subscription(VehicleLocalPosition, topic,
                                     self.my_pos_cb, qos)

        # Hedef pozisyonu
        for topic in ['/px4_2/fmu/out/vehicle_local_position',
                      '/px4_2/fmu/out/vehicle_local_position_v1']:
            self.create_subscription(VehicleLocalPosition, topic,
                                     self.tgt_pos_cb, qos)

        # ── Yayıncılar ────────────────────────────────────────────────── #
        self.pub_ocm = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos)
        self.pub_tsp = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos)
        self.pub_cmd = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos)

        self.timer = self.create_timer(self.dt, self.cmdloop)

        self.get_logger().info(
            f'FWPursuit başladı | follow={self.FOLLOW_DIST}m '
            f'min={self.MIN_SPEED} max={self.MAX_SPEED} Kp={self.KP}')

    # ── Callbacks ─────────────────────────────────────────────────────── #

    def status_cb(self, msg: VehicleStatus):
        self.nav_state    = msg.nav_state
        self.arming_state = msg.arming_state

    def my_pos_cb(self, msg: VehicleLocalPosition):
        self.my_x = msg.x; self.my_y = msg.y; self.my_z = msg.z
        self.my_vx = msg.vx; self.my_vy = msg.vy
        if msg.ref_lat != 0.0 or msg.ref_lon != 0.0:
            self.my_ref_lat = math.radians(msg.ref_lat)
            self.my_ref_lon = math.radians(msg.ref_lon)
            self.my_ref_alt = msg.ref_alt
        self.my_ok = True

    def tgt_pos_cb(self, msg: VehicleLocalPosition):
        a = self.LP_ALPHA
        if self.tgt_x is None:
            self.tgt_x = msg.x; self.tgt_y = msg.y; self.tgt_z = msg.z
            self.tgt_vx = msg.vx; self.tgt_vy = msg.vy
            self.get_logger().info(
                f'Hedef ilk: x={msg.x:.1f} y={msg.y:.1f} z={msg.z:.1f} '
                f'ref_lat={msg.ref_lat:.6f} ref_lon={msg.ref_lon:.6f}')
        else:
            self.tgt_x  = a*msg.x  + (1-a)*self.tgt_x
            self.tgt_y  = a*msg.y  + (1-a)*self.tgt_y
            self.tgt_z  = a*msg.z  + (1-a)*self.tgt_z
            self.tgt_vx = a*msg.vx + (1-a)*self.tgt_vx
            self.tgt_vy = a*msg.vy + (1-a)*self.tgt_vy

        if msg.ref_lat != 0.0 or msg.ref_lon != 0.0:
            self.tgt_ref_lat = math.radians(msg.ref_lat)
            self.tgt_ref_lon = math.radians(msg.ref_lon)
            self.tgt_ref_alt = msg.ref_alt
            self.tgt_ok = True

    # ── Komut yardımcıları ─────────────────────────────────────────────── #

    def _vehicle_cmd(self, command: int, p1=0.0, p2=0.0):
        msg = VehicleCommand()
        msg.timestamp        = int(self.get_clock().now().nanoseconds / 1000)
        msg.command          = command
        msg.param1           = float(p1)
        msg.param2           = float(p2)
        msg.target_system    = 1
        msg.target_component = 1
        msg.source_system    = 1
        msg.source_component = 1
        msg.from_external    = True
        self.pub_cmd.publish(msg)

    def _arm(self):
        self._vehicle_cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, p1=1.0)
        self.get_logger().info('→ ARM komutu gönderildi')

    def _set_offboard(self):
        self._vehicle_cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
                          p1=1.0, p2=6.0)
        self.get_logger().info('→ OFFBOARD komutu gönderildi')

    # ── Hız kırpma (fixed-wing min/max) ───────────────────────────────── #

    def _clamp_speed(self, vx: float, vy: float):
        spd = math.hypot(vx, vy)
        if spd < 1e-6:
            return float(self.MIN_SPEED), 0.0
        new_spd = float(np.clip(spd, self.MIN_SPEED, self.MAX_SPEED))
        s = new_spd / spd
        return vx * s, vy * s

    # ── Ana döngü ─────────────────────────────────────────────────────── #

    def cmdloop(self):
        ts = int(self.get_clock().now().nanoseconds / 1000)

        is_armed    = (self.arming_state == VehicleStatus.ARMING_STATE_ARMED)
        is_offboard = (self.nav_state    == VehicleStatus.NAVIGATION_STATE_OFFBOARD)

        # Offboard heartbeat — HER ZAMAN
        ocm = OffboardControlMode()
        ocm.timestamp    = ts
        ocm.position     = False
        ocm.velocity     = True
        ocm.acceleration = False
        ocm.attitude     = False
        ocm.body_rate    = False
        self.pub_ocm.publish(ocm)

        # Veri kontrolü
        if not self.my_ok or self.tgt_x is None:
            self._send_idle(ts)
            self.get_logger().info(
                f'Veri bekleniyor: my_ok={self.my_ok} tgt_ok={self.tgt_x is not None}',
                throttle_duration_sec=3.0)
            return

        # Koordinat dönüşümü gerekiyor mu?
        same_frame = (
            self.my_ref_lat is None or
            self.tgt_ref_lat is None or
            (abs(self.my_ref_lat - self.tgt_ref_lat) < 1e-8 and
             abs(self.my_ref_lon - self.tgt_ref_lon) < 1e-8)
        )

        if same_frame:
            # Aynı NED çerçevesi: basit çıkarma
            rel_n = self.tgt_x - self.my_x
            rel_e = self.tgt_y - self.my_y
            rel_d = self.tgt_z - self.my_z
            # Hedef hızı aynı çerçevede
            tgt_vn = self.tgt_vx
            tgt_ve = self.tgt_vy
        else:
            # Farklı NED çerçeveleri: GPS dönüşümü
            rel_n, rel_e, rel_d = ned_relative(
                self.my_ref_lat,  self.my_ref_lon,  self.my_ref_alt,
                self.my_x,        self.my_y,         self.my_z,
                self.tgt_ref_lat, self.tgt_ref_lon,  self.tgt_ref_alt,
                self.tgt_x,       self.tgt_y,         self.tgt_z
            )
            # Hedef hızı: NED vektörü, küçük açılar için çerçeve dönüşümü
            # ihmal edilebilir (yalnızca yönelim farkı — genellikle sıfır)
            tgt_vn = self.tgt_vx
            tgt_ve = self.tgt_vy

        dist_3d = math.sqrt(rel_n**2 + rel_e**2 + rel_d**2)
        dist_xy = math.hypot(rel_n, rel_e)

        # ── Follow point hesapla ─────────────────────────────────────── #
        tgt_spd_xy = math.hypot(tgt_vn, tgt_ve)
        if tgt_spd_xy > 1.0:
            # Hedef hareket ediyor: hız yönünün tersine offset
            bx = -tgt_vn / tgt_spd_xy
            by = -tgt_ve / tgt_spd_xy
        else:
            # Hedef duruyorsa: takipçi→hedef birim vektörünün tersine
            if dist_xy > 1.0:
                bx = -rel_n / dist_xy
                by = -rel_e / dist_xy
            else:
                bx, by = -1.0, 0.0

        # Follow point takipçi NED'inde (takipçi orijinde = 0,0)
        fp_n = rel_n + bx * self.FOLLOW_DIST
        fp_e = rel_e + by * self.FOLLOW_DIST
        fp_err = math.hypot(fp_n, fp_e)

        # ── Yaklaşma bileşeni ─────────────────────────────────────────── #
        if fp_err > self.DEADBAND_M:
            app_n = self.KP * fp_n
            app_e = self.KP * fp_e
        else:
            app_n = app_e = 0.0

        # ── Hız eşleşme bileşeni ─────────────────────────────────────── #
        dv_n = self.KD_VEL * (tgt_vn - self.my_vx)
        dv_e = self.KD_VEL * (tgt_ve - self.my_vy)

        # ── Feedforward ───────────────────────────────────────────────── #
        ff_n = self.FF_WEIGHT * tgt_vn
        ff_e = self.FF_WEIGHT * tgt_ve

        # ── Toplam XY komutu ──────────────────────────────────────────── #
        raw_vn = ff_n + app_n + dv_n
        raw_ve = ff_e + app_e + dv_e
        vn_cmd, ve_cmd = self._clamp_speed(raw_vn, raw_ve)

        # ── Yükseklik (Z) ─────────────────────────────────────────────── #
        # rel_d > 0 → hedef aşağıda → pozitif vz (aşağı) → yanlış!
        # rel_d < 0 → hedef yukarıda → negatif vz (yukarı) → doğru
        vz_cmd = float(np.clip(self.KP_ALT * rel_d, -self.MAX_VZ, self.MAX_VZ))

        # ── Yaw ───────────────────────────────────────────────────────── #
        yaw_cmd = math.atan2(ve_cmd, vn_cmd)   # NED: atan2(east, north)

        # ── Auto arm / offboard ──────────────────────────────────────── #
        if self.AUTO_ARM_OFFBOARD:
            self._sp_count += 1
            if self._sp_count == self.OFFBOARD_SETPOINT_COUNT and not is_armed:
                self._arm()
            if self._sp_count > self.OFFBOARD_SETPOINT_COUNT and not is_offboard:
                self._set_offboard()

        if not (is_armed and is_offboard):
            self._send_idle(ts)
            self.get_logger().info(
                f'Bekleniyor: armed={is_armed} offboard={is_offboard} '
                f'sayaç={self._sp_count}',
                throttle_duration_sec=3.0)
            return

        # ── TrajectorySetpoint gönder ─────────────────────────────────── #
        tsp = TrajectorySetpoint()
        tsp.timestamp    = ts
        tsp.position     = [float('nan')] * 3
        tsp.velocity     = [float(vn_cmd), float(ve_cmd), float(vz_cmd)]
        tsp.acceleration = [float('nan')] * 3
        tsp.jerk         = [float('nan')] * 3
        tsp.yaw          = float(yaw_cmd)
        tsp.yawspeed     = float('nan')
        self.pub_tsp.publish(tsp)

        # ── Debug ─────────────────────────────────────────────────────── #
        self.get_logger().info(
            f'[FW] dist={dist_3d:.0f}m  fp_err={fp_err:.0f}m  '
            f'rel=({rel_n:.0f}N,{rel_e:.0f}E,{rel_d:.0f}D)  '
            f'same_frame={same_frame}  '
            f'tgt_v=({tgt_vn:.1f},{tgt_ve:.1f})m/s  '
            f'cmd=({vn_cmd:+.1f},{ve_cmd:+.1f},{vz_cmd:+.1f})  '
            f'yaw={math.degrees(yaw_cmd):.0f}°',
            throttle_duration_sec=0.5)

    def _send_idle(self, ts: int):
        """Offboard mod geçişi için idle setpoint (sabit yön, min hız)."""
        tsp = TrajectorySetpoint()
        tsp.timestamp = ts
        tsp.position  = [float('nan')] * 3
        tsp.velocity  = [float(self.MIN_SPEED), 0.0, 0.0]
        tsp.yaw       = 0.0
        self.pub_tsp.publish(tsp)


def main(args=None):
    rclpy.init(args=args)
    node = FWPursuitController()
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

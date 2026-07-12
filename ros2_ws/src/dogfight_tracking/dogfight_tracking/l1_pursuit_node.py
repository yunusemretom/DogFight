#!/usr/bin/env python3
############################################################################
#
#   L1 Guidance Pursuit Node — mesafe korumalı hedef takibi (fixed-wing)
#
#   Mimari (PX4 v1.16 offboard):
#     - L1 yanal ivme burada hesaplanır ve doğrudan
#       /fmu/in/fixed_wing_lateral_setpoint  (lateral_acceleration) ile verilir.
#     - İrtifa + airspeed /fmu/in/fixed_wing_longitudinal_setpoint ile
#       PX4 TECS'e bırakılır (stall koruması, pitch/throttle PX4'te).
#     - OffboardControlMode.velocity=true yayınlanır ama TrajectorySetpoint
#       YAYINLANMAZ: böylece FixedWingModeManager pasif kalır
#       (FW_POSCTRL_MODE_OTHER) ve bizim setpoint'lerimiz
#       FwLateralLongitudinalControl tarafından işlenir.
#
#   Mesafe koruma:
#     - Takip noktası: hedefin hız vektörünün gerisinde standoff_distance kadar.
#     - Airspeed komutu: hedef yer hızı + kp_distance * (mesafe - standoff).
#       Yaklaşınca yavaşlar, açılınca hızlanır.
#
#   Koordinat: iki aracın home'ları farklı olduğundan pozisyonlar
#   VehicleGlobalPosition (lat/lon/alt AMSL) üzerinden karşılaştırılır;
#   hızlar VehicleLocalPosition NED'dir (NED eksenleri yerel olarak paralel).
#
############################################################################

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSReliabilityPolicy,
                       QoSHistoryPolicy, QoSDurabilityPolicy)
from rcl_interfaces.msg import SetParametersResult

from px4_msgs.msg import (OffboardControlMode, VehicleCommand, VehicleStatus,
                          VehicleLocalPosition, VehicleGlobalPosition,
                          FixedWingLateralSetpoint, FixedWingLongitudinalSetpoint)

R_EARTH = 6371000.0
G = 9.80665

# Durum makinesi
S_WAIT_DATA = 'WAIT_DATA'   # pozisyon/hedef verisi bekleniyor
S_TAKEOFF   = 'TAKEOFF'     # arm + kalkış, irtifa bekleniyor
S_ENGAGE    = 'ENGAGE'      # offboard'a geçiş komutları gönderiliyor
S_PURSUIT   = 'PURSUIT'     # aktif L1 takip
S_HOLD      = 'HOLD'        # hedef verisi kesildi — düz uçuş


def wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


class L1PursuitNode(Node):
    """Mesafe korumalı L1 guidance takip kontrolcüsü."""

    def __init__(self):
        super().__init__('l1_pursuit_node')

        # ── Parametreler ─────────────────────────────────────────────── #
        p = self.declare_parameter
        p('follower_ns', '')            # takipçi namespace ('' = /fmu)
        p('target_ns', '/px4_1')        # hedef namespace
        p('standoff_distance', 5.0)    # korunacak takip mesafesi [m]
        p('l1_period', 15.0)            # L1 periyodu [s] (küçük = agresif)
        p('l1_damping', 0.75)           # L1 sönümleme
        p('max_lateral_accel', 8.0)     # yanal ivme limiti [m/s²] (~39° roll)
        p('min_airspeed', 10.0)         # min airspeed komutu [m/s]
        p('max_airspeed', 30.0)         # max airspeed komutu [m/s]
        p('cruise_airspeed', 15.0)      # hold/idle airspeed [m/s]
        p('kp_distance', 0.15)          # mesafe hatası → hız kazancı [1/s]
        p('altitude_offset', 0.0)       # hedef irtifasına eklenecek fark [m]
        p('target_timeout', 2.0)        # hedef verisi zaman aşımı [s]
        p('auto_arm_takeoff', True)     # SITL: otomatik arm + kalkış
        p('auto_offboard', True)        # irtifa alınca offboard'a geç
        p('engage_altitude', 15.0)      # offboard'a geçilecek min irtifa [m AGL]
        p('system_id', 0)               # VehicleCommand hedef sistemi (0 = otomatik)
        # Yakın takip (formasyon) modu: L1 yerine hedef rotası + cross-track
        p('formation_enter_dist', 30.0)  # bu mesafenin altında formasyon modu [m]
        p('formation_exit_dist', 45.0)   # bu mesafenin üstünde L1 moduna dön [m]
        p('kp_cross', 0.4)               # cross-track → rota düzeltme kazancı [1/s]
        p('max_course_corr_deg', 35.0)   # rota düzeltme limiti [derece]
        p('ki_distance', 0.08)           # along-track integral kazancı [1/s²]
        p('kd_distance', 0.6)            # kapanma hızı sönümleme kazancı [-]
        p('min_separation', 3.0)         # emniyet mesafesi (3D) [m]

        gp = lambda n: self.get_parameter(n).value
        self.follower_ns = str(gp('follower_ns')).rstrip('/')
        self.target_ns = str(gp('target_ns')).rstrip('/')
        self.standoff = float(gp('standoff_distance'))
        self.l1_period = float(gp('l1_period'))
        self.l1_damping = float(gp('l1_damping'))
        self.max_lat_acc = float(gp('max_lateral_accel'))
        self.min_aspd = float(gp('min_airspeed'))
        self.max_aspd = float(gp('max_airspeed'))
        self.cruise_aspd = float(gp('cruise_airspeed'))
        self.kp_dist = float(gp('kp_distance'))
        self.alt_offset = float(gp('altitude_offset'))
        self.tgt_timeout = float(gp('target_timeout'))
        self.auto_arm_takeoff = bool(gp('auto_arm_takeoff'))
        self.auto_offboard = bool(gp('auto_offboard'))
        self.engage_alt = float(gp('engage_altitude'))
        self.form_enter = float(gp('formation_enter_dist'))
        self.form_exit = float(gp('formation_exit_dist'))
        self.kp_cross = float(gp('kp_cross'))
        self.max_course_corr = math.radians(float(gp('max_course_corr_deg')))
        self.ki_dist = float(gp('ki_distance'))
        self.kd_dist = float(gp('kd_distance'))
        self.min_sep = float(gp('min_separation'))

        sysid = int(gp('system_id'))
        if sysid == 0:
            # PX4 SITL: instance i → MAV_SYS_ID i+1 (/px4_N → N+1, ns'siz → 1)
            sysid = 1
            if self.follower_ns.startswith('/px4_'):
                sysid = int(self.follower_ns.split('_')[-1]) + 1
        self.sysid = sysid

        self.add_on_set_parameters_callback(self._param_cb)

        # ── Durum ────────────────────────────────────────────────────── #
        self.state = S_WAIT_DATA
        self.arming_state = VehicleStatus.ARMING_STATE_DISARMED
        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX

        self.my_glob = None      # VehicleGlobalPosition
        self.my_loc = None       # VehicleLocalPosition
        self.tgt_glob = None
        self.tgt_loc = None
        self.my_glob_t = 0.0
        self.tgt_glob_t = 0.0

        # Hedef dönüş hızı tahmini
        self.tgt_omega = 0.0        # [rad/s], + sağa dönüş
        self._tgt_course = None
        self._tgt_course_t = 0.0

        # Yakın takip modu (histerezisli)
        self.formation = False
        self._dist_i = 0.0       # along-track PI integrali [m/s]
        self._avoiding = False   # emniyet mesafesi ihlali aktif

        self._loop_count = 0
        self._last_cmd_t = 0.0
        self.dt = 0.02  # kontrol döngüsü periyodu [s]

        # ── QoS ──────────────────────────────────────────────────────── #
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1)

        # ── Abonelikler (versiyonlu + versiyonsuz topic varyantları) ──── #
        def sub_multi(msg_type, base, cb):
            for t in (base, base + '_v1', base + '_v2', base + '_v4'):
                self.create_subscription(msg_type, t, cb, qos)

        f, t = self.follower_ns, self.target_ns
        sub_multi(VehicleStatus, f + '/fmu/out/vehicle_status', self._status_cb)
        sub_multi(VehicleLocalPosition, f + '/fmu/out/vehicle_local_position',
                  self._my_loc_cb)
        sub_multi(VehicleGlobalPosition, f + '/fmu/out/vehicle_global_position',
                  self._my_glob_cb)
        sub_multi(VehicleLocalPosition, t + '/fmu/out/vehicle_local_position',
                  self._tgt_loc_cb)
        sub_multi(VehicleGlobalPosition, t + '/fmu/out/vehicle_global_position',
                  self._tgt_glob_cb)

        # ── Yayıncılar ───────────────────────────────────────────────── #
        self.pub_ocm = self.create_publisher(
            OffboardControlMode, f + '/fmu/in/offboard_control_mode', qos)
        self.pub_lat = self.create_publisher(
            FixedWingLateralSetpoint, f + '/fmu/in/fixed_wing_lateral_setpoint', qos)
        self.pub_lon = self.create_publisher(
            FixedWingLongitudinalSetpoint,
            f + '/fmu/in/fixed_wing_longitudinal_setpoint', qos)
        self.pub_cmd = self.create_publisher(
            VehicleCommand, f + '/fmu/in/vehicle_command', qos)

        self.timer = self.create_timer(0.02, self._loop)  # 50 Hz

        self.get_logger().info(
            f"L1 pursuit başladı | takipçi='{f or '/'}' hedef='{t}' "
            f'sysid={self.sysid} standoff={self.standoff}m '
            f'L1: T={self.l1_period}s ζ={self.l1_damping}')

    # ── Parametre güncelleme (çalışırken tune için) ───────────────────── #

    def _param_cb(self, params):
        attr = {'standoff_distance': 'standoff', 'l1_period': 'l1_period',
                'l1_damping': 'l1_damping', 'max_lateral_accel': 'max_lat_acc',
                'min_airspeed': 'min_aspd', 'max_airspeed': 'max_aspd',
                'cruise_airspeed': 'cruise_aspd', 'kp_distance': 'kp_dist',
                'altitude_offset': 'alt_offset', 'target_timeout': 'tgt_timeout',
                'formation_enter_dist': 'form_enter',
                'formation_exit_dist': 'form_exit', 'kp_cross': 'kp_cross',
                'ki_distance': 'ki_dist', 'kd_distance': 'kd_dist',
                'min_separation': 'min_sep'}
        for prm in params:
            if prm.name in attr:
                setattr(self, attr[prm.name], float(prm.value))
                self.get_logger().info(f'Parametre: {prm.name} = {prm.value}')
            elif prm.name == 'max_course_corr_deg':
                self.max_course_corr = math.radians(float(prm.value))
                self.get_logger().info(f'Parametre: {prm.name} = {prm.value}')
        return SetParametersResult(successful=True)

    # ── Callbacks ─────────────────────────────────────────────────────── #

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _status_cb(self, msg: VehicleStatus):
        self.arming_state = msg.arming_state
        self.nav_state = msg.nav_state

    def _my_loc_cb(self, msg: VehicleLocalPosition):
        self.my_loc = msg

    def _my_glob_cb(self, msg: VehicleGlobalPosition):
        self.my_glob = msg
        self.my_glob_t = self._now()

    def _tgt_loc_cb(self, msg: VehicleLocalPosition):
        # Hedefin dönüş hızını (course rate) filtreli türevle tahmin et:
        # takip noktasını teğet yerine yay üzerine koymak için kullanılır
        t = self._now()
        spd = math.hypot(msg.vx, msg.vy)
        if spd > 3.0:
            course = math.atan2(msg.vy, msg.vx)
            if self._tgt_course is not None and t > self._tgt_course_t:
                omega = wrap_pi(course - self._tgt_course) / (t - self._tgt_course_t)
                if abs(omega) < 1.5:  # sıçramaları at
                    self.tgt_omega += 0.2 * (omega - self.tgt_omega)
            self._tgt_course, self._tgt_course_t = course, t
        self.tgt_loc = msg

    def _tgt_glob_cb(self, msg: VehicleGlobalPosition):
        self.tgt_glob = msg
        self.tgt_glob_t = self._now()

    # ── VehicleCommand yardımcıları ───────────────────────────────────── #

    def _cmd(self, command, p1=0.0, p2=0.0, p3=0.0, p4=0.0,
             p5=0.0, p6=0.0, p7=0.0):
        msg = VehicleCommand()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.command = command
        msg.param1, msg.param2, msg.param3 = float(p1), float(p2), float(p3)
        msg.param4 = float(p4)
        msg.param5, msg.param6, msg.param7 = float(p5), float(p6), float(p7)
        msg.target_system = self.sysid
        msg.target_component = 1
        msg.source_system = self.sysid
        msg.source_component = 1
        msg.from_external = True
        self.pub_cmd.publish(msg)

    # ── L1 guidance çekirdeği ─────────────────────────────────────────── #

    def _relative_ned(self):
        """Hedefin takipçiye göre NED pozisyonu [m] (flat-earth, <~5 km)."""
        mg, tg = self.my_glob, self.tgt_glob
        rel_n = math.radians(tg.lat - mg.lat) * R_EARTH
        rel_e = (math.radians(tg.lon - mg.lon) * R_EARTH
                 * math.cos(math.radians(mg.lat)))
        rel_d = mg.alt - tg.alt  # NED: aşağı pozitif
        return rel_n, rel_e, rel_d

    def _l1_lateral_accel(self, fp_n, fp_e, v_n, v_e):
        """Klasik L1: takip noktasına doğru yanal ivme komutu [m/s², FRD sağ+]."""
        v_g = max(math.hypot(v_n, v_e), 3.0)
        l1_dist = max(self.l1_damping * self.l1_period * v_g / math.pi, 10.0)

        course = math.atan2(v_e, v_n)
        bearing = math.atan2(fp_e, fp_n)
        # eta: hız vektörü ile L1 doğrultusu arası açı, ±90°'ye sınırlı
        eta = wrap_pi(bearing - course)
        eta = max(-math.pi / 2.0, min(math.pi / 2.0, eta))

        a_cmd = 2.0 * v_g * v_g / l1_dist * math.sin(eta)
        return max(-self.max_lat_acc, min(self.max_lat_acc, a_cmd))

    def _pursuit_setpoints(self):
        """L1 + mesafe koruma → (lateral_accel, altitude_amsl, airspeed)."""
        rel_n, rel_e, rel_d = self._relative_ned()
        dist_xy = math.hypot(rel_n, rel_e)

        tgt_vn, tgt_ve = float(self.tgt_loc.vx), float(self.tgt_loc.vy)
        tgt_spd = math.hypot(tgt_vn, tgt_ve)

        # Takip noktası: hedefin izlediği yolun standoff kadar gerisinde.
        # Hedef dönüyorsa nokta teğet üzerinde değil YAY üzerinde hesaplanır;
        # aksi halde nokta dönüş dairesinin dışına düşer ve kalıcı mesafe
        # sapması oluşur.
        omega = self.tgt_omega
        if tgt_spd > 3.0 and abs(omega) > 0.03:
            u_n, u_e = tgt_vn / tgt_spd, tgt_ve / tgt_spd
            r_turn = tgt_spd / abs(omega)
            sgn = 1.0 if omega > 0.0 else -1.0
            # Dönüş merkezi: hız vektörünün sgn*90° yanında
            c_n = rel_n + r_turn * (-sgn * u_e)
            c_e = rel_e + r_turn * (sgn * u_n)
            # Yayda standoff kadar geriye dön
            theta = -sgn * min(self.standoff / r_turn, math.pi / 2.0)
            dn, de = rel_n - c_n, rel_e - c_e
            fp_n = c_n + dn * math.cos(theta) - de * math.sin(theta)
            fp_e = c_e + dn * math.sin(theta) + de * math.cos(theta)
        elif tgt_spd > 3.0:
            fp_n = rel_n - tgt_vn / tgt_spd * self.standoff
            fp_e = rel_e - tgt_ve / tgt_spd * self.standoff
        elif dist_xy > 1.0:
            fp_n = rel_n - rel_n / dist_xy * self.standoff
            fp_e = rel_e - rel_e / dist_xy * self.standoff
        else:
            fp_n, fp_e = rel_n - self.standoff, rel_e

        my_vn, my_ve = float(self.my_loc.vx), float(self.my_loc.vy)
        v_g = max(math.hypot(my_vn, my_ve), 3.0)

        # ── Mod seçimi (histerezisli) ────────────────────────────────── #
        # Yakında L1 kullanılmaz: lookahead takip noktasından çok uzakta
        # kaldığı için küçük hatalar büyük yön sıçramaları (weaving) üretir.
        if self.formation and (dist_xy > self.form_exit or tgt_spd <= 3.0):
            self.formation = False
            self.get_logger().info(f'{dist_xy:.0f} m → L1 moduna dönüldü')
        elif not self.formation and dist_xy < self.form_enter and tgt_spd > 3.0:
            self.formation = True
            self.get_logger().info(f'{dist_xy:.0f} m → FORMASYON modu '
                                   '(rota takibi + cross-track)')

        alt = float(self.tgt_glob.alt) + self.alt_offset

        if self.formation:
            # Yanal: hedefin rotası + cross-track P düzeltmesi.
            # Dönüş feedforward'u lateral_acceleration alanıyla verilir;
            # PX4 course sonlu iken bu alanı feedforward olarak toplar.
            u_n, u_e = tgt_vn / tgt_spd, tgt_ve / tgt_spd
            e_along = fp_n * u_n + fp_e * u_e         # + = nokta önümde
            e_cross = u_n * fp_e - u_e * fp_n         # + = nokta rotanın sağında
            corr = math.atan2(self.kp_cross * e_cross, v_g)
            corr = max(-self.max_course_corr, min(self.max_course_corr, corr))
            course_cmd = wrap_pi(math.atan2(tgt_ve, tgt_vn) + corr)
            a_cmd = v_g * self.tgt_omega              # viraj eşleme feedforward

            # Boylamsal: mesafe SADECE along-track hatasıyla tutulur.
            # Integral şart: TECS'in airspeed takip bias'ı (~0.5 m/s) sadece
            # P ile standoff'u metrelerce kaydırır (5 m'de çarpışmaya götürür).
            # D terimi analitik: kapanma hızı = hedef hızı - benim rota-boyu
            # hızım; TECS gecikmesine karşı erken fren sağlar.
            e_rate = tgt_spd - (my_vn * u_n + my_ve * u_e)
            if not self._avoiding:
                self._dist_i += self.ki_dist * e_along * self.dt
                self._dist_i = max(-2.0, min(2.0, self._dist_i))
            aspd = (tgt_spd + self.kp_dist * e_along
                    + self.kd_dist * e_rate + self._dist_i)
        else:
            # L1 modu (uzak): yanal ivme komutu, rota NaN
            course_cmd = float('nan')
            a_cmd = self._l1_lateral_accel(fp_n, fp_e, my_vn, my_ve)
            self._dist_i = 0.0
            aspd = tgt_spd + self.kp_dist * (dist_xy - self.standoff)

        aspd = max(self.min_aspd, min(self.max_aspd, aspd))

        # ── Emniyet: minimum ayrılma (3D, histerezisli) ──────────────── #
        dist_3d = math.hypot(dist_xy, rel_d)
        if not self._avoiding and dist_3d < self.min_sep:
            self._avoiding = True
            self.get_logger().warn(
                f'EMNİYET: {dist_3d:.1f} m < {self.min_sep:.1f} m — '
                'yavaşla + hedefin üstüne irtifa aç')
        elif self._avoiding and dist_3d > self.min_sep + 2.0:
            self._avoiding = False
            self.get_logger().info('Emniyet mesafesi geri kazanıldı')
        if self._avoiding:
            # İntegral dondurulur (yukarıda); yavaşla ve hedefin üstüne çık
            aspd = self.min_aspd
            alt = float(self.tgt_glob.alt) + 5.0

        return a_cmd, alt, aspd, dist_xy, course_cmd

    # ── Setpoint yayınlama ────────────────────────────────────────────── #

    def _publish_setpoints(self, a_lat, alt, aspd, course=float('nan')):
        ts = int(self.get_clock().now().nanoseconds / 1000)

        ocm = OffboardControlMode()
        ocm.timestamp = ts
        ocm.velocity = True  # FwLateralLongitudinalControl'ü çalıştırır;
        # TrajectorySetpoint yayınlanmadığı için mode manager pasif kalır
        self.pub_ocm.publish(ocm)

        lat_sp = FixedWingLateralSetpoint()
        lat_sp.timestamp = ts
        lat_sp.course = float(course)
        lat_sp.airspeed_direction = float('nan')
        lat_sp.lateral_acceleration = float(a_lat)
        self.pub_lat.publish(lat_sp)

        lon_sp = FixedWingLongitudinalSetpoint()
        lon_sp.timestamp = ts
        lon_sp.altitude = float(alt)          # NaN → mevcut irtifayı korur
        lon_sp.height_rate = float('nan')
        lon_sp.equivalent_airspeed = float(aspd)
        lon_sp.pitch_direct = float('nan')
        lon_sp.throttle_direct = float('nan')
        self.pub_lon.publish(lon_sp)

    # ── Ana döngü ─────────────────────────────────────────────────────── #

    def _loop(self):
        self._loop_count += 1
        now = self._now()
        log = self.get_logger()

        my_ok = self.my_glob is not None and self.my_loc is not None
        tgt_fresh = (self.tgt_glob is not None and self.tgt_loc is not None
                     and (now - self.tgt_glob_t) < self.tgt_timeout)

        is_armed = self.arming_state == VehicleStatus.ARMING_STATE_ARMED
        is_offboard = self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD
        agl = -float(self.my_loc.z) if self.my_loc is not None else 0.0
        airborne = agl > self.engage_alt

        # ── Durum geçişleri ──────────────────────────────────────────── #
        if self.state == S_WAIT_DATA:
            if my_ok and tgt_fresh:
                self.state = S_TAKEOFF if not airborne else S_ENGAGE
                log.info(f'Veri hazır → {self.state}')
        elif self.state == S_TAKEOFF:
            if airborne:
                self.state = S_ENGAGE
                log.info(f'İrtifa {agl:.0f} m → ENGAGE')
        elif self.state == S_ENGAGE:
            if is_offboard:
                self.state = S_PURSUIT
                log.info('OFFBOARD aktif → PURSUIT')
        elif self.state == S_PURSUIT:
            if not tgt_fresh:
                self.state = S_HOLD
                log.warn('Hedef verisi kesildi → HOLD (düz uçuş)')
            elif not is_offboard:
                # Failsafe/pilot modu değiştirdi; auto_offboard açıksa geri iste
                self.state = S_ENGAGE
                log.warn('OFFBOARD kaybedildi (failsafe/mod değişimi) → ENGAGE')
        elif self.state == S_HOLD:
            if tgt_fresh:
                self.state = S_PURSUIT
                log.info('Hedef verisi geldi → PURSUIT')

        # ── Durum eylemleri ──────────────────────────────────────────── #
        if self.state == S_WAIT_DATA:
            # Offboard heartbeat'i şimdiden akıt (geçiş için gerekli)
            self._publish_setpoints(0.0, float('nan'), self.cruise_aspd)
            log.info(f'Veri bekleniyor: takipçi={my_ok} hedef={tgt_fresh}',
                     throttle_duration_sec=3.0)
            return

        if self.state == S_TAKEOFF:
            self._publish_setpoints(0.0, float('nan'), self.cruise_aspd)
            if self.auto_arm_takeoff and now - self._last_cmd_t > 2.0:
                self._last_cmd_t = now
                if not is_armed:
                    self._cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, p1=1.0)
                    log.info('→ ARM')
                elif self.nav_state != VehicleStatus.NAVIGATION_STATE_AUTO_TAKEOFF:
                    self._cmd(VehicleCommand.VEHICLE_CMD_NAV_TAKEOFF)
                    log.info('→ TAKEOFF')
            elif not self.auto_arm_takeoff:
                log.info(f'Kalkış bekleniyor (AGL {agl:.0f}/{self.engage_alt:.0f} m) '
                         '— auto_arm_takeoff kapalı, elle kaldırın',
                         throttle_duration_sec=5.0)
            return

        if self.state == S_ENGAGE:
            # Geçişte hedefe uygun setpoint akıt, sonra offboard iste
            a_lat, alt, aspd, dist, course = self._pursuit_setpoints() if tgt_fresh \
                else (0.0, float('nan'), self.cruise_aspd, 0.0, float('nan'))
            self._publish_setpoints(a_lat, alt, aspd, course)
            if self.auto_offboard and now - self._last_cmd_t > 1.0:
                self._last_cmd_t = now
                self._cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, p1=1.0, p2=6.0)
                log.info('→ OFFBOARD istendi')
            elif not self.auto_offboard:
                log.info('auto_offboard kapalı — OFFBOARD moduna elle geçin',
                         throttle_duration_sec=5.0)
            return

        if self.state == S_HOLD:
            self._publish_setpoints(0.0, float('nan'), self.cruise_aspd)
            return

        # S_PURSUIT
        a_lat, alt, aspd, dist, course = self._pursuit_setpoints()
        self._publish_setpoints(a_lat, alt, aspd, course)
        mode = 'FORM' if self.formation else 'L1'
        log.info(
            f'[{mode}] mesafe={dist:.1f}m (hedef {self.standoff:.0f}m)  '
            f'a_lat={a_lat:+.1f}m/s²  aspd={aspd:.1f}m/s  alt={alt:.0f}m  '
            f'tgt_ω={math.degrees(self.tgt_omega):+.1f}°/s  offboard={is_offboard}',
            throttle_duration_sec=1.0)


def main(args=None):
    rclpy.init(args=args)
    node = L1PursuitNode()
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

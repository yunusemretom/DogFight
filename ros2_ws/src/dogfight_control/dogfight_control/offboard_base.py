#!/usr/bin/env python3
############################################################################
#
#   offboard_base — dogfight_control test node'larının ortak tabanı.
#
#   Sağladıkları (l1_pursuit_node'daki kanıtlanmış akışla aynı):
#     - PX4 v1.16 versiyonlu topic abonelikleri (_v1/_v2/_v4 varyantları)
#     - Çerçeve-güvenli bağıl konum: pozisyonlar VehicleGlobalPosition ile
#       karşılaştırılır (local NED origin'leri araçlar arası farklıdır!),
#       hızlar local NED'dir (eksenler yerel olarak paralel).
#     - Arm + kalkış + offboard geçiş durum makinesi:
#       WAIT_DATA → TAKEOFF → ENGAGE → ACTIVE, hedef timeout'unda HOLD.
#     - Hedef dönüş hızı (course-rate) tahmini (yay-üzerinde takip noktası için)
#
#   Alt sınıf sözleşmesi:
#     - __init__'te super().__init__(node_adı, ocm_flags={...}) çağrılır;
#       ocm_flags OffboardControlMode alanlarını belirler
#       (ör. {'attitude': True} veya {'velocity': True}).
#     - publish_active(ts): hedef verisi tazeyken kontrol yasası + setpoint.
#     - publish_idle(ts): beklerken/HOLD'da güvenli setpoint (offboard
#       heartbeat kabulü için setpoint akışı hep sürmelidir).
#     - Görsel gibi araç-dışı hedefler için subscribe_target=False verilip
#       target_fresh(now) override edilebilir.
#
############################################################################

import math

from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSReliabilityPolicy,
                       QoSHistoryPolicy, QoSDurabilityPolicy)

from px4_msgs.msg import (OffboardControlMode, VehicleCommand, VehicleStatus,
                          VehicleLocalPosition, VehicleGlobalPosition)

from dogfight_control.control_math import wrap_pi, relative_ned

# Durum makinesi
S_WAIT_DATA = 'WAIT_DATA'   # pozisyon/hedef verisi bekleniyor
S_TAKEOFF   = 'TAKEOFF'     # arm + kalkış, irtifa bekleniyor
S_ENGAGE    = 'ENGAGE'      # offboard'a geçiş komutları gönderiliyor
S_ACTIVE    = 'ACTIVE'      # aktif kontrol
S_HOLD      = 'HOLD'        # hedef verisi kesildi — güvenli bekleme


class OffboardTestBase(Node):

    def __init__(self, node_name: str, ocm_flags: dict,
                 subscribe_target: bool = True):
        super().__init__(node_name)
        self.ocm_flags = dict(ocm_flags)

        # ── Ortak parametreler ───────────────────────────────────────── #
        p = self.declare_parameter
        p('follower_ns', '')          # takipçi namespace ('' = /fmu)
        p('target_ns', '/px4_1')      # hedef araç namespace
        p('target_timeout', 2.0)      # hedef verisi zaman aşımı [s]
        p('auto_arm_takeoff', True)   # SITL: otomatik arm + kalkış
        p('auto_offboard', True)      # irtifa alınca offboard'a geç
        p('engage_altitude', 15.0)    # offboard'a geçilecek min irtifa [m AGL]
        p('system_id', 0)             # VehicleCommand hedefi (0 = otomatik)

        gp = lambda n: self.get_parameter(n).value
        self.follower_ns = str(gp('follower_ns')).rstrip('/')
        self.target_ns = str(gp('target_ns')).rstrip('/')
        self.tgt_timeout = float(gp('target_timeout'))
        self.auto_arm_takeoff = bool(gp('auto_arm_takeoff'))
        self.auto_offboard = bool(gp('auto_offboard'))
        self.engage_alt = float(gp('engage_altitude'))

        sysid = int(gp('system_id'))
        if sysid == 0:
            # PX4 SITL: instance i → MAV_SYS_ID i+1 (/px4_N → N+1, ns'siz → 1)
            sysid = 1
            if self.follower_ns.startswith('/px4_'):
                sysid = int(self.follower_ns.split('_')[-1]) + 1
        self.sysid = sysid

        # ── Durum ────────────────────────────────────────────────────── #
        self.state = S_WAIT_DATA
        self.arming_state = VehicleStatus.ARMING_STATE_DISARMED
        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX

        self.my_glob = None      # VehicleGlobalPosition
        self.my_loc = None       # VehicleLocalPosition
        self.tgt_glob = None
        self.tgt_loc = None
        self.tgt_glob_t = 0.0

        # Hedef dönüş hızı tahmini [rad/s], + sağa dönüş
        self.tgt_omega = 0.0
        self._tgt_course = None
        self._tgt_course_t = 0.0

        self._last_cmd_t = 0.0
        self.dt = 0.02  # kontrol döngüsü periyodu [s]

        # ── QoS ──────────────────────────────────────────────────────── #
        self.qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1)

        # ── Abonelikler (versiyonlu + versiyonsuz topic varyantları) ─── #
        f, t = self.follower_ns, self.target_ns
        self.sub_multi(VehicleStatus, f + '/fmu/out/vehicle_status',
                       self._status_cb)
        self.sub_multi(VehicleLocalPosition,
                       f + '/fmu/out/vehicle_local_position', self._my_loc_cb)
        self.sub_multi(VehicleGlobalPosition,
                       f + '/fmu/out/vehicle_global_position', self._my_glob_cb)
        if subscribe_target:
            self.sub_multi(VehicleLocalPosition,
                           t + '/fmu/out/vehicle_local_position',
                           self._tgt_loc_cb)
            self.sub_multi(VehicleGlobalPosition,
                           t + '/fmu/out/vehicle_global_position',
                           self._tgt_glob_cb)

        # ── Yayıncılar ───────────────────────────────────────────────── #
        self.pub_ocm = self.create_publisher(
            OffboardControlMode, f + '/fmu/in/offboard_control_mode', self.qos)
        self.pub_cmd = self.create_publisher(
            VehicleCommand, f + '/fmu/in/vehicle_command', self.qos)

        self.timer = self.create_timer(self.dt, self._loop)  # 50 Hz

    # ── Yardımcılar ───────────────────────────────────────────────────── #

    def sub_multi(self, msg_type, base, cb):
        for topic in (base, base + '_v1', base + '_v2', base + '_v4'):
            self.create_subscription(msg_type, topic, cb, self.qos)

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def now_us(self) -> int:
        return int(self.get_clock().now().nanoseconds / 1000)

    def rel_ned(self):
        """Hedefin takipçiye göre NED pozisyonu [m] (global pozisyondan)."""
        mg, tg = self.my_glob, self.tgt_glob
        return relative_ned(mg.lat, mg.lon, mg.alt, tg.lat, tg.lon, tg.alt)

    def my_heading(self) -> float:
        return float(self.my_loc.heading) if self.my_loc is not None else 0.0

    # ── Callbacks ─────────────────────────────────────────────────────── #

    def _status_cb(self, msg: VehicleStatus):
        self.arming_state = msg.arming_state
        self.nav_state = msg.nav_state

    def _my_loc_cb(self, msg: VehicleLocalPosition):
        self.my_loc = msg

    def _my_glob_cb(self, msg: VehicleGlobalPosition):
        self.my_glob = msg

    def _tgt_loc_cb(self, msg: VehicleLocalPosition):
        # Hedefin dönüş hızını (course rate) filtreli türevle tahmin et
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

    # ── VehicleCommand ────────────────────────────────────────────────── #

    def _cmd(self, command, p1=0.0, p2=0.0):
        msg = VehicleCommand()
        msg.timestamp = self.now_us()
        msg.command = command
        msg.param1, msg.param2 = float(p1), float(p2)
        msg.target_system = self.sysid
        msg.target_component = 1
        msg.source_system = self.sysid
        msg.source_component = 1
        msg.from_external = True
        self.pub_cmd.publish(msg)

    # ── Alt sınıf sözleşmesi ──────────────────────────────────────────── #

    def publish_active(self, ts: int):
        """Hedef tazeyken kontrol yasası + setpoint yayını."""
        raise NotImplementedError

    def publish_idle(self, ts: int):
        """Beklerken/HOLD'da güvenli setpoint (heartbeat akışı için şart)."""
        raise NotImplementedError

    def target_fresh(self, now: float) -> bool:
        return (self.tgt_glob is not None and self.tgt_loc is not None
                and (now - self.tgt_glob_t) < self.tgt_timeout)

    # ── Ana döngü ─────────────────────────────────────────────────────── #

    def _publish_ocm(self, ts: int):
        ocm = OffboardControlMode()
        ocm.timestamp = ts
        for field, value in self.ocm_flags.items():
            setattr(ocm, field, bool(value))
        self.pub_ocm.publish(ocm)

    def _loop(self):
        now = self._now()
        ts = self.now_us()
        log = self.get_logger()

        my_ok = self.my_glob is not None and self.my_loc is not None
        tgt_fresh = self.target_fresh(now)

        is_armed = self.arming_state == VehicleStatus.ARMING_STATE_ARMED
        is_offboard = self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD
        agl = -float(self.my_loc.z) if self.my_loc is not None else 0.0
        airborne = agl > self.engage_alt

        self._publish_ocm(ts)

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
                self.state = S_ACTIVE
                log.info('OFFBOARD aktif → ACTIVE')
        elif self.state == S_ACTIVE:
            if not tgt_fresh:
                self.state = S_HOLD
                log.warn('Hedef verisi kesildi → HOLD')
            elif not is_offboard:
                self.state = S_ENGAGE
                log.warn('OFFBOARD kaybedildi (failsafe/mod değişimi) → ENGAGE')
        elif self.state == S_HOLD:
            if tgt_fresh:
                self.state = S_ACTIVE
                log.info('Hedef verisi geldi → ACTIVE')

        # ── Durum eylemleri ──────────────────────────────────────────── #
        if self.state == S_WAIT_DATA:
            self.publish_idle(ts)
            log.info(f'Veri bekleniyor: takipçi={my_ok} hedef={tgt_fresh}',
                     throttle_duration_sec=3.0)
            return

        if self.state == S_TAKEOFF:
            self.publish_idle(ts)
            if self.auto_arm_takeoff and now - self._last_cmd_t > 2.0:
                self._last_cmd_t = now
                if not is_armed:
                    self._cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
                              p1=1.0)
                    log.info('→ ARM')
                elif self.nav_state != VehicleStatus.NAVIGATION_STATE_AUTO_TAKEOFF:
                    self._cmd(VehicleCommand.VEHICLE_CMD_NAV_TAKEOFF)
                    log.info('→ TAKEOFF')
            elif not self.auto_arm_takeoff:
                log.info(f'Kalkış bekleniyor (AGL {agl:.0f}/{self.engage_alt:.0f} m)'
                         ' — auto_arm_takeoff kapalı, elle kaldırın',
                         throttle_duration_sec=5.0)
            return

        if self.state == S_ENGAGE:
            if tgt_fresh:
                self.publish_active(ts)
            else:
                self.publish_idle(ts)
            if self.auto_offboard and now - self._last_cmd_t > 1.0:
                self._last_cmd_t = now
                self._cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, p1=1.0, p2=6.0)
                log.info('→ OFFBOARD istendi')
            elif not self.auto_offboard:
                log.info('auto_offboard kapalı — OFFBOARD moduna elle geçin',
                         throttle_duration_sec=5.0)
            return

        if self.state == S_HOLD:
            self.publish_idle(ts)
            return

        # S_ACTIVE
        self.publish_active(ts)

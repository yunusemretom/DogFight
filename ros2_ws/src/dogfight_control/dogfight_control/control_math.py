#!/usr/bin/env python3
############################################################################
#
#   control_math — ROS'tan bağımsız saf kontrol matematiği.
#
#   Bu modül dogfight_control node'larının paylaştığı matematiği içerir ve
#   rclpy/SITL olmadan pytest ile birim testlenebilir (test/test_control_math.py).
#
#   Koordinat: NED (PX4 standardı) — x kuzey, y doğu, z aşağı (+).
#
############################################################################

import math

G = 9.80665
R_EARTH = 6371000.0


def wrap_pi(a: float) -> float:
    """Açıyı [-pi, pi] aralığına sar."""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def relative_ned(my_lat, my_lon, my_alt, tgt_lat, tgt_lon, tgt_alt):
    """Hedefin takipçiye göre NED pozisyonu [m] (flat-earth, <~5 km).

    Araçların local NED origin'leri farklı olduğundan local pozisyonlar
    doğrudan çıkarılamaz; bu yüzden VehicleGlobalPosition (derece, m AMSL)
    üzerinden hesaplanır. rel_d aşağı pozitiftir (hedef alçaktaysa +).
    """
    rel_n = math.radians(tgt_lat - my_lat) * R_EARTH
    rel_e = (math.radians(tgt_lon - my_lon) * R_EARTH
             * math.cos(math.radians(my_lat)))
    rel_d = my_alt - tgt_alt
    return rel_n, rel_e, rel_d


def euler_to_quaternion(roll: float, pitch: float, yaw: float):
    """roll/pitch/yaw [rad] → PX4 quaternion sırası [w, x, y, z]."""
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    return [cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy]


def clamp_speed_xy(vx: float, vy: float, vmin: float, vmax: float):
    """XY hız vektörünün büyüklüğünü [vmin, vmax] aralığına kırp.

    Sabit kanat stall koruması: sıfıra yakın komutlar vmin ileri hıza çekilir.
    """
    spd = math.hypot(vx, vy)
    if spd < 1e-6:
        return float(vmin), 0.0
    s = clamp(spd, vmin, vmax) / spd
    return vx * s, vy * s


def follow_point(rel_n, rel_e, tgt_vn, tgt_ve, standoff, tgt_omega=0.0):
    """Hedefin izlediği yolun standoff kadar gerisindeki takip noktası.

    Girdiler takipçi merkezli NED [m, m/s]. Hedef dönüyorsa nokta teğet
    üzerinde değil YAY üzerinde hesaplanır (l1_pursuit_node ile aynı mantık);
    aksi halde nokta dönüş dairesinin dışına düşer ve kalıcı mesafe sapması
    oluşur. Dönüş: (fp_n, fp_e).
    """
    dist_xy = math.hypot(rel_n, rel_e)
    tgt_spd = math.hypot(tgt_vn, tgt_ve)

    if tgt_spd > 3.0 and abs(tgt_omega) > 0.03:
        u_n, u_e = tgt_vn / tgt_spd, tgt_ve / tgt_spd
        r_turn = tgt_spd / abs(tgt_omega)
        sgn = 1.0 if tgt_omega > 0.0 else -1.0
        # Dönüş merkezi: hız vektörünün sgn*90° yanında
        c_n = rel_n + r_turn * (-sgn * u_e)
        c_e = rel_e + r_turn * (sgn * u_n)
        # Yayda standoff kadar geriye dön
        theta = -sgn * min(standoff / r_turn, math.pi / 2.0)
        dn, de = rel_n - c_n, rel_e - c_e
        fp_n = c_n + dn * math.cos(theta) - de * math.sin(theta)
        fp_e = c_e + dn * math.sin(theta) + de * math.cos(theta)
    elif tgt_spd > 3.0:
        fp_n = rel_n - tgt_vn / tgt_spd * standoff
        fp_e = rel_e - tgt_ve / tgt_spd * standoff
    elif dist_xy > 1.0:
        # Hedef duruyor: takipçi tarafındaki noktada bekle
        fp_n = rel_n - rel_n / dist_xy * standoff
        fp_e = rel_e - rel_e / dist_xy * standoff
    else:
        fp_n, fp_e = rel_n - standoff, rel_e
    return fp_n, fp_e


class PID:
    """Genel PID — integral sınırı, çıkış sınırı ve reset ile.

    Türev hata üzerinden alınır; dt<=0 çağrılarında türev/integral atlanır.
    """

    def __init__(self, kp, ki=0.0, kd=0.0, i_limit=None, out_limit=None):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.i_limit, self.out_limit = i_limit, out_limit
        self.integral = 0.0
        self._prev_error = None

    def reset(self):
        self.integral = 0.0
        self._prev_error = None

    def update(self, error: float, dt: float) -> float:
        derivative = 0.0
        if dt > 0.0:
            self.integral += error * dt
            if self.i_limit is not None:
                self.integral = clamp(self.integral, -self.i_limit, self.i_limit)
            if self._prev_error is not None:
                derivative = (error - self._prev_error) / dt
        self._prev_error = error

        out = self.kp * error + self.ki * self.integral + self.kd * derivative
        if self.out_limit is not None:
            out = clamp(out, -self.out_limit, self.out_limit)
        return out

#!/usr/bin/env python3
"""control_math birim testleri — ROS/SITL gerektirmez.

Çalıştırma:  pytest src/dogfight_control/test/test_control_math.py -v
"""

import math

import pytest

from dogfight_control.control_math import (PID, clamp, clamp_speed_xy,
                                           euler_to_quaternion, follow_point,
                                           relative_ned, wrap_pi)


# ── wrap_pi ─────────────────────────────────────────────────────────────── #

@pytest.mark.parametrize('inp,expected', [
    (0.0, 0.0),
    (math.pi / 2, math.pi / 2),
    (3 * math.pi / 2, -math.pi / 2),
    (-3 * math.pi / 2, math.pi / 2),
    (2 * math.pi, 0.0),
])
def test_wrap_pi(inp, expected):
    assert wrap_pi(inp) == pytest.approx(expected, abs=1e-9)


# ── relative_ned ────────────────────────────────────────────────────────── #

def test_relative_ned_north():
    # 0.001° enlem farkı ≈ 111.19 m kuzey
    n, e, d = relative_ned(47.0, 8.0, 100.0, 47.001, 8.0, 100.0)
    assert n == pytest.approx(111.19, rel=0.01)
    assert e == pytest.approx(0.0, abs=1e-6)
    assert d == pytest.approx(0.0, abs=1e-9)


def test_relative_ned_east_cos_scaled():
    # Boylam farkı cos(lat) ile ölçeklenir
    n, e, d = relative_ned(60.0, 8.0, 100.0, 60.0, 8.001, 100.0)
    assert n == pytest.approx(0.0, abs=1e-6)
    assert e == pytest.approx(111.19 * math.cos(math.radians(60.0)), rel=0.01)


def test_relative_ned_down_positive_when_target_lower():
    # NED: aşağı pozitif → hedef 20 m alçaktaysa rel_d = +20
    _, _, d = relative_ned(47.0, 8.0, 120.0, 47.0, 8.0, 100.0)
    assert d == pytest.approx(20.0)


# ── euler_to_quaternion ─────────────────────────────────────────────────── #

def test_quaternion_identity():
    q = euler_to_quaternion(0.0, 0.0, 0.0)
    assert q == pytest.approx([1.0, 0.0, 0.0, 0.0])


def test_quaternion_yaw_90():
    q = euler_to_quaternion(0.0, 0.0, math.pi / 2)
    s = math.sqrt(0.5)
    assert q == pytest.approx([s, 0.0, 0.0, s], abs=1e-9)


def test_quaternion_roll_90():
    q = euler_to_quaternion(math.pi / 2, 0.0, 0.0)
    s = math.sqrt(0.5)
    assert q == pytest.approx([s, s, 0.0, 0.0], abs=1e-9)


def test_quaternion_is_unit():
    q = euler_to_quaternion(0.3, -0.5, 2.1)
    assert sum(v * v for v in q) == pytest.approx(1.0, abs=1e-9)


# ── clamp / clamp_speed_xy ──────────────────────────────────────────────── #

def test_clamp():
    assert clamp(5.0, 0.0, 1.0) == 1.0
    assert clamp(-5.0, 0.0, 1.0) == 0.0
    assert clamp(0.5, 0.0, 1.0) == 0.5


def test_clamp_speed_scales_up_to_min():
    # Stall koruması: yavaş komut min hıza yükseltilir, yön korunur
    vx, vy = clamp_speed_xy(3.0, 4.0, 10.0, 20.0)  # |v| = 5
    assert math.hypot(vx, vy) == pytest.approx(10.0)
    assert vy / vx == pytest.approx(4.0 / 3.0)


def test_clamp_speed_scales_down_to_max():
    vx, vy = clamp_speed_xy(30.0, 40.0, 10.0, 20.0)  # |v| = 50
    assert math.hypot(vx, vy) == pytest.approx(20.0)


def test_clamp_speed_zero_input_gives_min_forward():
    vx, vy = clamp_speed_xy(0.0, 0.0, 10.0, 20.0)
    assert (vx, vy) == (10.0, 0.0)


def test_clamp_speed_within_band_unchanged():
    vx, vy = clamp_speed_xy(9.0, 12.0, 10.0, 20.0)  # |v| = 15
    assert (vx, vy) == pytest.approx((9.0, 12.0))


# ── follow_point ────────────────────────────────────────────────────────── #

def test_follow_point_straight_target():
    # Kuzeye 15 m/s giden, 100 m kuzeydeki hedef: nokta 20 m gerisinde
    fp_n, fp_e = follow_point(100.0, 0.0, 15.0, 0.0, standoff=20.0)
    assert fp_n == pytest.approx(80.0)
    assert fp_e == pytest.approx(0.0)


def test_follow_point_stationary_target_on_follower_side():
    # Duran hedef: nokta takipçi–hedef doğrusu üzerinde, standoff kadar beride
    fp_n, fp_e = follow_point(100.0, 0.0, 0.0, 0.0, standoff=20.0)
    assert fp_n == pytest.approx(80.0)
    assert fp_e == pytest.approx(0.0)


def test_follow_point_turning_target_stays_on_arc():
    # Dönen hedef: nokta dönüş dairesi ÜZERİNDE kalmalı (teğet üzerinde değil)
    tgt_spd, omega, standoff = 15.0, 0.15, 20.0   # r_turn = 100 m
    r_turn = tgt_spd / omega
    rel_n, rel_e = 100.0, 0.0
    tgt_vn, tgt_ve = tgt_spd, 0.0                 # kuzeye gidiyor, sağa dönüyor
    fp_n, fp_e = follow_point(rel_n, rel_e, tgt_vn, tgt_ve, standoff, omega)

    # Dönüş merkezi hız vektörünün 90° sağında (doğuda)
    c_n, c_e = rel_n, rel_e + r_turn
    dist_center = math.hypot(fp_n - c_n, fp_e - c_e)
    assert dist_center == pytest.approx(r_turn, rel=1e-6)

    # Nokta hedeften yaklaşık standoff uzakta (kiriş ≈ yay küçük açıda)
    chord = math.hypot(fp_n - rel_n, fp_e - rel_e)
    expected_chord = 2 * r_turn * math.sin(standoff / r_turn / 2)
    assert chord == pytest.approx(expected_chord, rel=1e-6)
    assert chord == pytest.approx(standoff, rel=0.05)

    # Sağa dönüşte (heading artar) geçmiş nokta geride ve hafif İÇERİDE
    # (dönüş merkezi tarafında, burada doğuda) kalır
    assert fp_n < rel_n
    assert fp_e > rel_e


# ── PID ─────────────────────────────────────────────────────────────────── #

def test_pid_pure_p():
    pid = PID(kp=2.0)
    assert pid.update(3.0, 0.02) == pytest.approx(6.0)


def test_pid_integral_accumulates_and_limits():
    pid = PID(kp=0.0, ki=1.0, i_limit=0.5)
    for _ in range(1000):
        out = pid.update(1.0, 0.01)   # sınırsız olsa integral 10'a çıkardı
    assert pid.integral == pytest.approx(0.5)
    assert out == pytest.approx(0.5)


def test_pid_output_limit():
    pid = PID(kp=100.0, out_limit=1.0)
    assert pid.update(5.0, 0.02) == 1.0
    assert pid.update(-5.0, 0.02) == -1.0


def test_pid_derivative_sign():
    # Hata artıyorsa türev katkısı pozitif
    pid = PID(kp=0.0, kd=1.0)
    pid.update(0.0, 0.02)
    out = pid.update(1.0, 0.02)
    assert out == pytest.approx(1.0 / 0.02)


def test_pid_first_call_no_derivative_kick():
    pid = PID(kp=0.0, kd=10.0)
    assert pid.update(100.0, 0.02) == 0.0


def test_pid_reset():
    pid = PID(kp=1.0, ki=1.0, kd=1.0)
    pid.update(5.0, 0.02)
    pid.reset()
    assert pid.integral == 0.0
    assert pid.update(1.0, 0.02) == pytest.approx(1.0 + 1.0 * 0.02 * 1.0)


def test_pid_zero_dt_safe():
    pid = PID(kp=1.0, ki=1.0, kd=1.0)
    assert pid.update(2.0, 0.0) == pytest.approx(2.0)  # sadece P katkısı

---
layout: page
title: "Offboard Kontrol Test Node'ları Rehberi"
description: "dogfight_control paketinin mimarisi: attitude/trajectory/visual offboard test node'ları, ortak taban, birim testler ve SITL sonuçları"
icon: "🛠️"
order: 6
---

Bu rehber, `dogfight_control` paketinin **2026-07-12 temiz yeniden yazımını** açıklar: neden yeniden yazıldı, yeni mimari nasıl çalışıyor, her node ne yapıyor ve SITL testlerinde ne ölçüldü.

- **Kod:** `ros2_ws/src/dogfight_control/dogfight_control/`
- **Amaç:** PX4 v1.16'nın farklı offboard arayüzlerini (**attitude setpoint**, **trajectory position/velocity**, **görsel → FW lateral/longitudinal**) hedef takibi senaryosunda ayrı ayrı, kontrollü test etmek.
- **Not:** Bunlar **test/deney node'larıdır**. Üretim takip kontrolcüsü `dogfight_tracking/l1_pursuit_node`'dur (bkz. [L1 Takip Rehberi](l1-takip-rehberi.html)).

---

## 1. Neden yeniden yazıldı?

Eski `dogfight_control` dosyaları deneysel karalamalardı ve kritik hatalar içeriyordu:

| Dosya | Hata |
|---|---|
| `attitude_controller_node.py` | PID çıkışı ±20°'ye sınırlanıp quaternion'a girerken `pitch*10` ile çarpılıyordu (limit fiilen ±200°). Status aboneliği yanlış topic'teydi (`_v1`; v1.16'da `_v4`) — node offboard durumunu hiç göremiyordu. Quaternion hep yaw=0 (kuzey) ile kuruluyordu. İki aracın local `z`'leri doğrudan çıkarılıyordu (çapraz-çerçeve hatası). |
| `velocity_controller_node.py` | Hedefin **pozisyonu** hız setpoint'i olarak yayınlanıyordu (`velocity = [tgt_x, tgt_y, tgt_z]` — hedef 500 m uzaktaysa 500 m/s komut!). |
| `position_controller_node.py` | Hedefin kendi NED çerçevesindeki pozisyonu takipçiye setpoint veriliyordu (araçların NED origin'leri farklı — yanlış noktaya uçar). Dosya başlığı bozuktu, kullanılmayan publisher'lar vardı. |
| `visual_offboard_node.py` | Kopyala-yapıştır hatası: `vel_z`'ye yanlışlıkla `vel_x` yazılıyordu. Dikey görüntü hatası ileri hıza bağlanmıştı. Tespit kesilince (0,0,0) hız komutu = sabit kanatta stall. |
| Genel | Hepsi `minimal_publisher` adını kullanıyordu (aynı anda iki node çalışamaz), namespace'ler koda gömülüydü, hiçbir güvenlik katı ve test yoktu. |

Eski dosyalar silindi; `fw_pursuit_node.py` (l1 ile değiştirilen eski pure-pursuit) `archive/` klasörüne taşındı.

---

## 2. Yeni mimari

```
dogfight_control/
├── control_math.py             # Saf matematik — ROS'suz, pytest ile test edilir
├── offboard_base.py            # Ortak taban sınıfı (OffboardTestBase)
├── attitude_setpoint_node.py   # VehicleAttitudeSetpoint testi
├── trajectory_velocity_node.py # TrajectorySetpoint.velocity testi
├── trajectory_position_node.py # TrajectorySetpoint.position testi
├── visual_offboard_node.py     # Görsel tespit → FW lateral/longitudinal
└── px4_status_monitor.py       # Salt-okunur durum monitörü
test/
├── test_control_math.py        # 27 birim test
└── test_nodes_smoke.py         # 5 smoke test (SITL gerektirmez)
```

### 2.1 `control_math.py` — saf matematik katmanı

ROS'a hiç bağımlı olmayan fonksiyonlar; böylece kontrol matematiği SITL olmadan birim testlenebilir:

- `relative_ned(my_lat, my_lon, my_alt, tgt_lat, ...)` — hedefin takipçiye göre NED pozisyonu. **Araçların local NED origin'leri farklı olduğundan** pozisyonlar her zaman `VehicleGlobalPosition` üzerinden karşılaştırılır (flat-earth, <~5 km).
- `follow_point(...)` — hedefin yolunun `standoff` kadar gerisindeki takip noktası. Hedef dönüyorsa nokta teğet üzerinde değil **yay üzerinde** hesaplanır (l1_pursuit_node ile aynı mantık; teğet hesabı kalıcı mesafe sapması yaratır).
- `PID` — integral sınırı, çıkış sınırı, reset ve ilk-çağrıda türev tekmesi koruması olan genel PID.
- `euler_to_quaternion`, `clamp_speed_xy` (stall koruması: XY hız min'in altına düşürülmez), `wrap_pi`, `clamp`.

### 2.2 `offboard_base.py` — ortak taban

Tüm test node'ları `OffboardTestBase`'den türer. Taban şunları tek yerde çözer:

1. **v1.16 versiyonlu topic'ler:** her aboneliği `base`, `_v1`, `_v2`, `_v4` varyantlarının dördüne birden açar (tuzak: v1.16'da `vehicle_status` → `vehicle_status_v4`, `vehicle_local_position` → `_v1`).
2. **Çerçeve-güvenli bağıl konum:** `rel_ned()` global pozisyonlardan hesaplar; hızlar local NED'den alınır (NED eksenleri yerel olarak paraleldir).
3. **Durum makinesi** (l1_pursuit_node'daki kanıtlanmış akış):

   ```
   WAIT_DATA → TAKEOFF → ENGAGE → ACTIVE
                                    ↕ (hedef timeout / offboard kaybı)
                                  HOLD / ENGAGE
   ```

   - `WAIT_DATA`: takipçi + hedef verisi bekler, bu sırada offboard heartbeat + idle setpoint akıtır (PX4'ün offboard'u kabul etmesi için akış şarttır).
   - `TAKEOFF`: `auto_arm_takeoff=true` ise ARM + NAV_TAKEOFF komutları; `engage_altitude` (varsayılan 15 m AGL) aşılınca ENGAGE.
   - `ENGAGE`: `auto_offboard=true` ise DO_SET_MODE(offboard) ister.
   - `ACTIVE`: alt sınıfın kontrol yasası (`publish_active`).
   - `HOLD`: hedef verisi `target_timeout`'u (2 s) aşınca güvenli idle setpoint (`publish_idle`); veri dönünce ACTIVE'e döner.
4. **Hedef dönüş hızı tahmini** (`tgt_omega`): filtreli course-rate türevi; `follow_point`'in yay hesabında kullanılır.
5. **Ortak parametreler:** `follower_ns` (varsayılan `''` = `/fmu`), `target_ns` (varsayılan `/px4_1`), `target_timeout`, `auto_arm_takeoff`, `auto_offboard`, `engage_altitude`, `system_id` (0 = namespace'ten otomatik türet).

Alt sınıf sözleşmesi: `ocm_flags` (OffboardControlMode alanları) + `publish_active(ts)` + `publish_idle(ts)`; gerekirse `target_fresh(now)` override edilir (görsel node bunu yapar).

### 2.3 Node'lar ve kontrol yasaları

**`attitude_setpoint_node`** — `OffboardControlMode.attitude=true` + `VehicleAttitudeSetpoint` (hem versiyonsuz hem `_v1` topic'ine yayınlar):

```
roll   = clamp(kp_roll · heading_hatası, ±max_roll)      # hedefe dön
pitch  = PID(irtifa_hatası)                              # global alt farkından, ±20°
thrust = clamp(nominal + PID(mesafe − standoff), min..max) + low-pass
yaw    = mevcut heading                                  # yaw=0 sabitlemek roll ile kuplaj yaratır
```

⚠ Attitude offboard **TECS'i devre dışı bırakır** — stall/irtifa koruması yoktur; yalnızca SITL'de ve irtifada test edin.

**`trajectory_velocity_node`** — `velocity=true` + `TrajectorySetpoint.velocity`:

```
v_xy = clamp_speed(hedef_hızı + kp_approach · follow_point, min_speed..max_speed)
vz   = clamp(−kp_alt · irtifa_hatası, ±max_vz)
yaw  = hız vektörünün yönü
```

**`trajectory_position_node`** — `position=true`; takip noktası takipçinin local NED'ine taşınarak (kendi konumu + bağıl vektör) pozisyon setpoint'i verilir, diğer alanlar NaN.

⚠ Her iki trajectory node'u için: TrajectorySetpoint yayınlanınca PX4'ün **FixedWingModeManager**'ı devreye girer ve kendi guidance'ını işletir (PROJE_OZETI tuzak #6). Bu node'lar **davranış gözlemi içindir**, hassas takip için değil — ölçümler §4'te.

**`visual_offboard_node`** — girdi `/yolo/target_distance` (`geometry_msgs/Point`: x = yatay px sapma, y = dikey px sapma, z = güven). Çıkış, l1 ile aynı kanıtlanmış FW reçetesi (`velocity=true` heartbeat, TrajectorySetpoint YOK):

```
FixedWingLateralSetpoint.lateral_acceleration = clamp(kp_lateral · px_x, ±8 m/s²)
FixedWingLongitudinalSetpoint.height_rate     = clamp(−kp_height_rate · px_y, ±3 m/s)
FixedWingLongitudinalSetpoint.equivalent_airspeed = cruise_airspeed
```

Tespit kesilince HOLD: kanat düz + irtifa koru + cruise hızda uçuş sürer (**stall komutu yok**).

**`px4_status_monitor`** — eski `deneme.py`'nin düzenlenmişi; konum/hız/attitude/airspeed/durum loglar, hiçbir şey yayınlamaz. `vehicle_ns` ve `log_period` parametreli.

---

## 3. Kullanım

```bash
cd ros2_ws && colcon build --packages-select dogfight_control   # symlink-install KULLANMA
source install/setup.bash

# Birim testler (SITL gerekmez; anyio eklenti çakışması için autoload kapalı)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest src/dogfight_control/test/ -q

# Node'lar (varsayılan: takipçi /fmu, hedef /px4_1)
ros2 run dogfight_control attitude_setpoint_node --ros-args -p standoff_distance:=20.0
ros2 run dogfight_control trajectory_velocity_node
ros2 run dogfight_control trajectory_position_node
ros2 run dogfight_control visual_offboard_node --ros-args -p detection_topic:=/yolo/target_distance
ros2 run dogfight_control px4_status_monitor --ros-args -p vehicle_ns:=/px4_1

# Farklı araç düzeni için:
#   -p follower_ns:=/px4_2 -p target_ns:=/px4_3
```

### SITL test ortamı notları (2026-07-12'de saptandı)

- **ROS'tan `VEHICLE_CMD_NAV_TAKEOFF` SITL'de kalkış tetiklemiyor** (ARM ve DO_SET_MODE işliyor). Kalkışı elle verin: `px4-commander --instance N takeoff` — node 15 m'de offboard'u kendisi devralır.
- **rc_cessna Hold/loiter modunda spiralle yere iniyor.** Dönen hedef gerekiyorsa hedef aracı da offboard'da uçurun: sabit `lateral_acceleration=3 m/s²` + sabit irtifa/airspeed → ~17° bank ile kararlı daire.
- **Spawn pozisyonları pist üzerinde olmalı** (`-5,10` ve `5,-10`, yaw 0.698). Pist dışı spawn (ör. 20,20) kalkış koşusunda uçağı deviriyor.
- **Failsafe paramları kalıcı değil:** her SITL kurulumunda iki instance'ta da `NAV_DLL_ACT=0`, `COM_RCL_EXCEPT=4` set edin (yalnızca SITL; gerçek uçuşta geri alınır).

---

## 4. SITL test sonuçları

Ortam: Gazebo Harmonic + PX4 v1.16 SITL, 2 × rc_cessna. Hedef ~55 m AGL'de 3 m/s² yanal ivmeyle daire çizerken (offboard feeder), takipçi node'ları sırayla koşuldu.

| Node | Sonuç | Yorum |
|---|---|---|
| `attitude_setpoint_node` | 20 m hedefte **ort. 20.5 m, std 4.2 m** (son 60 s); irtifa \|hata\| ort. **0.48 m** | ✅ Çalışıyor. Salınım (14–28 m) thrust-tabanlı mesafe kontrolünün doğası; hassas mesafe için l1/formasyon modu üstün. |
| `trajectory_velocity_node` | Komut 30 m standoff isterken mesafe **~4 m'ye kilitlendi** (min 2.9); komutlar hedeften uzağa işaret ederken bile değişmedi | ⚠ Tuzak #6 ölçüldü: FW mode manager velocity setpoint'i kendi guidance'ıyla eziyor. Gerçek uçuşta bu davranış çarpışma riskidir. |
| `trajectory_position_node` | 30 m hedefe karşı mesafe **~75 m'ye açıldı** (60–91 m salınım) | ⚠ Hareketli position setpoint loiter benzeri gecikmeli işleniyor; hareketli hedef takibi için elverişsiz. |
| `visual_offboard_node` | Sahte tespit (+100 px sağ, −50 px yukarı) → **+5 m/s² yanal** (uçak +24° roll'a yattı), **+1 m/s tırmanma**; tespit kesilince temiz **HOLD** geçişi | ✅ Eksen eşlemesi ve timeout davranışı doğru. Gerçek tespit zinciriyle (RF-DETR/YOLO) uçtan uca test sıradaki adım. |

**Çıkarım:** Sabit kanatta hedef takibi için kullanılabilir offboard arayüzleri **attitude setpoint** (TECS'siz, dikkatli) ve **FW lateral/longitudinal** (önerilen — l1_pursuit_node'un kullandığı); `TrajectorySetpoint` position/velocity artık ölçülmüş şekilde elverişsizdir.

---

## 5. Güvenlik notları

- Bu node'larda l1_pursuit_node'daki `min_separation` katı **yoktur** — yalnızca hedef-timeout→HOLD vardır. Yakın mesafe denemelerini yalnızca SITL'de yapın.
- Gerçek uçuş öncesi: `auto_arm_takeoff:=false auto_offboard:=false` verin (pilot devralma prosedürü), SITL'de kapatılan failsafe paramlarını geri alın.
- `attitude_setpoint_node` thrust'ı doğrudan sürer; `min_thrust`/`max_thrust` sınırlarını gerçek uçağın zarfına göre daraltın.

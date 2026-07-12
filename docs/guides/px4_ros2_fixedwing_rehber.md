# PX4 + ROS 2 ile Fixed-Wing (Sabit Kanat) Araç Kontrolü — Kapsamlı Rehber

> Kaynak: [docs.px4.io/main/en/robotics](https://docs.px4.io/main/en/robotics/) · Güncel PX4 v1.17 / ROS 2 Humble

---

## 1. Genel Mimari: PX4 ↔ ROS 2 Nasıl Haberleşir?

```
[PX4 Firmware]
   uORB topic'leri
       │
   uXRCE-DDS Client  (PX4 içinde çalışır)
       │  (UDP/Serial)
   MicroXRCE-DDS Agent  (companion computer'da çalışır)
       │
   ROS 2 Graph  (/fmu/out/... ve /fmu/in/... topic'leri)
       │
   Kendi ROS 2 Node'larınız
```

ROS 2 ile PX4 arasındaki iletişim **uXRCE-DDS** (eski adıyla micro-RTPS) üzerinden gerçekleşir.
PX4 içindeki uORB mesajları doğrudan ROS 2 topic'lerine dönüştürülür. Bu sayede:
- **`/fmu/out/...`** → PX4'ün yayınladığı (subscribe edilecek) topic'ler
- **`/fmu/in/...`** → ROS 2'nin PX4'e gönderdiği (publish edilecek) topic'ler

**Önerilen platform:** ROS 2 Humble LTS + Ubuntu 22.04

---

## 2. Kurulum Adımları

### 2.1 uXRCE-DDS Agent Başlatma
```bash
# Agent'ı derle veya snap ile kur
MicroXRCEAgent udp4 -p 8888
```

### 2.2 PX4 Tarafında Client Başlatma (SITL için otomatik, hardware için manuel)
```bash
# PX4 shell üzerinden:
uxrce_dds_client start -t udp -h 127.0.0.1 -p 8888
```

### 2.3 ROS 2 Workspace Kurulumu
```bash
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src
git clone https://github.com/PX4/px4_msgs.git   # PX4 sürümünüzle eşleşen branch'i seçin
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select px4_msgs
source install/setup.bash
```

> ⚠️ **Kritik:** `px4_msgs` branch'i PX4 firmware sürümünüzle **birebir** uyuşmalıdır.
> PX4 v1.16+ ile **Message Translation Node** gerekebilir.

---

## 3. QoS Ayarı (Zorunlu!)

PX4, topic'lerini **best-effort** QoS ile yayınlar. Subscriber'ınız bunu match etmezse veri alamazsınız:

```python
# Python
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
qos = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1
)
```

```cpp
// C++
rclcpp::QoS(10).best_effort()
```

---

## 4. Fixed-Wing için ROS 2'den Veri OKUMA (Subscribe)

Tüm topic'ler `/fmu/out/` prefix'i ile gelir.

### 4.1 Konum Verisi

#### `VehicleLocalPosition` — NED Yerel Konum
**Topic:** `/fmu/out/vehicle_local_position`

| Alan | Tip | Açıklama |
|------|-----|----------|
| `x` | float32 | Kuzey (m) — EKF2 başlangıcına göre |
| `y` | float32 | Doğu (m) |
| `z` | float32 | Aşağı (m) — negatif = yüksek |
| `vx` | float32 | Kuzey hızı (m/s) |
| `vy` | float32 | Doğu hızı (m/s) |
| `vz` | float32 | Aşağı hız (m/s) |
| `ax` | float32 | Kuzey ivmesi (m/s²) |
| `heading` | float32 | Kurs açısı (rad, NED'e göre) |
| `xy_valid` | bool | XY konum geçerli mi? |
| `z_valid` | bool | Z konum geçerli mi? |
| `v_xy_valid` | bool | XY hız geçerli mi? |

#### `VehicleGlobalPosition` — GPS/WGS84 Global Konum
**Topic:** `/fmu/out/vehicle_global_position`

| Alan | Tip | Açıklama |
|------|-----|----------|
| `lat` | float64 | Enlem (derece) |
| `lon` | float64 | Boylam (derece) |
| `alt` | float32 | MSL irtifa (m) |
| `alt_ellipsoid` | float32 | Elipsoid irtifa (m) |
| `terrain_alt` | float32 | Arazi irtifası (m) |

### 4.2 Yön / Attitude Verisi

#### `VehicleAttitude` — Quaternion Orientation
**Topic:** `/fmu/out/vehicle_attitude`

| Alan | Tip | Açıklama |
|------|-----|----------|
| `q[4]` | float32[] | Quaternion [w, x, y, z] (Hamilton convention) |
| `delta_q_reset[4]` | float32[] | Quaternion reset delta |

> **Quaternion'dan Euler açısına çevirme:**
> ```python
> import math
> def q_to_euler(q):
>     w, x, y, z = q[0], q[1], q[2], q[3]
>     # Roll (x-axis rotation)
>     roll  = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
>     # Pitch (y-axis rotation)
>     pitch = math.asin(max(-1, min(1, 2*(w*y - z*x))))
>     # Yaw (z-axis rotation)
>     yaw   = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
>     return roll, pitch, yaw  # radyan
> ```

#### `VehicleAngularVelocity` — Açısal Hız
**Topic:** `/fmu/out/vehicle_angular_velocity`

| Alan | Tip | Açıklama |
|------|-----|----------|
| `xyz[3]` | float32[] | [roll_rate, pitch_rate, yaw_rate] (rad/s) |

### 4.3 Hava Hızı (Fixed-Wing için Kritik!)

#### `AirspeedValidated` — Doğrulanmış Hava Hızı
**Topic:** `/fmu/out/airspeed_validated`

| Alan | Tip | Açıklama |
|------|-----|----------|
| `calibrated_airspeed_m_s` | float32 | Kalibre hava hızı (m/s) |
| `true_airspeed_m_s` | float32 | Gerçek hava hızı (m/s) |
| `indicated_airspeed_m_s` | float32 | Gösterge hava hızı (m/s) |
| `airspeed_sensor_measurement_valid` | bool | Sensör geçerli mi? |

### 4.4 Durum Bilgisi

#### `VehicleStatus` — Araç Durumu
**Topic:** `/fmu/out/vehicle_status`

| Alan | Tip | Açıklama |
|------|-----|----------|
| `arming_state` | uint8 | 1=STANDBY, 2=ARMED |
| `nav_state` | uint8 | Aktif uçuş modu |
| `vehicle_type` | uint8 | 1=ROTARY, 2=FIXED_WING, 3=ROVER |
| `failsafe` | bool | Failsafe aktif mi? |

---

## 5. Fixed-Wing için Kontrol Komutları (Publish)

### 5.1 Yöntem A — PX4 ROS 2 Interface Library (ÖNERİLEN, v1.17+)

Bu yöntem, fixed-wing aracı en temiz şekilde kontrol etmenizi sağlar. PX4 v1.17 ile **`FwLateralLongitudinalSetpointType`** eklendi.

#### `FixedWingLateralSetpoint` — Yanal Kontrol
**Topic:** `/fmu/in/fixed_wing_lateral_setpoint`

| Alan | Tip | Açıklama |
|------|-----|----------|
| `course` | float32 | Hedef kurs açısı (rad, NaN = kontrol etme) |
| `airspeed_direction` | float32 | Hava hızı yönü (rad, NaN = kontrol etme) |
| `lateral_acceleration` | float32 | Yanal ivme komutu (m/s², NaN = kontrol etme) |

> Lateral için en az **bir** alan NaN olmayan değer içermelidir.

#### `FixedWingLongitudinalSetpoint` — Boylamasına Kontrol
**Topic:** `/fmu/in/fixed_wing_longitudinal_setpoint`

| Alan | Tip | Açıklama |
|------|-----|----------|
| `altitude` | float32 | Hedef irtifa (m MSL, NaN = kontrol etme) |
| `height_rate` | float32 | Tırmanma hızı (m/s, NaN = kontrol etme) |
| `equivalent_airspeed_sp` | float32 | Hedef hava hızı (m/s, NaN = kontrol etme) |
| `pitch_direct` | float32 | Direkt pitch açısı (rad, NaN = kontrol etme) |
| `throttle_direct` | float32 | Direkt gaz (0.0–1.0, NaN = kontrol etme) |

> Longitudinal için `altitude` veya `height_rate`'den en az biri dolu olmalı.

### 5.2 Yöntem B — Offboard Mod ile Doğrudan topic Yayınlama

#### Adım 1: OffboardControlMode Yayınla
**Topic:** `/fmu/in/offboard_control_mode`

```python
from px4_msgs.msg import OffboardControlMode

msg = OffboardControlMode()
msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
msg.position = True    # pozisyon kontrolü aktif
msg.velocity = False
msg.acceleration = False
msg.attitude = False
msg.body_rate = False
```

#### Adım 2: Setpoint Yayınla

**TrajectorySetpoint** — NED Pozisyon/Hız Hedefi
**Topic:** `/fmu/in/trajectory_setpoint`

```python
from px4_msgs.msg import TrajectorySetpoint
import math

msg = TrajectorySetpoint()
msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
# Pozisyon (NED, metre) — NaN = kontrol etme
msg.position = [100.0, 50.0, -80.0]   # 80m yükseklikte
# Hız (NED, m/s) — NaN = kontrol etme
msg.velocity = [float('nan')] * 3
# Yaw (NED'e göre, radyan)
msg.yaw = math.radians(90)             # Doğuya bak
```

#### Adım 3: VehicleCommand ile Offboard Mod Aktif Et

```python
from px4_msgs.msg import VehicleCommand

def arm_and_set_offboard(self):
    # Offboard mod aktifleştir
    cmd = VehicleCommand()
    cmd.command = VehicleCommand.VEHICLE_CMD_DO_SET_MODE
    cmd.param1 = 1.0
    cmd.param2 = 6.0  # PX4_CUSTOM_MAIN_MODE_OFFBOARD
    cmd.target_system = 1
    cmd.target_component = 1
    cmd.source_system = 1
    cmd.source_component = 1
    cmd.from_external = True
    cmd.timestamp = int(self.get_clock().now().nanoseconds / 1000)
    self.vehicle_command_pub.publish(cmd)
    
    # ARM
    cmd.command = VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
    cmd.param1 = 1.0  # arm
    self.vehicle_command_pub.publish(cmd)
```

> ⚠️ **Önemli:** Offboard moda geçmeden önce en az **10 setpoint** gönderilmelidir, aksi halde mod reddedilir.

### 5.3 Yöntem C — Attitude Setpoint (Açı Kontrolü)

**VehicleAttitudeSetpoint** — Doğrudan Açı Komutu
**Topic:** `/fmu/in/vehicle_attitude_setpoint`

```python
from px4_msgs.msg import VehicleAttitudeSetpoint
import math

msg = VehicleAttitudeSetpoint()
msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
# Hedef yaw (radyan)
msg.yaw_sp_move_rate = 0.0
msg.q_d = [1.0, 0.0, 0.0, 0.0]  # Quaternion hedef
msg.thrust_body = [0.0, 0.0, -0.7]  # Gaz %70
```

### 5.4 Yöntem D — Rate Setpoint (En Alt Seviye)

**VehicleRatesSetpoint** — Roll/Pitch/Yaw Rate Komutu
**Topic:** `/fmu/in/vehicle_rates_setpoint`

```python
from px4_msgs.msg import VehicleRatesSetpoint

msg = VehicleRatesSetpoint()
msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
msg.roll  = 0.0    # rad/s
msg.pitch = 0.05   # rad/s — hafif nose-up
msg.yaw   = 0.0    # rad/s
msg.thrust_body = [0.0, 0.0, -0.8]  # %80 gaz
```

---

## 6. Fixed-Wing'e Özel Kavramlar

### 6.1 Lateral (Yanal) vs Longitudinal (Boylamasına) Ayrımı

Fixed-wing araçlar, multikopterden farklı olarak iki bağımsız kontrol ekseni üzerinden komutlanır:

```
LATERAL (Yanal):          LONGITUDINAL (Boylamasına):
  - Dönüş / banking          - İrtifa
  - Kurs açısı               - Tırmanma hızı
  - Yanal ivme               - Hava hızı
  - Aileron/rudder ile       - Elevator/throttle ile
```

### 6.2 Koordinat Sistemi (NED)

```
     N (Kuzey / +X)
     │
W ───┼─── E (Doğu / +Y)
     │
     S

Z ekseni: AŞAĞI pozitif
Yani yükseklik = -z (negatif z değeri)
```

### 6.3 Fixed-Wing için Kritik Parametreler

| Parametre | Açıklama | Tipik Değer |
|-----------|----------|-------------|
| `FW_AIRSPD_MIN` | Minimum hava hızı | 10-15 m/s |
| `FW_AIRSPD_MAX` | Maksimum hava hızı | 25-30 m/s |
| `FW_AIRSPD_TRIM` | Nominal hava hızı | 15-20 m/s |
| `FW_L1_PERIOD` | L1 rehberlik periyodu | 20 s |
| `FW_THR_CRUISE` | Cruise gazı | 0.6 |

---

## 7. Eksiksiz Python Node Örneği — FW Durum Okuyucu

```python
#!/usr/bin/env python3
"""
PX4 Fixed-Wing Durum Monitörü
Konum, hız, yön ve hava hızı verilerini okur.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import math

from px4_msgs.msg import (
    VehicleLocalPosition,
    VehicleGlobalPosition,
    VehicleAttitude,
    VehicleAngularVelocity,
    AirspeedValidated,
    VehicleStatus,
)

class FWStatusMonitor(Node):
    def __init__(self):
        super().__init__('fw_status_monitor')

        # PX4 best-effort QoS
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscriber'lar
        self.create_subscription(VehicleLocalPosition,
            '/fmu/out/vehicle_local_position',
            self.local_pos_cb, qos)

        self.create_subscription(VehicleGlobalPosition,
            '/fmu/out/vehicle_global_position',
            self.global_pos_cb, qos)

        self.create_subscription(VehicleAttitude,
            '/fmu/out/vehicle_attitude',
            self.attitude_cb, qos)

        self.create_subscription(VehicleAngularVelocity,
            '/fmu/out/vehicle_angular_velocity',
            self.ang_vel_cb, qos)

        self.create_subscription(AirspeedValidated,
            '/fmu/out/airspeed_validated',
            self.airspeed_cb, qos)

        self.create_subscription(VehicleStatus,
            '/fmu/out/vehicle_status',
            self.status_cb, qos)

    def local_pos_cb(self, msg):
        altitude = -msg.z  # NED'de z negatif = yukarı
        speed_2d = math.sqrt(msg.vx**2 + msg.vy**2)
        self.get_logger().info(
            f'Konum NED: x={msg.x:.1f}m  y={msg.y:.1f}m  alt={altitude:.1f}m | '
            f'Hız: vx={msg.vx:.1f}  vy={msg.vy:.1f}  vz={msg.vz:.1f} m/s | '
            f'Yatay hız: {speed_2d:.1f} m/s'
        )

    def global_pos_cb(self, msg):
        self.get_logger().info(
            f'GPS: lat={msg.lat:.6f}°  lon={msg.lon:.6f}°  alt={msg.alt:.1f}m MSL'
        )

    def attitude_cb(self, msg):
        q = msg.q
        roll, pitch, yaw = self._q_to_euler(q)
        self.get_logger().info(
            f'Attitude: roll={math.degrees(roll):.1f}°  '
            f'pitch={math.degrees(pitch):.1f}°  '
            f'yaw={math.degrees(yaw):.1f}°'
        )

    def ang_vel_cb(self, msg):
        self.get_logger().info(
            f'Angular vel: p={math.degrees(msg.xyz[0]):.1f}°/s  '
            f'q={math.degrees(msg.xyz[1]):.1f}°/s  '
            f'r={math.degrees(msg.xyz[2]):.1f}°/s'
        )

    def airspeed_cb(self, msg):
        self.get_logger().info(
            f'Hava hızı: CAS={msg.calibrated_airspeed_m_s:.1f} m/s  '
            f'TAS={msg.true_airspeed_m_s:.1f} m/s  '
            f'Geçerli={msg.airspeed_sensor_measurement_valid}'
        )

    def status_cb(self, msg):
        armed = (msg.arming_state == 2)
        self.get_logger().info(
            f'Durum: armed={armed}  nav_state={msg.nav_state}  '
            f'vehicle_type={msg.vehicle_type}  failsafe={msg.failsafe}'
        )

    @staticmethod
    def _q_to_euler(q):
        w, x, y, z = q[0], q[1], q[2], q[3]
        roll  = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        pitch = math.asin(max(-1, min(1, 2*(w*y - z*x))))
        yaw   = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        return roll, pitch, yaw


def main():
    rclpy.init()
    node = FWStatusMonitor()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

---

## 8. Topic Özet Tablosu

### Okuma (Subscribe) — `/fmu/out/...`

| Topic Adı | Mesaj Tipi | Ne Verir? |
|-----------|-----------|-----------|
| `vehicle_local_position` | `VehicleLocalPosition` | x/y/z NED konum, vx/vy/vz hız |
| `vehicle_global_position` | `VehicleGlobalPosition` | lat/lon/alt GPS konumu |
| `vehicle_attitude` | `VehicleAttitude` | Quaternion yönelim (roll/pitch/yaw) |
| `vehicle_angular_velocity` | `VehicleAngularVelocity` | Roll/pitch/yaw rate (rad/s) |
| `airspeed_validated` | `AirspeedValidated` | CAS, TAS hava hızı |
| `vehicle_status` | `VehicleStatus` | Arming, mod, failsafe durumu |
| `sensor_combined` | `SensorCombined` | Ham IMU verileri |
| `vehicle_odometry` | `VehicleOdometry` | Pozisyon + hız + quaternion birlikte |

### Yazma (Publish) — `/fmu/in/...`

| Topic Adı | Mesaj Tipi | Ne Yapar? |
|-----------|-----------|-----------|
| `offboard_control_mode` | `OffboardControlMode` | Hangi kontrol tipinin aktif olduğunu bildirir |
| `trajectory_setpoint` | `TrajectorySetpoint` | NED pozisyon/hız/yaw hedefi |
| `vehicle_attitude_setpoint` | `VehicleAttitudeSetpoint` | Doğrudan açı (quaternion) komutu |
| `vehicle_rates_setpoint` | `VehicleRatesSetpoint` | Doğrudan angular rate komutu |
| `fixed_wing_lateral_setpoint` | `FixedWingLateralSetpoint` | FW yanal kontrol (v1.17+) |
| `fixed_wing_longitudinal_setpoint` | `FixedWingLongitudinalSetpoint` | FW boylamasına kontrol (v1.17+) |
| `vehicle_command` | `VehicleCommand` | Arm/disarm, mod değiştirme, vs. |

---

## 9. Kontrol Seviyeleri — Hangi Yöntemi Kullanmalısın?

```
YÜK SEVİYESİ        YÖNTEM                          FIXED-WING DESTEĞİ
─────────────────────────────────────────────────────────────────────────
En Yüksek   → FwLateralLongitudinalSetpointType   ✅ v1.17+ (önerilen)
Seviye          (px4_ros2_interface_lib)

             → TrajectorySetpoint + OffboardMode   ✅ Tüm sürümler
               (NED pozisyon/hız hedefleri)

             → VehicleAttitudeSetpoint             ✅ Tüm sürümler
               (Quaternion hedef)

En Alt      → VehicleRatesSetpoint                ✅ Tüm sürümler
Seviye        (Angular rate doğrudan komutu)

             → DirectActuatorsSetpointType         ✅ Motor/servo direkt
               (px4_ros2_interface_lib)
```

---

## 10. TEKNOFEST / Simulation Notları

SITL'de fixed-wing test ederken kullanabileceğin Gazebo modeli:
```bash
PX4_SYS_AUTOSTART=4004 PX4_GZ_MODEL=rc_cessna ./build/px4_sitl_default/bin/px4
```

Çok araçlı senaryoda topic namespace'leri değişir:
```
/px4_1/fmu/out/vehicle_local_position   # 1. araç
/px4_2/fmu/out/vehicle_local_position   # 2. araç
```

Topic listesini görmek için:
```bash
ros2 topic list | grep fmu
ros2 topic echo /fmu/out/vehicle_attitude --no-arr
```

---

*Referanslar: [PX4 ROS 2 User Guide](https://docs.px4.io/main/en/ros2/user_guide) · [Control Interface](https://docs.px4.io/main/en/ros2/px4_ros2_control_interface) · [uORB Msg Reference](https://docs.px4.io/main/en/msg_docs/) · [dds_topics.yaml](https://docs.px4.io/main/en/middleware/dds_topics)*

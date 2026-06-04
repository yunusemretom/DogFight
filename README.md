# 🐶✈️ DogFight — PX4 + ROS 2 ile RC Uçak Hedef Takip Sistemi

> **TEKNOFEST** yarışmasına yönelik, **ROS 2 Humble** ve **PX4 Autopilot** tabanlı, yapay zeka destekli hedef takip ve dogfight (hava muharebesi) platformu.

RC uçağımızı otonom olarak kontrol edip, GPS ve görüntü işleme verilerine dayanarak rakip aracı takip etmeyi hedefleyen bu proje; nesne tespiti (YOLO, RF-DETR), görsel takip (TCTrack), GPS tabanlı konumlandırma ve offboard uçuş kontrolünü tek bir çatı altında birleştirir.

---

## 📐 Sistem Mimarisi

```
┌─────────────────────────────────────────────────────────────────┐
│                        DogFight Sistemi                         │
├─────────────────┬──────────────────┬────────────────────────────┤
│   📸 Tespit     │   🎯 Takip       │   🕹️ Kontrol              │
│ (Detection)     │ (Tracking)       │ (Control)                  │
├─────────────────┼──────────────────┼────────────────────────────┤
│ YOLO Node       │ GPS Tracker Node │ Attitude Controller Node   │
│ RF-DETR Node    │ Visual Tracker   │ Velocity Controller Node   │
│                 │ Node             │ Position Controller Node   │
│                 │                  │ Visual Offboard Node       │
└────────┬────────┴────────┬─────────┴──────────────┬─────────────┘
         │                 │                        │
         ▼                 ▼                        ▼
    /detection/       /px4_X/fmu/out/         /px4_1/fmu/in/
    target_distance   vehicle_gps_position    offboard_control_mode
                                              trajectory_setpoint
                                              vehicle_attitude_setpoint
```

### ROS 2 Topic'leri

| Topic | Mesaj Tipi | Açıklama |
|-------|-----------|----------|
| `/yolo/target_distance` | `geometry_msgs/Point` | YOLO hedef merkez sapması (dx, dy, confidence) |
| `/px4_1/fmu/out/vehicle_gps_position` | `px4_msgs/SensorGps` | Araç 1 GPS konumu |
| `/px4_3/fmu/out/vehicle_gps_position` | `px4_msgs/SensorGps` | Araç 2 GPS konumu |
| `/px4_1/fmu/in/offboard_control_mode` | `px4_msgs/OffboardControlMode` | Offboard kontrol modu |
| `/px4_1/fmu/in/trajectory_setpoint` | `px4_msgs/TrajectorySetpoint` | Yörünge hedef noktası |
| `/px4_1/fmu/in/vehicle_attitude_setpoint_v1` | `px4_msgs/VehicleAttitudeSetpoint` | Attitude hedef noktası |

---

## 📁 Klasör Yapısı

```
DogFight/
│
├── ros2_ws/                           # 🤖 ROS 2 Çalışma Alanı
│   └── src/
│       ├── px4_msgs/                  #   PX4 mesaj tanımları (submodule)
│       ├── px4_ros_com/               #   PX4-ROS2 köprü paketi (submodule)
│       │
│       ├── dogfight_detection/        # 📸 Nesne Tespit Paketi
│       │   └── dogfight_detection/
│       │       ├── yolo_detection_node.py      # YOLO ile hedef tespiti
│       │       └── rfdetr_detection_node.py    # RF-DETR ile hedef tespiti
│       │
│       ├── dogfight_tracking/         # 🎯 Hedef Takip Paketi
│       │   └── dogfight_tracking/
│       │       ├── gps_tracker_node.py         # İki araç GPS takibi
│       │       └── visual_tracker_node.py      # Görsel ofboard takip
│       │
│       ├── dogfight_control/          # 🕹️ Uçuş Kontrol Paketi
│       │   └── dogfight_control/
│       │       ├── attitude_controller_node.py # Attitude setpoint (PID)
│       │       ├── velocity_controller_node.py # Velocity setpoint
│       │       ├── position_controller_node.py # GPS pozisyon takip
│       │       └── visual_offboard_node.py     # YOLO tabanlı velocity
│       │
│       └── dogfight_bringup/          # 🚀 Launch & Konfigürasyon
│           ├── launch/
│           │   ├── detection_launch.py
│           │   ├── tracking_launch.py
│           │   └── full_system_launch.py
│           └── config/
│               ├── detection_params.yaml
│               ├── control_params.yaml
│               └── tracking_params.yaml
│
├── simulation/                        # 🌍 Gazebo Simülasyon Ortamı
│   ├── gazebo/
│   │   ├── models/                    #   Uçak & ortam modelleri (SDF)
│   │   └── worlds/                    #   Dünya dosyaları
│   └── scripts/
│       ├── install_dependencies.sh    #   Ortam kurulum scripti
│       └── launch_multi_aircraft.sh   #   Çoklu uçak başlatıcı
│
├── experiments/                       # 🧪 Deneyler & Benchmark'lar
│   ├── model_benchmark/               #   YOLO vs RF-DETR karşılaştırma
│   ├── rfdetr_tctrack/                #   RF-DETR + TCTrack entegrasyonu
│   ├── yolo_inference/                #   YOLO test scriptleri
│   ├── rfdetr_inference/              #   RF-DETR test scriptleri
│   └── training/                      #   Model eğitim notebook'ları
│
├── tools/                             # 🔧 Yardımcı Araçlar
│   ├── gps_bearing_calculator.py      #   GPS mesafe/yön hesaplayıcı
│   └── convert_video_format.py        #   Video format dönüştürücü
│
├── TCTrack/                           # 📦 TCTrack Takip Framework'ü (submodule)
│
├── docs/                              # 📚 Dokümantasyon
│   ├── architecture.md
│   └── setup_guide.md
│
├── .gitignore
├── .gitmodules
└── README.md                          # 📖 Bu dosya
```

---

## ✅ Gereksinimler

### Sistem
- **OS**: Ubuntu 22.04 LTS
- **ROS 2**: Humble Hawksbill
- **Gazebo**: Harmonic (gz-sim)
- **PX4**: v1.15+ (SITL veya gerçek donanım)

### Python Paketleri
```bash
pip3 install ultralytics opencv-python supervision rfdetr torch
```

### PX4 Köprüsü
```bash
# Micro XRCE-DDS Agent kurulumu
cd ~/Micro-XRCE-DDS-Agent/build
sudo make install
sudo ldconfig /usr/local/lib/
```

---

## 🛠️ Kurulum

### 1. Depoyu Klonla
```bash
git clone --recursive https://github.com/YOUR_USER/DogFight.git
cd DogFight
```

### 2. Bağımlılıkları Kur
```bash
# Otomatik kurulum (PX4 + ROS 2 + Agent)
bash simulation/scripts/install_dependencies.sh
```

### 3. ROS 2 Workspace Build
```bash
source /opt/ros/humble/setup.bash
cd ros2_ws
colcon build --symlink-install
source install/local_setup.bash
```

---

## 🚀 Çalıştırma

### Adım 1: Micro XRCE-DDS Agent
```bash
MicroXRCEAgent udp4 -p 8888
```

### Adım 2: PX4 SITL Simülasyonu
```bash
# Tek uçak
PX4_SYS_AUTOSTART=4003 PX4_SIM_MODEL=gz_rc_cessna \
  ./build/px4_sitl_default/bin/px4 -i 1

# Çoklu uçak (TEKNOFEST senaryosu)
bash simulation/scripts/launch_multi_aircraft.sh
```

### Adım 3: ROS 2 Node'ları

#### Tek Tek Çalıştırma
```bash
# YOLO Tespit
ros2 run dogfight_detection yolo_detection_node

# GPS Takip
ros2 run dogfight_tracking gps_tracker_node

# Attitude Kontrolcü
ros2 run dogfight_control attitude_controller_node
```

#### Launch Dosyası ile Toplu Çalıştırma
```bash
# Tüm sistemi başlat
ros2 launch dogfight_bringup full_system_launch.py

# Sadece tespit pipeline'ı
ros2 launch dogfight_bringup detection_launch.py

# Sadece takip + kontrol
ros2 launch dogfight_bringup tracking_launch.py
```

---

## 📸 ROS 2 Paketleri (Detay)

### `dogfight_detection` — Nesne Tespiti

| Node | Açıklama |
|------|----------|
| `yolo_detection_node` | YOLO modeli ile kamera görüntüsünden hedef tespiti. Tespit edilen hedefin merkez sapmasını `/yolo/target_distance` topic'ine yayınlar. |
| `rfdetr_detection_node` | RF-DETR transformer modeli ile hedef tespiti. Supervision kütüphanesi ile görselleştirme. |

### `dogfight_tracking` — Hedef Takibi

| Node | Açıklama |
|------|----------|
| `gps_tracker_node` | İki araç (PX4_1 ve PX4_3) için GPS konumlarını dinler, aralarındaki mesafeyi Haversine formülü ile hesaplar, CSV'ye loglar. |
| `visual_tracker_node` | YOLO/RF-DETR tespitlerini subscribe ederek fixed-wing lateral/longitudinal setpoint ile görsel takip yapar. |

### `dogfight_control` — Uçuş Kontrolü

| Node | Açıklama |
|------|----------|
| `attitude_controller_node` | GPS tabanlı hedef takibi: Yaw PID (roll), Altitude PID (pitch), Distance PID (thrust). Attitude setpoint ile kontrol. |
| `velocity_controller_node` | Velocity setpoint ile offboard kontrol denemesi. |
| `position_controller_node` | İki uçağın lokal pozisyonlarını karşılaştırarak pozisyon tabanlı takip. |
| `visual_offboard_node` | YOLO'dan gelen sapma verisini velocity komutu haline çevirerek görsel takip. |

### `dogfight_bringup` — Launch & Konfigürasyon

Tüm sistemi veya alt bileşenleri tek komutla başlatmak için launch dosyaları ve merkezi YAML konfigürasyon dosyaları içerir.

---

## 🧪 Deneyler

`experiments/` klasörü, farklı model ve yöntemlerin performans testleri için scriptler içerir:

| Klasör | İçerik |
|--------|--------|
| `model_benchmark/` | YOLO vs RF-DETR (Normal + TensorRT) karşılaştırmalı video çıktısı |
| `rfdetr_tctrack/` | RF-DETR tespit + TCTrack temporal takip entegrasyonu |
| `yolo_inference/` | YOLO inferans, video test, TensorRT dönüştürme |
| `rfdetr_inference/` | RF-DETR inferans, ekran yakalama, SAHI dilimleme, ONNX/TensorRT dönüştürme |
| `training/` | RF-DETR model eğitim notebook'u |

---

## 🌍 Simülasyon

Gazebo Harmonic simülasyonu, 3 adet RC Cessna uçak ile TEKNOFEST senaryosunu çalıştırır:

- **HERO**: Ana uçak (kameralı), kontrol ettiğimiz araç
- **ENEMY1**: Rakip uçak 1
- **ENEMY2**: Rakip uçak 2

GPS Home: Şanlıurfa GAP Havalimanı (LTCS) — Pist 04/22

```bash
# Simülasyonu başlat
bash simulation/scripts/launch_multi_aircraft.sh
```

---

## 🧰 Araçlar

| Araç | Açıklama |
|------|----------|
| `tools/gps_bearing_calculator.py` | İki GPS koordinatı arasında mesafe (km) ve yön (bearing) hesaplar |
| `tools/convert_video_format.py` | `.webm` dosyalarını `.mp4`'e dönüştürür (FFmpeg gerekir) |

---

## ⚠️ Bilinen Sorunlar ve Notlar

- **Model yolları**: Bazı scriptlerde mutlak yollar (`/home/tom/Downloads/...`) bulunur. Kendi sisteminize göre güncelleyin veya `config/` YAML'larını kullanın.
- **Kamera index**: `VideoCapture(0)` veya `VideoCapture(3)` sisteminize uymayabilir.
- **px4_msgs**: ROS 2 workspace'de build edilmiş olmalı.
- **Topic namespace**: Bazı node'lar `/px4_1/...` kullanırken, bazıları namespace'siz `fmu/...` kullanıyor. PX4-ROS2 köprü konfigürasyonunuza göre düzenleyin.
- **Offboard kontrol**: PX4 tarafında doğru mod/arming/parametre ayarı gerekir.

---

## 📜 Lisans

Bu depo içinde bazı dosyalar [PX4 örneklerinden](https://github.com/PX4/px4_ros_com) türetilmiştir ve ilgili dosyaların başında PX4 BSD-3-Clause lisans metni yer alır. Lütfen bu lisans başlıklarını koruyun.

---

## 👤 İletişim

**Yunus Emre Tom** — yunusemretom@gmail.com

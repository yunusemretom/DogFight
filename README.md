<!--
README dili: Türkçe 🇹🇷
Bu repo: ROS 2 + PX4 + YOLO tabanlı takip/"dogfight" denemeleri
-->

# 🐶✈️ DogFight

Bu depo, **ROS 2** üzerinde **PX4 (SITL/gerçek)** telemetri-konum takibi ve **YOLO tabanlı görüntüyle hedef takibi** denemelerini içerir.

> Not: Kodlar araştırma/deneme amaçlıdır. Bazı dosyalarda yol/cihaz numarası gibi makineye özel ayarlar bulunur (örn. model dosya yolu, kamera index’i).

---

## 🧭 Klasör Yapısı

- `ros2_ws/` → Ana ROS 2 çalışma alanı (colcon ile build edilen kısım)
- `ros2_ws/src/object_detection/` → `object_detection` ROS 2 Python paketi
- `gps_log_*.csv` → GPS takip node’unun ürettiği log çıktıları
- `build/`, `install/`, `log/` → (muhtemelen) daha önce alınmış build çıktıları

---

## 🧩 İçerik / Node ve Script’ler

ROS 2 paketi: `object_detection`

### 🎥 Basit Kamera Test Node’u
- Giriş noktası (entry point): `test_cam`
- Çalıştırma:
	```bash
	ros2 run object_detection test_cam
	```

### 🧠 YOLO Tespit (ROS 2 Node)
- Dosya: `ros2_ws/src/object_detection/object_detection/yolo_detection_node.py`
- Kamera: `cv2.VideoCapture(0)`
- Model: `ultralytics.YOLO("/home/tom/Downloads/best(1).pt")` (makineye özel)
- Yayın (publish) topic’i:
	- `/yolo/target_distance` (`geometry_msgs/Point`)
		- `x`: hedef bbox merkezinin frame merkezine göre $
			dx
			$
		- `y`: hedef bbox merkezinin frame merkezine göre $
			dy
			$
		- `z`: confidence

Çalıştırma (script olarak):
```bash
python3 ros2_ws/src/object_detection/object_detection/yolo_detection_node.py
```

### 🛰️ İki Araç GPS Takibi (ROS 2 Node)
- Dosya: `ros2_ws/src/object_detection/object_detection/gps_tracker.py`
- Dinlenen topic’ler:
	- `/px4_1/fmu/out/vehicle_gps_position`
	- `/px4_3/fmu/out/vehicle_gps_position`
- Çıktı:
	- Anlık durum ekranı (terminal)
	- `gps_log_YYYYMMDD_HHMMSS.csv` dosyasına kayıt 📝

Çalıştırma:
```bash
python3 ros2_ws/src/object_detection/object_detection/gps_tracker.py
```

### 🎯 YOLO ile Uçak/Drone Offboard Takip Denemeleri
Bu repo içinde birden fazla offboard kontrol denemesi var. Bazıları YOLO’dan gelen `/yolo/target_distance` verisini subscribe ederek setpoint üretiyor.

- `ros2_ws/src/object_detection/object_detection/visual_tracking_offboard.py` → YOLO ile velocity tabanlı takip denemesi
- `ros2_ws/src/object_detection/object_detection/gps_position_control.py` → Konum bazlı setpoint denemesi
- `ros2_ws/src/object_detection/object_detection/attitude_setpoint_control.py` → Attitude setpoint denemesi

> ⚠️ Uyarı: Offboard kontrol, PX4 tarafında doğru mod/arming/parametre ayarı ve doğru topic namespace’leri gerektirir.

### 🎬 Video Dönüştürme Aracı
- Dosya: `ros2_ws/src/object_detection/object_detection/webm_to_mp4.py`
- Amaç: `.webm` videolarını `.mp4`’e çevirmek (FFmpeg gerekir)
- Örnek:
	```bash
	python3 ros2_ws/src/object_detection/object_detection/webm_to_mp4.py video.webm
	python3 ros2_ws/src/object_detection/object_detection/webm_to_mp4.py --dir ./videos --recursive
	```

### 🧰 Diğer Yardımcı Script’ler
- `ros2_ws/src/object_detection/object_detection/yolo_target_lock.py` → YOLO ile "lock / hedef kilidi" görselleştirmesi (ROS'suz, direkt OpenCV döngüsü)
- `ros2_ws/src/object_detection/object_detection/camera_capture_test.py` → Kamera cihazı/format/FPS denemesi (OpenCV)
- `ros2_ws/src/object_detection/object_detection/px4_arm_takeoff_test.py` → PX4'e `VehicleCommand` ile ARM/MODE/TAKEOFF komut denemesi
- `ros2_ws/src/object_detection/object_detection/velocity_setpoint_control.py` → Basit velocity setpoint denemesi
- `ros2_ws/src/object_detection/object_detection/yolo_visual_drone_control.py` → YOLO tabanlı drone velocity kontrolü

---

## ✅ Gereksinimler

### 🐧 Sistem
- Linux (Ubuntu önerilir)
- ROS 2 (repo notlarında **Humble** kullanılmış)
- `colcon`

### 🐍 Python Paketleri
YOLO tarafı için tipik olarak:
- `ultralytics`
- `opencv-python` (veya sistem OpenCV)

Kurulum örneği:
```bash
pip3 install ultralytics opencv-python
```

### 🛰️ PX4 Mesajları
GPS ve offboard scriptleri `px4_msgs` kullanır.

- ROS ortamında `px4_msgs`/`px4_ros_com` kurulumu veya çalışma alanına eklenmesi gerekebilir.

---

## 🛠️ Kurulum / Build

1) ROS ortamını kaynakla:
```bash
source /opt/ros/humble/setup.sh
```

2) Workspace build:
```bash
cd ros2_ws
colcon build --symlink-install
```

3) Workspace’i kaynakla:
```bash
source install/local_setup.sh
```

---

## 🚀 Çalıştırma Akışı (Örnek)

### 1) Micro XRCE Agent (PX4 ↔ ROS köprüsü)
Repo notlarında şu komut kullanılmış:
```bash
MicroXRCEAgent udp4 -p 8888
```

### 2) PX4 SITL örnek komutları
`ros2_ws/src/object_detection/test/notlarım.txt` içinde geçen örnekler:
```bash
PX4_SYS_AUTOSTART=4003 PX4_SIM_MODEL=gz_rc_cessna ./build/px4_sitl_default/bin/px4 -i 1
PX4_GZ_STANDALONE=1 PX4_SYS_AUTOSTART=4003 PX4_GZ_MODEL_POSE="1,2" PX4_SIM_MODEL=gz_rc_cessna ./build/px4_sitl_default/bin/px4 -i 3
```

### 3) ROS Node’ları
Örn. YOLO tespiti:
```bash
python3 ros2_ws/src/object_detection/object_detection/yolo_detection_node.py
```

Örn. GPS takip:
```bash
python3 ros2_ws/src/object_detection/object_detection/gps_tracker.py
```

---

## 🧪 Test / Kalite

Paket altında temel test şablonları mevcut:
```bash
cd ros2_ws
colcon test
colcon test-result --verbose
```

---

## 🧯 Sık Karşılaşılan Sorunlar

- 🧠 **YOLO model yolu hatası**: `best(1).pt` yolu makineye özel. `yolo_detection_node.py` ve `yolo_target_lock.py` içinde kendi model yolunu güncelle.
- 🎥 **Kamera açılmıyor**: `VideoCapture(0)` veya `VideoCapture(3)` sistemine uymayabilir. Doğru kamera index’ini dene.
- 🛰️ **px4_msgs import hatası**: `px4_msgs` ortamda yoksa kurulmalı/derlenmeli.
- 🧷 **Topic namespace farklı**: Bazı scriptler `/px4_1/...` kullanıyor, bazıları çıplak `fmu/...` kullanıyor. PX4-ROS köprüsündeki namespace’lere göre düzen gerekebilir.

---

## 📜 Lisans / Üçüncü Parti Notu

Bu repo içinde bazı dosyalar PX4 örneklerinden türetilmiş olabilir ve ilgili dosyaların başında PX4 lisans metni yer alır. Lütfen bu dosyaların lisans başlıklarını koruyun.


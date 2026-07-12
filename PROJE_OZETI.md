# DogFight — Proje Özeti

> Bu dosya, projeye ilk kez bakan bir yapay zeka asistanının/geliştiricinin sistemi hızla kavraması içindir.
> Kardeş dosyalar: `DOSYA_YAPISI.md` (nerede ne var) ve `GOREVLER.md` (durum/yapılacaklar).
> Son güncelleme: 2026-07-12

## Proje nedir?

TEKNOFEST'e yönelik **RC uçakla otonom hedef takibi (dogfight)** platformu. Bir sabit kanat İHA (takipçi), diğer bir İHA'yı (hedef/rakip) GPS ve görüntü işleme verisiyle, belirli bir mesafeyi koruyarak otonom takip eder.

- **Uçuş yığını:** PX4 Autopilot v1.16 (SITL: `~/PX4-Autopilot`) + ROS 2 Humble, uXRCE-DDS köprüsü
- **Simülasyon:** Gazebo, çoklu araç (`simulation/scripts/launch_multi_aircraft.sh`)
- **Algılama:** RF-DETR ve YOLO tabanlı hedef tespiti (TensorRT/ONNX ihracı dahil), TCTrack ile görsel takip deneyleri
- **Dil/ortam:** Python (rclpy), Ubuntu 22.04, workspace `ros2_ws/`

## Çekirdek işlev: mesafe korumalı L1 takibi (ÇALIŞIYOR, SITL doğrulamalı)

Ana takip düğümü: `ros2_ws/src/dogfight_tracking/dogfight_tracking/l1_pursuit_node.py`

- PX4 v1.16'nın **fixed_wing_lateral_setpoint** (yanal ivme → L1 bizde) + **fixed_wing_longitudinal_setpoint** (irtifa+airspeed → TECS PX4'te) offboard arayüzünü kullanır.
- Püf noktası: `OffboardControlMode.velocity=true` yayınlanır ama `TrajectorySetpoint` YAYINLANMAZ; böylece PX4'ün kendi mode manager'ı pasif kalır, bizim setpoint'ler işlenir.
- İki rejim: uzakta klasik **L1** (yay-üzerinde takip noktası, hedef dönüş hızı tahmini), yakında (<30 m) **formasyon modu** (hedef rotası + cross-track + viraj feedforward; mesafe along-track PID ile).
- Güvenlik: `min_separation` (3 m'de yavaşla+üste çık), hedef verisi timeout→HOLD, offboard kaybında yeniden bağlanma, irtifa kilidi.
- **Doğrulanmış sonuç:** dönen hedefe karşı 5 m hedefte ort 5.7 m / std 0.28 m; 50 m hedefte ~57 m / std 0.9 m.
- Ayrıntılı rehber: `docs/guides/l1-takip-rehberi.md`

## Bilinmesi gereken tuzaklar (sık düşülen)

1. **Topic sürümleri:** PX4 v1.16'da bazı topic'ler `_v1`/`_v4` son ekli (`vehicle_local_position_v1`, `vehicle_status_v4`). Yeni node'lar her iki varyanta da abone olmalı.
2. **Koordinat çerçevesi:** Her aracın NED orijini kendi kalkış noktası — local pozisyonlar araçlar arası DOĞRUDAN ÇIKARILAMAZ. Pozisyon için `VehicleGlobalPosition`, hız için local NED kullanılır.
3. **SITL failsafe:** GCS bağlı değilse datalink-loss failsafe offboard'u ~40 s'de RTL'e düşürür. SITL takipçisinde `NAV_DLL_ACT=0`, `COM_RCL_EXCEPT=4` yapılmıştır — **gerçek uçuşta geri alınmalı**.
4. **Build:** `colcon build --symlink-install` setuptools 83 ile KIRIK. Symlink'siz derleyin; gerekirse önce `rm -rf build/<pkg> install/<pkg>`.
5. **SITL araç düzeni (şu anki):** takipçi namespace'siz (`/fmu`, MAV sysid 1), hedef `/px4_1` (sysid 2). Eski kodda `/px4_1`+`/px4_3` gibi farklı düzenler kalmış olabilir — node parametreleriyle geçersiz kılın.
6. **Sabit kanatta offboard velocity/position setpoint kullanmayın** — PX4'ün kendi guidance'ı devreye girer, harici L1 çalışmaz (bkz. l1-takip-rehberi §1).

## Hızlı başlangıç

```bash
# 1) Simülasyon (PX4 SITL çoklu araç + Gazebo)
./simulation/scripts/launch_multi_aircraft.sh

# 2) Workspace
cd ros2_ws && colcon build && source install/setup.bash   # symlink-install KULLANMA

# 3) Takip
ros2 run dogfight_tracking l1_pursuit_node --ros-args -p l1_period:=8.0
```

## Dokümantasyon

`docs/` Jekyll sitesidir (GitHub Pages). Rehberler `docs/guides/` altında (kurulum, kullanım, RF-DETR, YOLO, PX4+ROS2 fixed-wing, L1 takip). Yeni rehber = klasöre front-matter'lı `.md` eklemek (otomatik listelenir).

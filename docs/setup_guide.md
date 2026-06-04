# Kurulum ve Çalıştırma Rehberi

Bu rehber, DogFight projesini sıfırdan kurmak ve çalıştırmak için adım adım talimatlar içerir.

## Ön Gereksinimler

| Bileşen | Minimum Sürüm | Notlar |
|---------|---------------|--------|
| Ubuntu | 22.04 LTS | ARM64 veya x86_64 |
| ROS 2 | Humble | `ros-humble-desktop` |
| Gazebo | Harmonic | gz-sim |
| PX4 | v1.15+ | SITL build |
| Python | 3.10+ | — |
| CUDA | 11.8+ | GPU kullanımı için (opsiyonel) |

## 1. Sistem Kurulumu

### Otomatik Kurulum
```bash
cd DogFight
bash simulation/scripts/install_dependencies.sh
```

### Manuel Kurulum

#### ROS 2 Humble
```bash
sudo apt install -y ros-humble-desktop ros-dev-tools
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

#### PX4 Autopilot
```bash
git clone --recursive https://github.com/PX4/PX4-Autopilot.git ~/PX4-Autopilot
bash ~/PX4-Autopilot/Tools/setup/ubuntu.sh
cd ~/PX4-Autopilot && make px4_sitl_default
```

#### Micro XRCE-DDS Agent
```bash
git clone -b v2.4.3 https://github.com/eProsima/Micro-XRCE-DDS-Agent.git
cd Micro-XRCE-DDS-Agent && mkdir build && cd build
cmake .. && make -j$(nproc)
sudo make install && sudo ldconfig /usr/local/lib/
```

#### Python Paketleri
```bash
pip3 install ultralytics opencv-python supervision rfdetr torch cv-bridge
```

## 2. ROS 2 Workspace Build

```bash
source /opt/ros/humble/setup.bash
cd DogFight/ros2_ws
colcon build --symlink-install
source install/local_setup.bash
```

> **İpucu**: `~/.bashrc` dosyanıza ekleyin:
> ```bash
> source /opt/ros/humble/setup.bash
> source ~/DogFight/ros2_ws/install/local_setup.bash
> ```

## 3. Simülasyon Başlatma

### Terminal 1: XRCE Agent
```bash
MicroXRCEAgent udp4 -p 8888
```

### Terminal 2: PX4 SITL
```bash
# Tek uçak
cd ~/PX4-Autopilot
PX4_SYS_AUTOSTART=4003 PX4_SIM_MODEL=gz_rc_cessna \
  ./build/px4_sitl_default/bin/px4 -i 1

# İkinci uçak (farklı terminal)
PX4_GZ_STANDALONE=1 PX4_SYS_AUTOSTART=4003 \
  PX4_GZ_MODEL_POSE="1,2" PX4_SIM_MODEL=gz_rc_cessna \
  ./build/px4_sitl_default/bin/px4 -i 3
```

### Terminal 3: ROS 2 Node'ları
```bash
# Tüm sistemi başlat
ros2 launch dogfight_bringup full_system_launch.py
```

## 4. QGroundControl Bağlantısı

1. QGroundControl'ü açın
2. Application Settings → Comm Links
3. Her uçak için yeni UDP bağlantısı ekleyin:
   - Uçak 1: Port `14540`
   - Uçak 2: Port `14541`
   - Uçak 3: Port `14542`
4. Automatically Connect: ON

## 5. Uçuş Prosedürü

1. QGC'de aracı seçin
2. **Arm** (güç düğmesi)
3. **Takeoff** (manuel takeoff veya mission)
4. Yeterli irtifaya ulaştıktan sonra **Offboard** moduna geçin
5. ROS 2 node'ları otomatik olarak kontrol komutları göndermeye başlayacaktır

## Sorun Giderme

| Sorun | Çözüm |
|-------|-------|
| `px4_msgs` import hatası | Workspace'i rebuild edin: `colcon build --symlink-install` |
| Kamera açılmıyor | `VideoCapture(0)` index'ini değiştirin |
| Offboard moda geçemiyor | PX4 tarafında `COM_OBL_ACT` parametresini kontrol edin |
| YOLO model bulunamadı | Model yolunu `config/detection_params.yaml` içinde güncelleyin |
| Agent bağlanamıyor | Port ve IP adresini kontrol edin: `MicroXRCEAgent udp4 -p 8888` |

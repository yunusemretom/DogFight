#!/bin/bash
set -euo pipefail

# ============================================================

# PX4 + ROS2 + Micro XRCE-DDS Agent Automatic Installer

# ============================================================

echo "===== Kurulum Başlıyor ====="

#############################################

# Yardımcı Fonksiyonlar

#############################################

log() {
echo -e "\n[INFO] $1\n"
}

error_exit() {
echo -e "\n[ERROR] $1\n"
exit 1
}

command_exists() {
command -v "$1" >/dev/null 2>&1
}

#############################################

# Sudo Yetki Kontrolü

#############################################

if ! sudo -v; then
error_exit "Sudo yetkisi gerekli."
fi

#############################################

# Sistem Güncelleme

#############################################

log "Sistem güncelleniyor..."
sudo apt update
sudo apt upgrade -y

#############################################

# PX4 Kurulumu

#############################################

PX4_DIR="$HOME/PX4-Autopilot"

if [ ! -d "$PX4_DIR" ]; then
log "PX4 indiriliyor..."
cd "$HOME"
git clone https://github.com/PX4/PX4-Autopilot.git --recursive
else
log "PX4 zaten mevcut, güncelleniyor..."
cd "$PX4_DIR"
git pull
git submodule update --init --recursive
fi

log "PX4 bağımlılıkları kuruluyor..."
bash "$PX4_DIR/Tools/setup/ubuntu.sh"

log "PX4 derleniyor..."
cd "$PX4_DIR"
make px4_sitl_default

#############################################

# ROS2 Kurulumu (Humble)

#############################################

if [ ! -d "/opt/ros/humble" ]; then
log "ROS2 Humble kuruluyor..."

```
sudo apt install -y locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

sudo apt install -y software-properties-common
sudo add-apt-repository universe -y

sudo apt install -y curl

sudo curl -sSL \
    https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) \
signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu \
$(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
| sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install -y ros-humble-desktop ros-dev-tools
```

else
log "ROS2 zaten kurulu."
fi

#############################################

# ROS2 Environment

#############################################

if ! grep -q "source /opt/ros/humble/setup.bash" "$HOME/.bashrc"; then
echo "source /opt/ros/humble/setup.bash" >> "$HOME/.bashrc"
fi

source /opt/ros/humble/setup.bash

#############################################

# Python Paketleri

#############################################

log "Python bağımlılıkları kuruluyor..."
pip3 install --user -U empy==3.3.4 pyros-genmsg setuptools

#############################################

# Micro XRCE-DDS Agent Kurulumu

#############################################

AGENT_DIR="$HOME/Micro-XRCE-DDS-Agent"

if [ ! -d "$AGENT_DIR" ]; then
log "Micro XRCE-DDS Agent indiriliyor..."
cd "$HOME"
git clone -b v2.4.3 https://github.com/eProsima/Micro-XRCE-DDS-Agent.git
fi

log "Micro XRCE-DDS Agent derleniyor..."
cd "$AGENT_DIR"
mkdir -p build
cd build
cmake ..
make -j$(nproc)
sudo make install
sudo ldconfig /usr/local/lib/

#############################################

# Kurulum Bitti

#############################################

log "Kurulum tamamlandı!"
echo "Terminali yeniden başlatman önerilir."

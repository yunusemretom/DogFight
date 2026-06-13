#!/bin/bash
# =====================================================================
# TEKNOFEST Multi-Aircraft Simulation Launcher
# 3 adet Cessna uçak için PX4 SITL + Gazebo Harmonic
# =====================================================================

set -e

# ==================== YAPILANDIRMA ====================
PX4_DIR="${PX4_DIR:-$HOME/PX4-Autopilot}"
BUILD_DIR="$PX4_DIR/build/px4_sitl_default"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SIMULATION_WS="$(dirname "$SCRIPT_DIR")"
SIMULATION_MODELS="$SIMULATION_WS/gazebo/models"
WORLD_FILE="$SIMULATION_WS/gazebo/worlds/default.sdf"

# gz_env.sh will be dynamically sourced in check_prerequisites once build exists

#Hero uçak için kameralı özel model
HERO_MODEL="rc_cessna"

AIRFRAME_ID=4003  # gz_rc_cessna
LOG_DIR="/tmp/px4_multi_sim"

# GPS Home pozisyonu - Şanlıurfa GAP Havalimanı (LTCS)
# Pist 04/22 - Heading ~40°
HOME_LAT=37.445617
HOME_LON=38.895592
HOME_ALT=825

# Renkler
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'


# ==================== FONKSİYONLAR ====================
cleanup() {
    echo ""
    echo -e "${YELLOW}[CLEANUP]${NC} Simülasyon durduruluyor..."
    
    # PX4 instances
    pkill -9 -x px4 2>/dev/null || true
    
    # Gazebo
    pkill -9 -f "gz sim" 2>/dev/null || true
    pkill -9 -f "ruby" 2>/dev/null || true
    
    # MAVLink router
    pkill -9 -f "mavlink-routerd" 2>/dev/null || true
    
    # ROS 2 Gazebo Köprüsü
    pkill -9 -f "ros_gz_bridge" 2>/dev/null || true
    
    echo -e "${GREEN}[CLEANUP]${NC} Tamamlandı."
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

check_prerequisites() {
    echo -e "${CYAN}[CHECK]${NC} Ön kontroller yapılıyor..."
    
    # PX4 build kontrolü
    if [ ! -f "$BUILD_DIR/bin/px4" ]; then
        echo -e "${YELLOW}[BUILD]${NC} PX4 build bulunamadı. Derleniyor..."
        cd "$PX4_DIR"
        make px4_sitl_default
    fi
    
    # World dosyası kontrolü
    if [ ! -f "$WORLD_FILE" ]; then
        echo -e "${RED}[ERROR]${NC} World dosyası bulunamadı: $WORLD_FILE"
        exit 1
    fi
    
    # gz_env.sh dosyasını source et
    local GZ_ENV_FILE="$BUILD_DIR/rootfs/gz_env.sh"
    if [ -f "$GZ_ENV_FILE" ]; then
        source "$GZ_ENV_FILE"
    else
        echo -e "${RED}[ERROR]${NC} Gazebo ortam dosyası bulunamadı: $GZ_ENV_FILE"
        echo "Lütfen PX4 build'ini kontrol edin."
        exit 1
    fi
    
    echo -e "${GREEN}[CHECK]${NC} Ön kontroller tamam."
}

start_bridge() {
    echo -e "${CYAN}[BRIDGE]${NC} ROS 2 Gazebo Köprüsü başlatılıyor..."
    
    # HERO uçağı için kamera görüntüsünü köprüle (Instance 1 -> rc_cessna_1)
    ros2 run ros_gz_bridge parameter_bridge \
        "/world/default/model/rc_cessna_1/link/camera_link/sensor/camera/image@sensor_msgs/msg/Image[gz.msgs.Image" \
        > "$LOG_DIR/ros_gz_bridge.log" 2>&1 &
    
    sleep 2
    if ! pgrep -f "ros_gz_bridge" > /dev/null; then
        echo -e "${RED}[ERROR]${NC} ROS 2 Gazebo Köprüsü başlatılamadı!"
        echo "Log: $LOG_DIR/ros_gz_bridge.log"
        cat "$LOG_DIR/ros_gz_bridge.log" | tail -20
    else
        echo -e "${GREEN}[BRIDGE]${NC} ROS 2 Gazebo Köprüsü hazır."
    fi
}

start_gazebo() {
    echo -e "${CYAN}[GAZEBO]${NC} Gazebo Harmonic başlatılıyor..."
    
    # Model path'i ayarla
    export GZ_SIM_RESOURCE_PATH="${SIMULATION_MODELS}:$SIMULATION_WS/gazebo/worlds:${PX4_DIR}/Tools/simulation/gz/models:${PX4_DIR}/Tools/simulation/gz/worlds:$GZ_SIM_RESOURCE_PATH:$PX4_GZ_MODELS:$PX4_GZ_WORLDS"
    
    # Gazebo sunucusunu (headless/server) başlat
    gz sim -v 1 -s -r "$WORLD_FILE" > "$LOG_DIR/gazebo_server.log" 2>&1 &
    GAZEBO_PID=$!
    
    # Sunucunun başlaması için kısa bir süre bekle
    sleep 5
    
    # Gazebo arayüzünü (GUI client) ayrı bir işlem olarak başlat (deadlock/donmaları önlemek için)
    gz sim -g > "$LOG_DIR/gazebo_gui.log" 2>&1 &
    GUI_PID=$!
    
    # Gazebo sunucusunun hazır olmasını bekle
    echo -e "${YELLOW}[GAZEBO]${NC} Gazebo sunucusu yükleniyor (15 saniye)..."
    sleep 15
    
    # Kontrol
    if ! kill -0 $GAZEBO_PID 2>/dev/null; then
        echo -e "${RED}[ERROR]${NC} Gazebo sunucusu başlatılamadı!"
        echo "Log: $LOG_DIR/gazebo_server.log"
        cat "$LOG_DIR/gazebo_server.log" | tail -20
        exit 1
    fi
    
    echo -e "${GREEN}[GAZEBO]${NC} Gazebo sunucusu hazır."
    
    # ROS 2 Köprüsünü Başlat
    start_bridge
}

launch_aircraft() {
    local INSTANCE=$1
    local NAME=$2
    local X=$3
    local Y=$4
    local Z=$5
    local YAW=$6
    local MODEL=${7:-"rc_cessna"}  # Varsayılan: rc_cessna
    
    local UDP_PORT=$((14540 + INSTANCE))
    
    echo -e "${CYAN}[AIRCRAFT $INSTANCE]${NC} $NAME spawn ediliyor..."
    echo "  Model: $MODEL"
    echo "  Pozisyon: X=$X, Y=$Y, Z=$Z, Yaw=$YAW"
    echo "  UDP Port: $UDP_PORT"
    
    (
        cd "${BUILD_DIR}"
        
        # Instance-specific rootfs
        mkdir -p "rootfs/instance_$INSTANCE"
        cd "rootfs/instance_$INSTANCE"
        
        # Environment variables
        export PX4_SYS_AUTOSTART=${AIRFRAME_ID}
        export PX4_GZ_MODEL="${MODEL}"
        export PX4_GZ_MODEL_POSE="${X},${Y},${Z},0,0,${YAW}"
        export PX4_GZ_WORLD="default"
        export PX4_GZ_STANDALONE=1
        export PX4_SIM_SPEED_FACTOR=1
        export PX4_HOME_LAT=${HOME_LAT}
        export PX4_HOME_LON=${HOME_LON}
        export PX4_HOME_ALT=${HOME_ALT}
        export GZ_SIM_RESOURCE_PATH="${GZ_SIM_RESOURCE_PATH}"
        
        # PX4 instance başlat
        # -i: instance ID, her uçak için farklı
        ${BUILD_DIR}/bin/px4 -i ${INSTANCE} -d
        
    ) > "$LOG_DIR/px4_$INSTANCE.log" 2>&1 &
    
    echo -e "${GREEN}[AIRCRAFT $INSTANCE]${NC} $NAME başlatıldı."
}

wait_for_models() {
    local MAX_WAIT=60
    local WAITED=0
    
    echo -e "${YELLOW}[WAIT]${NC} Modellerin spawn olması bekleniyor..."
    
    while [ $WAITED -lt $MAX_WAIT ]; do
        MODELS=$(gz model --list 2>/dev/null | grep "cessna" | wc -l)
        if [ "$MODELS" -ge 3 ]; then
            echo -e "${GREEN}[WAIT]${NC} 3 uçak spawn edildi!"
            return 0
        fi
        sleep 2
        WAITED=$((WAITED + 2))
        echo "  $WAITED saniye... ($MODELS/3 model)"
    done
    
    echo -e "${YELLOW}[WAIT]${NC} Timeout! Spawn edilen model sayısı: $MODELS"
    return 1
}

print_status() {
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}✓ TEKNOFEST Multi-Aircraft Simülasyonu Hazır!${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "${YELLOW}Spawn Edilen Modeller:${NC}"
    gz model --list 2>/dev/null | grep -E "cessna" | while read line; do
        echo "  • $line"
    done
    echo ""
    echo -e "${YELLOW}QGroundControl Bağlantıları:${NC}"
    echo "  Uçak 1 (HERO):   udp://127.0.0.1:14541"
    echo "  Uçak 2 (ENEMY1): udp://127.0.0.1:14542"
    echo "  Uçak 3 (ENEMY2): udp://127.0.0.1:14543"
    echo ""
    echo -e "${YELLOW}QGC Ayarları:${NC}"
    echo "  1. QGroundControl açın"
    echo "  2. Application Settings > Comm Links"
    echo "  3. Her uçak için yeni UDP bağlantısı ekleyin:"
    echo "     - Type: UDP"
    echo "     - Port: 14541, 14542, 14543"
    echo "     - Automatically connect: ON"
    echo ""
    echo -e "${YELLOW}Takeoff Prosedürü (Her uçak için ayrı ayrı):${NC}"
    echo "  1. QGC'de ilgili aracı seçin"
    echo "  2. Arm (Güç düğmesi)"
    echo "  3. Takeoff (Veya Mission başlatın)"
    echo ""
    echo -e "${YELLOW}Log Dosyaları:${NC} $LOG_DIR/"
    echo ""
    echo -e "${RED}Durdurmak için: Ctrl+C${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
}

# ==================== ANA PROGRAM ====================
# clear
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}     TEKNOFEST Multi-Aircraft Simulation Launcher${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# Temizlik
echo -e "${YELLOW}[1/5]${NC} Eski işlemler temizleniyor..."
pkill -9 -x px4 2>/dev/null || true
pkill -9 -f "gz sim" 2>/dev/null || true
pkill -9 -f "ruby" 2>/dev/null || true
sleep 2

# Log dizini
rm -rf "$LOG_DIR" 2>/dev/null || true
mkdir -p "$LOG_DIR"

# Ön kontroller
echo -e "${YELLOW}[2/5]${NC} Ön kontroller..."
check_prerequisites

# Gazebo başlat
echo -e "${YELLOW}[3/5]${NC} Gazebo başlatılıyor..."
start_gazebo

# Uçakları spawn et
# Şanlıurfa GAP Havalimanı Pist 04/22
# Pist 04 yönü: ~40° heading = 0.698 radyan
echo -e "${YELLOW}[4/5]${NC} Uçaklar spawn ediliyor..."

# ENEMY 1 (KOUSTECH) - Sağda
launch_aircraft 2     "ENEMY1"  -5   10   0.3    0.698     "rc_cessna"
sleep 10
#6.1-5.1
# ENEMY 2 (ITU_ATA) - Solda
launch_aircraft 3     "ENEMY2"  5   -10   0.3    0.698     "rc_cessna"
sleep 10
#12.3-10.3
# Model kontrolü
echo -e "${YELLOW}[5/5]${NC} Modeller kontrol ediliyor..."
wait_for_models

# Durum bilgisi
print_status

# Bekle
while true; do
    sleep 5
    
    # Sağlık kontrolü
    if ! kill -0 $GAZEBO_PID 2>/dev/null; then
        echo -e "${RED}[ERROR]${NC} Gazebo kapandı!"
        exit 1
    fi
    
    PX4_COUNT=$(pgrep -x px4 | wc -l)
    if [ "$PX4_COUNT" -lt 3 ]; then
        echo -e "${YELLOW}[WARN]${NC} Çalışan PX4 instance: $PX4_COUNT/3"
    fi
done

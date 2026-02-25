#!/bin/bash
# =====================================================================
# TEKNOFEST Multi-Aircraft Simulation Launcher
# 3 adet Cessna uçak için PX4 SITL + Gazebo Harmonic
# =====================================================================

set -e

# ==================== YAPILANDIRMA ====================
PX4_DIR="${PX4_DIR:-$HOME/Desktop/PX4-Autopilot}"
BUILD_DIR="$PX4_DIR/build/px4_sitl_default"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SIMULATION_WS="$(dirname "$SCRIPT_DIR")"
SIMULATION_MODELS="$SIMULATION_WS/models"
WORLD_FILE="$SIMULATION_WS/worlds/baylands.sdf"
source ~/Desktop/PX4-Autopilot/build/px4_sitl_default/rootfs/gz_env.sh
#PX4_DIR="${PX4_DIR:-$HOME/Desktop/PX4-Autopilot}"
#BUILD_DIR="$PX4_DIR/build/px4_sitl_default"
#SIMULATION_WS="/home/tom/Documents/Dog_Fight/gazebo_simulation"
#SIMULATION_MODELS="$SIMULATION_WS/models"
#WORLD_FILE="$SIMULATION_WS/worlds/baylands.sdf"
 #Hero uçak için kameralı özel model
HERO_MODEL="zephyr"

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
        make px4_sitl gz_rc_cessna
    fi
    
    # World dosyası kontrolü
    if [ ! -f "$WORLD_FILE" ]; then
        echo -e "${RED}[ERROR]${NC} World dosyası bulunamadı: $WORLD_FILE"
        exit 1
    fi
    
    echo -e "${GREEN}[CHECK]${NC} Ön kontroller tamam."
}

start_gazebo() {
    echo -e "${CYAN}[GAZEBO]${NC} Gazebo Harmonic başlatılıyor..."
    
    # Model path'i ayarla
    export GZ_SIM_RESOURCE_PATH="${SIMULATION_MODELS}:$SIMULATION_WS/worlds:${PX4_DIR}/Tools/simulation/gz/models:${PX4_DIR}/Tools/simulation/gz/worlds"
    
    # Gazebo'yu başlat
    gz sim -v 1 -r "$WORLD_FILE" > "$LOG_DIR/gazebo.log" 2>&1 &
    GAZEBO_PID=$!
    
    # Gazebo'nun başlamasını bekle
    echo -e "${YELLOW}[GAZEBO]${NC} Gazebo yükleniyor (10 saniye)..."
    sleep 20
    
    # Kontrol
    if ! pgrep -f "gz sim" > /dev/null; then
        echo -e "${RED}[ERROR]${NC} Gazebo başlatılamadı!"
        echo "Log: $LOG_DIR/gazebo.log"
        cat "$LOG_DIR/gazebo.log" | tail -20
        exit 1
    fi
    
    echo -e "${GREEN}[GAZEBO]${NC} Gazebo hazır."
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
        export PX4_GZ_WORLD=$WORLD_FILE
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
        MODELS=$(gz model --list 2>/dev/null | grep -c "cessna" || echo "0")
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
    echo "  Uçak 1 (HERO):   udp://127.0.0.1:14540"
    echo "  Uçak 2 (ENEMY1): udp://127.0.0.1:14541"
    echo "  Uçak 3 (ENEMY2): udp://127.0.0.1:14542"
    echo ""
    echo -e "${YELLOW}QGC Ayarları:${NC}"
    echo "  1. QGroundControl açın"
    echo "  2. Application Settings > Comm Links"
    echo "  3. Her uçak için yeni UDP bağlantısı ekleyin:"
    echo "     - Type: UDP"
    echo "     - Port: 14540, 14541, 14542"
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
clear
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

# HERO (GUCLU) - Ana uçak - KAMERALI
#            Instance  Name      X    Y     Z     Yaw(rad)  Model
launch_aircraft 0     "HERO"    0    0    0.3    0.698     "${HERO_MODEL}"
sleep 15  # İlk uçak için daha uzun bekle

# ENEMY 1 (KOUSTECH) - Sağda
launch_aircraft 1     "ENEMY1"  -5   10   0.3    0.698     "rc_cessna"
sleep 10
#6.1-5.1
# ENEMY 2 (ITU_ATA) - Solda
launch_aircraft 2     "ENEMY2"  5   -10   0.3    0.698     "rc_cessna"
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
    if ! pgrep -f "gz sim" > /dev/null; then
        echo -e "${RED}[ERROR]${NC} Gazebo kapandı!"
        exit 1
    fi
    
    PX4_COUNT=$(pgrep -x px4 | wc -l)
    if [ "$PX4_COUNT" -lt 3 ]; then
        echo -e "${YELLOW}[WARN]${NC} Çalışan PX4 instance: $PX4_COUNT/3"
    fi
done

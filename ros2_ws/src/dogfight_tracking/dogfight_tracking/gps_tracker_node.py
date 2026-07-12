#!/usr/bin/env python3
"""
İki Araçlı GPS Takip Sistemi
PX4_1 (ID: 2) ve PX4_3 (ID: 4) için GPS takibi
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import SensorGps
import math
from datetime import datetime
import os

class GPSTracker(Node):
    """İki araç için GPS takip sistemi"""
    
    def __init__(self):
        super().__init__("gps_tracker")
        
        # QoS Profili - PX4 ile uyumlu
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # GPS verileri için depolama
        self.vehicle1_gps = None  # PX4_1 (ID: 2)
        self.vehicle2_gps = None  # PX4_3 (ID: 4)
        
        # GPS geçmişi (rota çizimi için)
        self.vehicle1_history = []
        self.vehicle2_history = []
        self.max_history = 100  # Son 100 nokta
        
        # İstatistikler
        self.v1_message_count = 0
        self.v2_message_count = 0
        
        # Subscribers - Her iki araç için GPS dinle
        self.gps_sub1 = self.create_subscription(
            SensorGps,
            "fmu/out/vehicle_gps_position",
            self.gps_callback_vehicle1,
            qos
        )
        
        self.gps_sub2 = self.create_subscription(
            SensorGps,
            "/px4_1/fmu/out/vehicle_gps_position",
            self.gps_callback_vehicle2,
            qos
        )
        
        # Timer - Her saniye durum göster
        self.timer = self.create_timer(1.0, self.display_status)
        
        # Log dosyası
        self.log_file = f"gps_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.init_log_file()
        
        self.get_logger().info("🛰️  GPS Takip Sistemi Başlatıldı")
        self.get_logger().info("📡 Araç 1 (PX4_1, ID:2): /px4_1/fmu/out/vehicle_gps_position")
        self.get_logger().info("📡 Araç 2 (PX4_3, ID:4): /px4_3/fmu/out/vehicle_gps_position")
        self.get_logger().info(f"📝 Log dosyası: {self.log_file}")
    
    def init_log_file(self):
        """CSV log dosyası başlat"""
        with open(self.log_file, 'w') as f:
            f.write("timestamp,vehicle_id,latitude,longitude,altitude,satellites,fix_type,speed,distance_between\n")
    
    def gps_callback_vehicle1(self, msg):
        """Araç 1 (PX4_1, ID:2) GPS callback"""
        self.vehicle1_gps = msg
        self.v1_message_count += 1
        # Koordinat geçmişine ekle
        if msg.fix_type >= 3:  # 3D fix varsa
            lat = msg.latitude_deg
            lon = msg.longitude_deg
            alt = msg.altitude_msl_m
            
            self.vehicle1_history.append((lat, lon, alt))
            if len(self.vehicle1_history) > self.max_history:
                self.vehicle1_history.pop(0)
            
            # Log'a yaz
            self.log_gps_data(2, msg)
    
    def gps_callback_vehicle2(self, msg):
        """Araç 2 (PX4_3, ID:4) GPS callback"""
        self.vehicle2_gps = msg
        self.v2_message_count += 1
        
        # Koordinat geçmişine ekle
        if msg.fix_type >= 3:  # 3D fix varsa
            lat = msg.latitude_deg
            lon = msg.longitude_deg
            alt = msg.altitude_msl_m
            
            self.vehicle2_history.append((lat, lon, alt))
            if len(self.vehicle2_history) > self.max_history:
                self.vehicle2_history.pop(0)
            
            # Log'a yaz
            self.log_gps_data(4, msg)
    
    def log_gps_data(self, vehicle_id, msg):
        """GPS verisini dosyaya kaydet"""
        distance = self.calculate_distance() if self.vehicle1_gps and self.vehicle2_gps else 0.0
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        line = f"{timestamp},{vehicle_id},{msg.latitude_deg:.8f},{msg.longitude_deg:.8f},"
        line += f"{msg.altitude_msl_m:.2f},{msg.satellites_used},{msg.fix_type},"
        line += f"{msg.velocity_m_s:.2f},{distance:.2f}\n"
        
        with open(self.log_file, 'a') as f:
            f.write(line)
    
    def calculate_distance(self):
        """İki araç arasındaki mesafeyi hesapla (Haversine formülü)"""
        if not self.vehicle1_gps or not self.vehicle2_gps:
            return 0.0
        
        # Her iki aracın da geçerli GPS fix'i olmalı
        if self.vehicle1_gps.fix_type < 3 or self.vehicle2_gps.fix_type < 3:
            return 0.0
        
        lat1 = math.radians(self.vehicle1_gps.latitude_deg)
        lon1 = math.radians(self.vehicle1_gps.longitude_deg)
        lat2 = math.radians(self.vehicle2_gps.latitude_deg)
        lon2 = math.radians(self.vehicle2_gps.longitude_deg)
        
        # Haversine formülü
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Dünya yarıçapı (metre)
        r = 6371000
        
        distance = r * c
        return distance
    
    def get_fix_type_str(self, fix_type):
        """GPS fix tipini string'e çevir"""
        fix_types = {
            0: "YOK",
            1: "FIX YOK",
            2: "2D FIX",
            3: "3D FIX",
            4: "DGPS",
            5: "RTK FLOAT",
            6: "RTK FIXED"
        }
        return fix_types.get(fix_type, f"BİLİNMEYEN({fix_type})")
    
    def display_status(self):
        """GPS durumunu terminale yazdır"""
        os.system('clear')  # Terminali temizle
        
        print("=" * 80)
        print("🛰️  İKİ ARAÇLI GPS TAKİP SİSTEMİ".center(80))
        print("=" * 80)
        print()
        
        # ARAÇ 1 (PX4_1, ID:2)
        print("🚁 ARAÇ 1 - PX4_1 (ID: 2)".ljust(80, "─"))
        if self.vehicle1_gps:
            gps = self.vehicle1_gps
            print(f"  📍 Konum      : {gps.latitude_deg:.8f}°, {gps.longitude_deg:.8f}°")
            print(f"  🏔️  Yükseklik   : {gps.altitude_msl_m:.2f} m (MSL), {gps.altitude_ellipsoid_m:.2f} m (WGS84)")
            print(f"  🛰️  Uydu Sayısı : {gps.satellites_used}")
            print(f"  📡 Fix Tipi   : {self.get_fix_type_str(gps.fix_type)}")
            print(f"  🚀 Hız        : {gps.velocity_m_s:.2f} m/s ({gps.velocity_m_s * 3.6:.2f} km/h)")
            print(f"  📊 Mesaj      : {self.v1_message_count}")
            print(f"  📈 Güzergah   : {len(self.vehicle1_history)} nokta")
        else:
            print("  ❌ GPS verisi bekleniyor...")
        
        print()
        
        # ARAÇ 2 (PX4_3, ID:4)
        print("🚁 ARAÇ 2 - PX4_3 (ID: 4)".ljust(80, "─"))
        if self.vehicle2_gps:
            gps = self.vehicle2_gps
            print(f"  📍 Konum      : {gps.latitude_deg:.8f}°, {gps.longitude_deg:.8f}°")
            print(f"  🏔️  Yükseklik   : {gps.altitude_msl_m:.2f} m (MSL), {gps.altitude_ellipsoid_m:.2f} m (WGS84)")
            print(f"  🛰️  Uydu Sayısı : {gps.satellites_used}")
            print(f"  📡 Fix Tipi   : {self.get_fix_type_str(gps.fix_type)}")
            print(f"  🚀 Hız        : {gps.velocity_m_s:.2f} m/s ({gps.velocity_m_s * 3.6:.2f} km/h)")
            print(f"  📊 Mesaj      : {self.v2_message_count}")
            print(f"  📈 Güzergah   : {len(self.vehicle2_history)} nokta")
        else:
            print("  ❌ GPS verisi bekleniyor...")
        
        print()
        print("=" * 80)
        
        # İKİ ARAÇ ARASI MESAFE
        if self.vehicle1_gps and self.vehicle2_gps:
            if self.vehicle1_gps.fix_type >= 3 and self.vehicle2_gps.fix_type >= 3:
                distance = self.calculate_distance()
                print(f"📏 ARAÇLAR ARASI MESAFE: {distance:.2f} m ({distance/1000:.3f} km)".center(80))
                
                # Uyarılar
                if distance < 10:
                    print("⚠️  UYARI: Araçlar çok yakın! (<10m)".center(80))
                elif distance < 50:
                    print("⚡ DİKKAT: Araçlar yakın (<50m)".center(80))
            else:
                print("⏳ Mesafe hesaplaması için 3D GPS fix bekleniyor...".center(80))
        else:
            print("⏳ Her iki araçtan GPS verisi bekleniyor...".center(80))
        
        print("=" * 80)
        print()
        print(f"📝 Log: {self.log_file}")
        print("🛑 Çıkmak için: Ctrl+C")
        print()


def main():
    rclpy.init()
    node = GPSTracker()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\n⚠️  Program durduruldu")
    finally:
        node.get_logger().info("📊 İstatistikler:")
        node.get_logger().info(f"  Araç 1: {node.v1_message_count} mesaj")
        node.get_logger().info(f"  Araç 2: {node.v2_message_count} mesaj")
        node.get_logger().info(f"📝 Log: {node.log_file}")
        node.destroy_node()
        rclpy.shutdown()
        print("👋 GPS Takip sistemi kapatıldı")


if __name__ == "__main__":
    main()

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import VehicleStatus
from px4_msgs.msg import FixedWingLateralSetpoint
from px4_msgs.msg import FixedWingLongitudinalSetpoint
from geometry_msgs.msg import Point


def wrap_pi(angle_rad: float) -> float:
    """Wrap angle to [-pi, pi]."""
    while angle_rad > math.pi:
        angle_rad -= 2.0 * math.pi
    while angle_rad < -math.pi:
        angle_rad += 2.0 * math.pi
    return angle_rad


class OffboardControl(Node):

    def __init__(self):
        super().__init__('fw_offboard_controller')

        # QoS profiles (PX4 DDS genelde BEST_EFFORT ister)
        qos_profile_pub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        qos_profile_sub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Status subscriber (ikisini birden tutmak yerine farklı isim verelim)
        self.status_sub = self.create_subscription(
            VehicleStatus,
            'fmu/out/vehicle_status_v1',
            self.vehicle_status_callback,
            qos_profile_sub
        )
        
        # YOLO detection subscriber
        self.yolo_sub = self.create_subscription(
            Point,
            '/yolo/target_distance',
            self.yolo_callback,
            qos_profile_sub
        )

        # Publishers
        self.publisher_offboard_mode = self.create_publisher(
            OffboardControlMode,
            'fmu/in/offboard_control_mode',
            qos_profile_pub
        )

        self.publisher_trajectory = self.create_publisher(
            TrajectorySetpoint,
            'fmu/in/trajectory_setpoint',
            qos_profile_pub
        )

        self.publisher_fw_lateral = self.create_publisher(
            FixedWingLateralSetpoint,
            'fmu/in/fixed_wing_lateral_setpoint',
            qos_profile_pub
        )

        self.publisher_fw_longitudinal = self.create_publisher(
            FixedWingLongitudinalSetpoint,
            'fmu/in/fixed_wing_longitudinal_setpoint',
            qos_profile_pub
        )

        # Timer
        self.dt = 0.02  # 50 Hz
        self.timer = self.create_timer(self.dt, self.cmdloop_callback)

        # Vehicle states
        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX
        self.arming_state = VehicleStatus.ARMING_STATE_DISARMED

        # YOLO detection verileri
        self.target_distance_x = 0.0  # Yatay uzaklık (piksel)
        self.target_distance_y = 0.0  # Dikey uzaklık (piksel)
        self.target_confidence = 0.0  # Tespit güvenilirliği
        
        # Kontrol parametreleri
        self.base_heading = 0.0  # Başlangıç heading (radyan)
        self.base_pitch = 0.0    # Başlangıç pitch (radyan)
        
        # Dönüşüm faktörleri (piksel -> radyan)
        # Kamera görüş alanına göre ayarlanmalı
        self.pixel_to_rad_x = 0.003  # Yatay için gain
        self.pixel_to_rad_y = 0.001  # Dikey için gain
        
        # Maksimum sapma limitleri
        self.max_heading_offset = math.radians(40)  # ±30 derece
        self.max_pitch_offset = math.radians(20)    # ±20 derece
        
        # Throttle değeri (0-1 arası normalize)
        self.target_throttle = 0.6

        # İleri hız hedefi (m/s). Fixed-wing için mantıklı bir değer seç.
        # 5 m/s çoğu uçak için stall olabilir. Sim/SITL'de bile düşük kalabilir.
        # Uçağın minimum güvenli hızına göre ayarla.
        self.target_speed = 15.0

    def vehicle_status_callback(self, msg: VehicleStatus):
        self.nav_state = msg.nav_state
        self.arming_state = msg.arming_state

        # İstersen debug:
        # self.get_logger().info(f"NAV={msg.nav_state} ARMED={msg.arming_state}")
    
    def yolo_callback(self, msg: Point):
        """YOLO detection verilerini al ve hedef açıları hesapla"""
        self.target_distance_x = msg.x
        self.target_distance_y = msg.y
        self.target_confidence = msg.z
        
        # Debug log (isteğe bağlı)
        if self.target_confidence > 0.0:
            self.get_logger().info(
                f'YOLO: dx={self.target_distance_x:.1f}, dy={self.target_distance_y:.1f}, conf={self.target_confidence:.2f}'
            )

    def cmdloop_callback(self):
        # Offboard control mode publish (her döngü)
        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)

        # Fixed-wing için: velocity setpoint + (ayrıca) fixed-wing lateral setpoint kullanıyoruz.
        # OffboardControlMode burada "niyet" gibi; yine de velocity'yi açık tutalım.
        offboard_msg.position = False
        offboard_msg.velocity = True
        offboard_msg.acceleration = False
        offboard_msg.attitude = False
        offboard_msg.body_rate = False

        self.publisher_offboard_mode.publish(offboard_msg)

        # YOLO verilerine göre heading ve pitch hesapla
        if self.target_confidence > 0.5:  # Sadece yüksek güvenilirlikte
            # Heading offset hesapla (x eksenindeki sapma)
            heading_offset = -self.target_distance_x * self.pixel_to_rad_x
            heading_offset = max(min(heading_offset, self.max_heading_offset), -self.max_heading_offset)
            target_heading = self.base_heading + heading_offset
            
            # Pitch offset hesapla (y eksenindeki sapma)
            pitch_offset = self.target_distance_y * self.pixel_to_rad_y
            pitch_offset = max(min(pitch_offset, self.max_pitch_offset), -self.max_pitch_offset)
            target_pitch = self.base_pitch + pitch_offset
        else:
            # Tespit yoksa veya düşük güvenilirlikteyse varsayılan değerleri kullan
            target_heading = self.base_heading
            target_pitch = self.base_pitch

        # Sadece OFFBOARD + ARMED iken setpoint gönder (istersen OFFBOARD öncesi de akıtabilirsin)
        if not (self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD and
                self.arming_state == VehicleStatus.ARMING_STATE_ARMED):
            return

        now_us = int(self.get_clock().now().nanoseconds / 1000)

        # 1) Fixed-wing lateral: airspeed_direction ile heading kontrolü
        fw_lat = FixedWingLateralSetpoint()
        fw_lat.timestamp = now_us

        # YOLO'dan hesaplanan heading kullan
        fw_lat.airspeed_direction = wrap_pi(target_heading)

        # Diğerleri kullanılmıyorsa NaN olmalı
        fw_lat.course = math.nan
        fw_lat.lateral_acceleration = math.nan

        self.publisher_fw_lateral.publish(fw_lat)

        # 2) Fixed-wing longitudinal: pitch ve throttle kontrolü
        fw_long = FixedWingLongitudinalSetpoint()
        fw_long.timestamp = now_us

        # YOLO'dan hesaplanan pitch kullan
        fw_long.pitch_direct = float(target_pitch)
        
        # Doğrudan throttle kontrolü (0-1 normalize)
        fw_long.throttle_direct = math.nan
        # Diğer alanları NaN yapıyoruz (kullanılmıyor)
        fw_long.altitude = math.nan
        fw_long.height_rate = math.nan
        fw_long.equivalent_airspeed = math.nan

        self.publisher_fw_longitudinal.publish(fw_long)

        # 3) TrajectorySetpoint: sadece ileri hız rejimi (velocity[0])
        traj = TrajectorySetpoint()

        # Position kullanılmıyor
        traj.position[0] = math.nan
        traj.position[1] = math.nan
        traj.position[2] = math.nan

        # Velocity: fixed-wing için anlamlı olan ileri hız
        traj.velocity[0] = float(self.target_speed)

        # Yana ve dikey velocity fixed-wing için burada kullanılmıyor
        traj.velocity[1] = math.nan
        traj.velocity[2] = math.nan

        # Yaw/yaw_rate da burada kullanılmıyor (lateral setpoint yönetecek)
        # Alan adlarını senin px4_msgs sürümün belirler; yoksa dokunma.
        # traj.yaw = math.nan
        # traj.yaw_rate = math.nan

        self.publisher_trajectory.publish(traj)


def main(args=None):
    rclpy.init(args=args)
    node = OffboardControl()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

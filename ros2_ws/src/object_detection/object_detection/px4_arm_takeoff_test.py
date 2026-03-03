#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import VehicleCommand

PX4_NS = "/px4_1"   # <-- SENDE AKTİF OLAN ARAÇ BU
PX4_SYS_ID = 2
PX4_COMP_ID = 1
class FWTakeoff(Node):
    def __init__(self):
        super().__init__("fw_takeoff")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.cmd_pub = self.create_publisher(
            VehicleCommand,
            f"{PX4_NS}/fmu/in/vehicle_command",
            qos
        )

        self.step = 0
        self.timer = self.create_timer(1.0, self.tick)

    def ts_us(self):
        return int(self.get_clock().now().nanoseconds / 1000)

    def send_cmd(self, cmd, p1=0.0, p2=0.0, p3=0.0, p4=0.0, p5=0.0, p6=0.0, p7=0.0):
        m = VehicleCommand()
        m.timestamp = self.ts_us()
        m.command = int(cmd)
        m.param1, m.param2, m.param3, m.param4 = float(p1), float(p2), float(p3), float(p4)
        m.param5, m.param6, m.param7 = float(p5), float(p6), float(p7)
        m.target_system = PX4_SYS_ID
        m.target_component = PX4_COMP_ID
        m.source_system = 255      # external / companion computer
        m.source_component = 1
        m.from_external = True
        self.cmd_pub.publish(m)

    def tick(self):
        if self.step == 0:
            self.get_logger().info("ARM")
            self.send_cmd(400, 1)  # VEHICLE_CMD_COMPONENT_ARM_DISARM
            self.step += 1

        elif self.step == 1:
            self.get_logger().info("SET AUTO MODE")
            # VEHICLE_CMD_DO_SET_MODE = 176
            # param1=1 (custom), param2=4 (AUTO)
            self.send_cmd(176, 1, 4)
            self.step += 1

        elif self.step == 2:
            self.get_logger().info("NAV_TAKEOFF")
            # VEHICLE_CMD_NAV_TAKEOFF = 22
            # param7: hedef irtifa (MSL). Simde çoğu zaman "current + X" gibi davranmaz.
            # O yüzden önce global position'a bakarak anlamlı bir MSL seçmek gerekir.
            # Şimdilik 50 m deneyelim; işe yaramazsa aşağıdaki bölümde düzeltiyoruz.
            self.send_cmd(22, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 50.0)
            self.step += 1

        else:
            # canlı kalsın
            pass

def main():
    rclpy.init()
    n = FWTakeoff()
    rclpy.spin(n)
    n.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

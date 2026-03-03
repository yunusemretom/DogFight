#!/usr/bin/env python3
import rclpy    
from rclpy.node import Node

class CamNode(Node):
    def __init__(self):
        super().__init__('cam')
        self.get_logger().info('Camera node has been initialized.')
        self.timer = self.create_timer(1.0, self.timer_callback)


    def timer_callback(self):
        self.get_logger().info('Camera node is running.')
        
def main(args=None):
    rclpy.init(args=args)
    node = CamNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
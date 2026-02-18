#!/usr/bin/env python3
import rclpy    
from rclpy.node import Node

def main(args=None):
    rclpy.init(args=args)
    node = Node('cam')
    node.get_logger().info('Camera node has been started.')
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
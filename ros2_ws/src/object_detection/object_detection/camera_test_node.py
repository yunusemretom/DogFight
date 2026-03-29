import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraSubscriber(Node):

    def __init__(self):
        super().__init__('camera_subscriber')

        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            '/world/default/model/rc_cessna_1/link/camera_link/sensor/camera/image',
            self.image_callback,
            10)

    def image_callback(self, msg):

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        cv2.imshow("camera", frame)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)

    node = CameraSubscriber()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
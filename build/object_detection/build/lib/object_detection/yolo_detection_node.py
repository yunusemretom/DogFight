#!/usr/bin/env python

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from geometry_msgs.msg import Point
import cv2
from ultralytics import YOLO
import time

class YoloDetectionNode(Node):

    def __init__(self):
        super().__init__('yolo_detection_node')
        
        # QoS profile
        qos_profile_pub = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Publisher - merkeze uzaklık verilerini yayınla (x=distance_x, y=distance_y, z=confidence)
        self.distance_publisher = self.create_publisher(
            Point, 
            '/yolo/target_distance', 
            qos_profile_pub
        )
        
        # YOLO modeli ve kamera
        self.model = YOLO("/home/tom/Downloads/best(1).pt")
        self.cap = cv2.VideoCapture(0)
        self.score_threshold = 0.5
        
        # Timer - 50Hz (0.02 saniye)
        timer_period = 0.02
        self.timer = self.create_timer(timer_period, self.detection_callback)
        
        self.get_logger().info('YOLO Detection Node başlatıldı')
    
    def detection_callback(self):
        ret, frame = self.cap.read()
        frame = cv2.flip(frame, 1)
        if not ret:
            self.get_logger().warn('Kamera görüntüsü alınamadı')
            return
        
        # Görüntünün merkez noktasını hesapla
        frame_height, frame_width = frame.shape[:2]
        frame_center_x = frame_width // 2
        frame_center_y = frame_height // 2
        
        # YOLO ile nesne tespiti
        results = self.model(frame, stream=True)
        
        detection_found = False
        
        for result in results:
            for box in result.boxes:
                if box.conf > self.score_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Tespit edilen nesnenin merkez noktasını hesapla
                    box_center_x = (x1 + x2) // 2
                    box_center_y = (y1 + y2) // 2
                    
                    # Merkeze olan uzaklığı hesapla
                    distance_x = box_center_x - frame_center_x
                    distance_y = box_center_y - frame_center_y
                    
                    # Mesaj oluştur ve yayınla
                    msg = Point()
                    msg.x = float(distance_x)
                    msg.y = float(distance_y)
                    msg.z = float(box.conf[0].item())  # Confidence değeri
                    
                    self.distance_publisher.publish(msg)
                    detection_found = True
                    
                    # Görselleştirme
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"Class: {int(box.cls[0])}, Score: {box.conf[0].item():.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    distance_label = f"dx: {distance_x}, dy: {distance_y}"
                    cv2.putText(frame, distance_label, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    
                    self.get_logger().info(f'Tespit: dx={distance_x}, dy={distance_y}, conf={box.conf[0].item():.2f}')
                    
                    # İlk tespiti kullan
                    break
            if detection_found:
                break
        
        # Eğer tespit yoksa sıfır değerleri yayınla
        if not detection_found:
            msg = Point()
            msg.x = 0.0
            msg.y = 0.0
            msg.z = 0.0
            self.distance_publisher.publish(msg)
        
        # Görüntüyü göster
        cv2.imshow('YOLOv8 ROS Detection', frame)
        cv2.waitKey(1)
    
    def destroy_node(self):
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

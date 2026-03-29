import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import time
import supervision as sv
from rfdetr import RFDETRLarge,RFDETRNano
from rfdetr.assets.coco_classes import COCO_CLASSES

# 1. Modelin Başlatılması
# RF-DETR mimarisi, transformer tabanlı bir nesne tespit modelidir.

class CameraSubscriber(Node):

    def __init__(self):
        super().__init__('camera_subscriber')

        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            '/world/default/model/rc_cessna_1/link/camera_link/sensor/camera/image',
            self.image_callback,
            10)
        self.model = RFDETRNano(pretrain_weights="/home/tom/Downloads/checkpoint_best_regular_kaggle.pth")
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()


    def image_callback(self, msg):

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        detections = self.model.predict(frame, threshold=0.3)

        # 4. Etiketlerin Hazırlanması (Düzeltilmiş Kısım)
        # detections.class_id ve detections.confidence listelerini 'zip' ile eşleştiriyoruz.
        # Bu sayede her tespite tam olarak 1 etiket karşılık gelir (N=N eşleşmesi).
        labels = [
            f"{COCO_CLASSES[class_id]} {confidence:.2f}" 
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]

        # 6. Görüntü Üzerine Çizim Yapılması
        # .copy() kullanımı orijinal görselin bozulmasını engeller.
        annotated_image = self.box_annotator.annotate(
            scene=frame.copy(), 
            detections=detections
        )

        annotated_image = self.label_annotator.annotate(
            scene=annotated_image, 
            detections=detections, 
            labels=labels
        )

        
        cv2.imshow("camera", annotated_image)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)

    node = CameraSubscriber()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
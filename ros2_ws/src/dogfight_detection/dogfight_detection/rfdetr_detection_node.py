import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import time
import numpy as np
import supervision as sv
from rfdetr import RFDETRLarge,RFDETRNano,RFDETRSmall
from rfdetr.assets.coco_classes import COCO_CLASSES

# 1. Modelin Başlatılması
# RF-DETR mimarisi, transformer tabanlı bir nesne tespit modelidir.

class CameraSubscriber(Node):

    def __init__(self):
        super().__init__('camera_subscriber')

        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            '/world/default/model/rc_cessna_mono_cam_0/link/camera_link/sensor/camera/image',
            self.image_callback,
            10)
        self.model = RFDETRSmall(
            pretrain_weights="/home/tom/Downloads/checkpoint_best_regular (09.08.16).pth",
            resolution=640,
            num_classes=2
        )
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

        # FPS Göstergesi
        self.last_time = time.perf_counter()
        self.fps = 0.0

        # SAHI Parametreleri
        self.slice_width = 640
        self.slice_height = 640
        self.overlap_ratio = 0.2
        self.nms_iou_threshold = 0.5
        self.conf_threshold = 0.3
        self.resize_scale = 0.7  # Görsel küçültme oranı (FPS artırmak için)
        self.video_writer = None  # Video kayıt aracı

        # Kenetlenme (Lock-on) Durumu
        self.lock_start_time = None
        self.lock_duration = 0.0

    def iter_slices(self, image_height: int, image_width: int):
        step_x = max(int(self.slice_width * (1.0 - self.overlap_ratio)), 1)
        step_y = max(int(self.slice_height * (1.0 - self.overlap_ratio)), 1)
        seen = set()

        for y_start in range(0, image_height, step_y):
            y_end = min(y_start + self.slice_height, image_height)
            y_start = max(0, y_end - self.slice_height)

            for x_start in range(0, image_width, step_x):
                x_end = min(x_start + self.slice_width, image_width)
                x_start = max(0, x_end - self.slice_width)
                key = (x_start, y_start, x_end, y_end)
                if key in seen:
                    continue
                seen.add(key)
                yield key

    def class_aware_nms(
        self,
        xyxy: np.ndarray,
        scores: np.ndarray,
        classes: np.ndarray,
        score_threshold: float,
        iou_threshold: float,
    ) -> np.ndarray:
        keep_indices = []

        for cls in np.unique(classes):
            cls_indices = np.where(classes == cls)[0]
            cls_boxes = xyxy[cls_indices]
            cls_scores = scores[cls_indices].tolist()

            cls_boxes_xywh = np.column_stack(
                (
                    cls_boxes[:, 0],
                    cls_boxes[:, 1],
                    cls_boxes[:, 2] - cls_boxes[:, 0],
                    cls_boxes[:, 3] - cls_boxes[:, 1],
                )
            ).tolist()

            selected = cv2.dnn.NMSBoxes(
                bboxes=cls_boxes_xywh,
                scores=cls_scores,
                score_threshold=score_threshold,
                nms_threshold=iou_threshold,
            )

            if selected is None or len(selected) == 0:
                continue

            selected = np.array(selected).reshape(-1).astype(int)
            keep_indices.extend(cls_indices[selected].tolist())

        if not keep_indices:
            return np.empty((0,), dtype=np.int32)

        return np.array(sorted(set(keep_indices)), dtype=np.int32)

    def sahi_predict(self, image: np.ndarray) -> sv.Detections:
        frame_h, frame_w = image.shape[:2]

        all_xyxy = []
        all_conf = []
        all_cls = []

        for x_start, y_start, x_end, y_end in self.iter_slices(frame_h, frame_w):
            tile = image[y_start:y_end, x_start:x_end]
            tile_detections = self.model.predict(tile, threshold=self.conf_threshold)

            if len(tile_detections) == 0:
                continue

            tile_xyxy = tile_detections.xyxy.astype(np.float32).copy()
            tile_xyxy[:, [0, 2]] += x_start
            tile_xyxy[:, [1, 3]] += y_start

            all_xyxy.append(tile_xyxy)
            all_conf.append(tile_detections.confidence.astype(np.float32))
            all_cls.append(tile_detections.class_id.astype(np.int32))

        if not all_xyxy:
            return sv.Detections(
                xyxy=np.empty((0, 4), dtype=np.float32),
                confidence=np.empty((0,), dtype=np.float32),
                class_id=np.empty((0,), dtype=np.int32),
            )

        xyxy = np.concatenate(all_xyxy, axis=0)
        merged_scores = np.concatenate(all_conf, axis=0)
        merged_classes = np.concatenate(all_cls, axis=0)

        selected_indices = self.class_aware_nms(
            xyxy=xyxy,
            scores=merged_scores,
            classes=merged_classes,
            score_threshold=max(self.conf_threshold * 0.5, 0.01),
            iou_threshold=self.nms_iou_threshold,
        )

        if selected_indices.size == 0:
            return sv.Detections(
                xyxy=np.empty((0, 4), dtype=np.float32),
                confidence=np.empty((0,), dtype=np.float32),
                class_id=np.empty((0,), dtype=np.int32),
            )

        return sv.Detections(
            xyxy=xyxy[selected_indices],
            confidence=merged_scores[selected_indices],
            class_id=merged_classes[selected_indices],
        )

    def image_callback(self, msg):
        # FPS Hesaplama
        now = time.perf_counter()
        dt = now - self.last_time
        self.last_time = now
        fps_val = 1.0 / dt if dt > 0 else 0.0
        smoothing = 0.9
        self.fps = (smoothing * self.fps) + ((1.0 - smoothing) * fps_val) if self.fps > 0 else fps_val

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # Ekran görüntüsünü (frame) hızlandırmak için bir tık küçült
        if self.resize_scale != 1.0:
            frame = cv2.resize(frame, (0, 0), fx=self.resize_scale, fy=self.resize_scale)

        # SAHI ile Nesne Tespiti
        detections = self.sahi_predict(frame)

        # 4. Etiketlerin Hazırlanması
        labels = []
        for class_id, confidence in zip(detections.class_id, detections.confidence):
            class_id = int(class_id)
            if 0 <= class_id < len(COCO_CLASSES):
                labels.append(f"{COCO_CLASSES[class_id]} {confidence:.2f}")
            else:
                labels.append(f"id:{class_id} {confidence:.2f}")

        # 6. Görüntü Üzerine Çizim Yapılması
        annotated_image = self.box_annotator.annotate(
            scene=frame.copy(), 
            detections=detections
        )

        annotated_image = self.label_annotator.annotate(
            scene=annotated_image, 
            detections=detections, 
            labels=labels
        )

        # Görüntü Boyutları
        height, width = frame.shape[:2]

        # Hedef Bounding Box (%10 Alt/Üst, %25 Sağ/Sol Boşluk)
        y_min = int(height * 0.10)
        y_max = int(height * 0.90)
        x_min = int(width * 0.25)
        x_max = int(width * 0.75)

        # Merkezden hedefe doğru vektör çizimi
        cx, cy = width // 2, height // 2
        # Merkez artı işareti (crosshair)
        cv2.drawMarker(annotated_image, (cx, cy), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=16, thickness=2)

        target_in_zone = False

        if len(detections) > 0:
            frame_center = np.array([cx, cy], dtype=np.float32)
            closest_idx = -1
            min_dist = float('inf')
            
            for i, box in enumerate(detections.xyxy):
                tx = (box[0] + box[2]) / 2
                ty = (box[1] + box[3]) / 2
                dist = np.linalg.norm(np.array([tx, ty]) - frame_center)
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i
            
            if closest_idx != -1:
                box = detections.xyxy[closest_idx]
                tx = int((box[0] + box[2]) / 2)
                ty = int((box[1] + box[3]) / 2)
                
                # Hedef merkezinin hedef karesi içinde olup olmadığını kontrol et
                if x_min <= tx <= x_max and y_min <= ty <= y_max:
                    target_in_zone = True

                # Hedefe giden vektör (ok)
                cv2.arrowedLine(annotated_image, (cx, cy), (tx, ty), (255, 0, 0), 2, tipLength=0.1)
                
                # Vektörün üstüne piksel mesafesini yaz
                cv2.putText(
                    annotated_image,
                    f"{int(min_dist)}px",
                    (int((cx + tx)/2) + 10, int((cy + ty)/2) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA
                )

        # Kenetlenme Süresi Güncelleme
        if target_in_zone:
            if self.lock_start_time is None:
                self.lock_start_time = time.perf_counter()
                self.lock_duration = 0.0
            else:
                self.lock_duration = time.perf_counter() - self.lock_start_time
        else:
            self.lock_start_time = None
            self.lock_duration = 0.0

        # Kenetlenme HUD Göstergesi (Merkez nişangahın hemen altına)
        if self.lock_duration > 0:
            if self.lock_duration >= 2.0:
                lock_text = f"LOCKED: {self.lock_duration:.1f}s"
                color = (0, 255, 0)  # Yeşil
            else:
                lock_text = f"LOCKING: {self.lock_duration:.1f}s"
                color = (0, 165, 255)  # Turuncu
        else:
            lock_text = "NO LOCK"
            color = (0, 0, 255)  # Kırmızı

        cv2.putText(
            annotated_image,
            lock_text,
            (cx - 60, cy + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA
        )

        # Hedef Bounding Box (%10 Alt/Üst, %25 Sağ/Sol Boşluk)
        cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

        # HUD Üst Bar (Yarı saydam siyah arka plan)
        overlay = annotated_image.copy()
        cv2.rectangle(overlay, (0, 0), (width, 40), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, annotated_image, 0.5, 0, annotated_image)

        # FPS Göstergesi (HUD Sol)
        cv2.putText(
            annotated_image,
            f"FPS: {self.fps:.1f}",
            (15, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),  # Yeşil
            2,
            cv2.LINE_AA,
        )

        # Durum Saati (HUD Orta)
        current_time_str = time.strftime("%Y-%m-%d %H:%M:%S")
        text_size = cv2.getTextSize(current_time_str, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = (width - text_size[0]) // 2
        cv2.putText(
            annotated_image,
            current_time_str,
            (text_x, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),  # Beyaz
            2,
            cv2.LINE_AA,
        )

        # Hedef ve Kenetlenme Bilgisi (HUD Sağ)
        target_str = f"LOCK: {self.lock_duration:.1f}s | TGT: {len(detections)}"
        text_size_tgt = cv2.getTextSize(target_str, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.putText(
            annotated_image,
            target_str,
            (width - text_size_tgt[0] - 15, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),  # Cyan
            2,
            cv2.LINE_AA,
        )

        # Video kaydı
        if self.video_writer is None:
            output_h, output_w = annotated_image.shape[:2]
            filename = f"detection_record_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
            self.video_writer = cv2.VideoWriter(
                filename,
                cv2.VideoWriter_fourcc(*'mp4v'),
                15.0,  # 15 FPS
                (output_w, output_h)
            )
            self.get_logger().info(f"Video kaydı başladı: {filename}")
        
        if self.video_writer is not None:
            self.video_writer.write(annotated_image)

        cv2.imshow("camera", annotated_image)
        cv2.waitKey(1)


    def destroy_node(self):
        if hasattr(self, 'video_writer') and self.video_writer is not None:
            self.video_writer.release()
            self.get_logger().info("Video kaydı başarıyla sonlandırıldı.")
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = CameraSubscriber()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt algılandı, kapatılıyor...")
    finally:
        node.destroy_node()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

if __name__ == '__main__':
    main()
import time

import cv2
import mss
import numpy as np
import supervision as sv
from rfdetr import RFDETRSmall
from rfdetr.assets.coco_classes import COCO_CLASSES
COCO_CLASSES = {
        1: "drone",
        2: "f16",
        3: "helicopter",
        4: "rocket",
        5: "missile"}

MODEL_WEIGHTS = "/home/tom/Downloads/checkpoint_best_regular_celik_2.pth"
CONF_THRESHOLD = 0.5
RESIZE_SCALE = 0.5
MONITOR_INDEX = 1  # 1 = ana ekran
OUTPUT_FPS = 30.0
OUTPUT_PATH = f"/home/tom/Documents/DogFight/ekran_tespit_{time.strftime('%Y%m%d_%H%M%S')}.mp4"


model = RFDETRSmall(pretrain_weights=MODEL_WEIGHTS)
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

prev_time = time.perf_counter()
fps = 0.0
fps_smoothing = 0.9


with mss.mss() as sct:
    if MONITOR_INDEX >= len(sct.monitors):
        raise RuntimeError(f"Gecersiz monitor indexi: {MONITOR_INDEX}. Mevcut monitor sayisi: {len(sct.monitors) - 1}")

    monitor = sct.monitors[MONITOR_INDEX]
    video_writer = None

    while True:
        screenshot = sct.grab(monitor)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        if RESIZE_SCALE != 1.0:
            frame = cv2.resize(frame, (0, 0), fx=RESIZE_SCALE, fy=RESIZE_SCALE)

        now = time.perf_counter()
        dt = now - prev_time
        if dt > 0:
            current_fps = 1.0 / dt
            fps = (fps_smoothing * fps) + ((1.0 - fps_smoothing) * current_fps) if fps > 0 else current_fps
        prev_time = now

        detections = model.predict(frame, threshold=CONF_THRESHOLD)

        labels = []
        for class_id, confidence in zip(detections.class_id, detections.confidence):
            class_id = int(class_id)
            if 0 <= class_id < len(COCO_CLASSES):
                labels.append(f"{COCO_CLASSES[class_id]} {confidence:.2f}")
            else:
                labels.append(f"id:{class_id} {confidence:.2f}")

        annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels,
        )

        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            annotated_frame,
            f"Nesne: {len(detections)}",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if video_writer is None:
            output_height, output_width = annotated_frame.shape[:2]
            video_writer = cv2.VideoWriter(
                OUTPUT_PATH,
                cv2.VideoWriter_fourcc(*"mp4v"),
                OUTPUT_FPS,
                (output_width, output_height),
            )
            if not video_writer.isOpened():
                raise RuntimeError(f"Video kaydi baslatilamadi: {OUTPUT_PATH}")

        video_writer.write(annotated_frame)

        cv2.imshow("RF-DETR Ekran Tespiti", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

if video_writer is not None:
    video_writer.release()
cv2.destroyAllWindows()
print(f"Kayit tamamlandi: {OUTPUT_PATH}")
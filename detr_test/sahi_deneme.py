import time

import cv2
import mss
import numpy as np
import supervision as sv
from rfdetr import RFDETRSmall
from rfdetr.assets.coco_classes import COCO_CLASSES
"""COCO_CLASSES = {
        1: "drone",
        2: "f16",
        3: "helicopter",
        4: "rocket",
        5: "missile"}"""


MODEL_WEIGHTS = "/home/tom/Downloads/checkpoint_best_regular_colab_4.pth"
CONF_THRESHOLD = 0.5
RESIZE_SCALE = .5
MONITOR_INDEX = 1  # 1 = ana ekran
OUTPUT_FPS = 30.0
OUTPUT_PATH = f"/home/tom/Documents/DogFight/ekran_tespit_sahi_{time.strftime('%Y%m%d_%H%M%S')}.mp4"

# SAHI benzeri dilimleme parametreleri
SLICE_WIDTH = 640
SLICE_HEIGHT = 640
OVERLAP_RATIO = 0.2
NMS_IOU_THRESHOLD = 0.5


model = RFDETRSmall(pretrain_weights=MODEL_WEIGHTS)
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()


def empty_detections() -> sv.Detections:
    return sv.Detections(
        xyxy=np.empty((0, 4), dtype=np.float32),
        confidence=np.empty((0,), dtype=np.float32),
        class_id=np.empty((0,), dtype=np.int32),
    )


def iter_slices(image_height: int, image_width: int):
    step_x = max(int(SLICE_WIDTH * (1.0 - OVERLAP_RATIO)), 1)
    step_y = max(int(SLICE_HEIGHT * (1.0 - OVERLAP_RATIO)), 1)
    seen = set()

    for y_start in range(0, image_height, step_y):
        y_end = min(y_start + SLICE_HEIGHT, image_height)
        y_start = max(0, y_end - SLICE_HEIGHT)

        for x_start in range(0, image_width, step_x):
            x_end = min(x_start + SLICE_WIDTH, image_width)
            x_start = max(0, x_end - SLICE_WIDTH)
            key = (x_start, y_start, x_end, y_end)
            if key in seen:
                continue
            seen.add(key)
            yield key


def class_aware_nms(
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


def sahi_predict(image: np.ndarray) -> sv.Detections:
    frame_h, frame_w = image.shape[:2]

    all_xyxy = []
    all_conf = []
    all_cls = []

    for x_start, y_start, x_end, y_end in iter_slices(frame_h, frame_w):
        tile = image[y_start:y_end, x_start:x_end]
        tile_detections = model.predict(tile, threshold=CONF_THRESHOLD)

        if len(tile_detections) == 0:
            continue

        tile_xyxy = tile_detections.xyxy.astype(np.float32).copy()
        tile_xyxy[:, [0, 2]] += x_start
        tile_xyxy[:, [1, 3]] += y_start

        all_xyxy.append(tile_xyxy)
        all_conf.append(tile_detections.confidence.astype(np.float32))
        all_cls.append(tile_detections.class_id.astype(np.int32))

    if not all_xyxy:
        return empty_detections()

    xyxy = np.concatenate(all_xyxy, axis=0)
    merged_scores = np.concatenate(all_conf, axis=0)
    merged_classes = np.concatenate(all_cls, axis=0)

    selected_indices = class_aware_nms(
        xyxy=xyxy,
        scores=merged_scores,
        classes=merged_classes,
        score_threshold=max(CONF_THRESHOLD * 0.5, 0.01),
        iou_threshold=NMS_IOU_THRESHOLD,
    )

    if selected_indices.size == 0:
        return empty_detections()

    return sv.Detections(
        xyxy=xyxy[selected_indices],
        confidence=merged_scores[selected_indices],
        class_id=merged_classes[selected_indices],
    )


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

        detections = sahi_predict(frame)

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

        cv2.putText(
            annotated_frame,
            "Mod: SAHI slicing",
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
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
        cv2.imshow("RF-DETR SAHI Ekran Tespiti", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

if video_writer is not None:
    video_writer.release()

cv2.destroyAllWindows()
print(f"Kayit tamamlandi: {OUTPUT_PATH}")

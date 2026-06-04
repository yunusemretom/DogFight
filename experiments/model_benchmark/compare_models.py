import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from rfdetr import RFDETRSmall
from rfdetr.assets.coco_classes import COCO_CLASSES
from ultralytics import YOLO

# TensorRT COCO sınıfları (kustom)
COCO_CLASSES_TRT = {
    0: "plane", 1: "plane", 2: "car", 3: "motorbike", 4: "aeroplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "cat", 15: "dog", 16: "horse", 17: "sheep", 18: "cow", 19: "elephant",
    20: "bear", 21: "zebra", 22: "giraffe", 23: "backpack", 24: "umbrella",
    25: "handbag", 26: "tie", 27: "suitcase", 28: "frisbee", 29: "skis",
    30: "snowboard", 31: "sports ball", 32: "kite", 33: "baseball bat",
    34: "baseball glove", 35: "skateboard", 36: "surfboard", 37: "tennis racket",
    38: "bottle", 39: "wine glass", 40: "cup", 41: "fork", 42: "knife",
    43: "spoon", 44: "bowl", 45: "banana", 46: "apple", 47: "sandwich",
    48: "orange", 49: "broccoli", 50: "carrot", 51: "hot dog", 52: "pizza",
    53: "donut", 54: "cake", 55: "chair", 56: "sofa", 57: "pottedplant",
    58: "bed", 59: "diningtable", 60: "toilet", 61: "tvmonitor", 62: "laptop",
    63: "mouse", 64: "remote", 65: "keyboard", 66: "microwave", 67: "oven",
    68: "toaster", 69: "sink", 70: "refrigerator", 71: "book", 72: "clock",
    73: "vase", 74: "scissors", 75: "teddy bear", 76: "hair drier",
    77: "toothbrush",
}


def parse_size(size_text: str) -> tuple[int, int]:
    try:
        w_text, h_text = size_text.lower().split("x")
        w = int(w_text)
        h = int(h_text)
        if w <= 0 or h <= 0:
            raise ValueError
        return w, h
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--output-size formati WxH olmali. Ornek: 1920x1080"
        ) from exc


def fit_with_letterbox(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    src_h, src_w = frame.shape[:2]
    scale = min(target_w / src_w, target_h / src_h)
    new_w = max(1, int(src_w * scale))
    new_h = max(1, int(src_h * scale))

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def put_panel_title(frame: np.ndarray, title: str) -> None:
    cv2.rectangle(frame, (0, 0), (430, 42), (0, 0, 0), thickness=-1)
    cv2.putText(
        frame,
        title,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def put_fps_top_right(frame: np.ndarray, fps_text: str) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.85
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(fps_text, font, scale, thickness)

    margin = 10
    x1 = frame.shape[1] - margin
    y1 = margin + text_h
    x0 = x1 - text_w - 16
    y0 = margin

    cv2.rectangle(
        frame,
        (max(0, x0), max(0, y0)),
        (min(frame.shape[1] - 1, x1), min(frame.shape[0] - 1, y1 + baseline + 6)),
        (0, 0, 0),
        thickness=-1,
    )
    cv2.putText(
        frame,
        fps_text,
        (x0 + 8, y1),
        font,
        scale,
        (0, 255, 0),
        thickness,
        cv2.LINE_AA,
    )


def run_yolo(
    input_video: Path,
    weights_path: Path,
    output_video: Path,
    conf_threshold: float,
    preview: bool,
) -> None:
    print("[1/6] YOLO calisiyor...")
    model = YOLO(str(weights_path))

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"YOLO: video acilamadi -> {input_video}")

    fps_in = cap.get(cv2.CAP_PROP_FPS)
    fps_out = fps_in if fps_in and fps_in > 0 else 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_out,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"YOLO: cikti videosu olusturulamadi -> {output_video}")

    fps_smooth = 0.0
    alpha = 0.9

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.perf_counter()
        results = model.predict(frame, conf=conf_threshold, verbose=False)

        annotated = frame.copy()
        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                if conf < conf_threshold:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label_name = model.names.get(cls_id, str(cls_id))
                label = f"{label_name} {conf:.2f}"

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 230, 0), 2)
                cv2.putText(
                    annotated,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 230, 0),
                    2,
                    cv2.LINE_AA,
                )

        infer_time = max(1e-6, time.perf_counter() - start)
        current_fps = 1.0 / infer_time
        fps_smooth = (alpha * fps_smooth) + ((1.0 - alpha) * current_fps) if fps_smooth > 0 else current_fps

        put_fps_top_right(annotated, f"FPS: {fps_smooth:.1f}")

        writer.write(annotated)

        if preview:
            cv2.imshow("YOLO", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    writer.release()
    if preview:
        cv2.destroyWindow("YOLO")
    print(f"YOLO bitti -> {output_video}")


def run_yolo_tensorrt(
    input_video: Path,
    engine_path: Path,
    output_video: Path,
    conf_threshold: float,
    preview: bool,
) -> None:
    print("[2a/6] YOLO TensorRT calisiyor...")
    model = YOLO(str(engine_path))

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"YOLO TensorRT: video acilamadi -> {input_video}")

    fps_in = cap.get(cv2.CAP_PROP_FPS)
    fps_out = fps_in if fps_in and fps_in > 0 else 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_out,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"YOLO TensorRT: cikti videosu olusturulamadi -> {output_video}")

    fps_smooth = 0.0
    alpha = 0.9

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.perf_counter()
        results = model.predict(frame, conf=conf_threshold, verbose=False)

        annotated = frame.copy()
        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                if conf < conf_threshold:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label_name = model.names.get(cls_id, str(cls_id))
                label = f"{label_name} {conf:.2f}"

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 100, 0), 2)
                cv2.putText(
                    annotated,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 100, 0),
                    2,
                    cv2.LINE_AA,
                )

        infer_time = max(1e-6, time.perf_counter() - start)
        current_fps = 1.0 / infer_time
        fps_smooth = (alpha * fps_smooth) + ((1.0 - alpha) * current_fps) if fps_smooth > 0 else current_fps

        put_fps_top_right(annotated, f"FPS: {fps_smooth:.1f}")

        writer.write(annotated)

        if preview:
            cv2.imshow("YOLO TensorRT", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    writer.release()
    if preview:
        cv2.destroyWindow("YOLO TensorRT")
    print(f"YOLO TensorRT bitti -> {output_video}")


def run_rfdetr(
    input_video: Path,
    weights_path: Path,
    output_video: Path,
    conf_threshold: float,
    preview: bool,
) -> None:
    print("[3/6] RF-DETR calisiyor...")
    model = RFDETRSmall(pretrain_weights=str(weights_path))

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"RF-DETR: video acilamadi -> {input_video}")

    fps_in = cap.get(cv2.CAP_PROP_FPS)
    fps_out = fps_in if fps_in and fps_in > 0 else 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_out,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"RF-DETR: cikti videosu olusturulamadi -> {output_video}")

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    fps_smooth = 0.0
    alpha = 0.9

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.perf_counter()
        detections = model.predict(frame, threshold=conf_threshold)

        labels = [
            f"{COCO_CLASSES[int(class_id)]} {float(conf):.2f}"
            for class_id, conf in zip(detections.class_id, detections.confidence)
        ]

        annotated = box_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated = label_annotator.annotate(
            scene=annotated,
            detections=detections,
            labels=labels,
        )

        infer_time = max(1e-6, time.perf_counter() - start)
        current_fps = 1.0 / infer_time
        fps_smooth = (alpha * fps_smooth) + ((1.0 - alpha) * current_fps) if fps_smooth > 0 else current_fps

        put_fps_top_right(annotated, f"FPS: {fps_smooth:.1f}")

        writer.write(annotated)

        if preview:
            cv2.imshow("RF-DETR", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    writer.release()
    if preview:
        cv2.destroyWindow("RF-DETR")
    print(f"RF-DETR bitti -> {output_video}")


def run_rfdetr_tensorrt(
    input_video: Path,
    engine_path: Path,
    output_video: Path,
    conf_threshold: float,
    preview: bool,
) -> None:
    print("[4/6] RF-DETR TensorRT calisiyor...")
    print(f"[4/6] Kullanilan RF-DETR engine: {engine_path}")
    model_error = None

    # RFDETRTensorRT modeli yüklemeyi dene
    try:
        from detr_test.Tensorrt_ciktisi.modeli_dene import RFDETRTensorRT
        model = RFDETRTensorRT(str(engine_path))
    except Exception as e:
        model_error = str(e)
        print(f"[UYARI] RF-DETR TensorRT baslatilamadi: {e}")
        print("Bu adim placeholder/fallback video ile devam edecek.")
        model = None

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"RF-DETR TensorRT: video acilamadi -> {input_video}")

    fps_in = cap.get(cv2.CAP_PROP_FPS)
    fps_out = fps_in if fps_in and fps_in > 0 else 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_out,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"RF-DETR TensorRT: cikti videosu olusturulamadi -> {output_video}")

    fps_smooth = 0.0
    alpha = 0.9
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.perf_counter()
        
        if model is not None:
            detections = model.predict(frame, conf_threshold)
            labels = [
                f"{COCO_CLASSES_TRT.get(int(class_id), str(int(class_id)))} {float(conf):.2f}"
                for class_id, conf in zip(detections.class_id, detections.confidence)
            ]
            annotated = box_annotator.annotate(scene=frame.copy(), detections=detections)
            annotated = label_annotator.annotate(
                scene=annotated,
                detections=detections,
                labels=labels,
            )
        else:
            annotated = frame.copy()
            cv2.putText(
                annotated,
                "TensorRT unavailable",
                (12, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            if model_error:
                cv2.putText(
                    annotated,
                    "Engine version mismatch or deserialize error",
                    (12, 102),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

        infer_time = max(1e-6, time.perf_counter() - start)
        current_fps = 1.0 / infer_time
        fps_smooth = (alpha * fps_smooth) + ((1.0 - alpha) * current_fps) if fps_smooth > 0 else current_fps

        put_fps_top_right(annotated, f"FPS: {fps_smooth:.1f}")

        writer.write(annotated)

        if preview:
            cv2.imshow("RF-DETR TensorRT", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    writer.release()
    if preview:
        cv2.destroyWindow("RF-DETR TensorRT")
    print(f"RF-DETR TensorRT bitti -> {output_video}")


def combine_side_by_side(
    left_video: Path,
    right_video: Path,
    output_video: Path,
    output_size: tuple[int, int],
) -> None:
    print("[5/6] Yan yana video olusturuluyor...")
    out_w, out_h = output_size
    cell_w = out_w // 2
    cell_h = out_h

    cap_l = cv2.VideoCapture(str(left_video))
    cap_r = cv2.VideoCapture(str(right_video))

    if not cap_l.isOpened() or not cap_r.isOpened():
        raise RuntimeError("Yan yana birlestirme icin giris videolari acilamadi.")

    fps_l = cap_l.get(cv2.CAP_PROP_FPS)
    fps_r = cap_r.get(cv2.CAP_PROP_FPS)
    fps = min(x for x in [fps_l, fps_r] if x and x > 0) if (fps_l > 0 and fps_r > 0) else 30.0

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (out_w, out_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Yan yana cikti olusturulamadi -> {output_video}")

    while True:
        ret_l, frame_l = cap_l.read()
        ret_r, frame_r = cap_r.read()
        if not ret_l or not ret_r:
            break

        left = fit_with_letterbox(frame_l, cell_w, cell_h)
        right = fit_with_letterbox(frame_r, cell_w, cell_h)

        put_panel_title(left, "YOLO (Normal)")
        put_panel_title(right, "RF-DETR (Normal)")

        combined = np.hstack([left, right])
        writer.write(combined)

    cap_l.release()
    cap_r.release()
    writer.release()
    print(f"Yan yana video hazir -> {output_video}")


def make_placeholder_frame(w: int, h: int, title: str) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    put_panel_title(frame, title)
    put_fps_top_right(frame, "FPS: N/A")
    cv2.putText(
        frame,
        "TensorRT videosu verilmedi.",
        (30, h // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (180, 180, 180),
        2,
        cv2.LINE_AA,
    )
    return frame


def combine_four_grid(
    normal_yolo_video: Path,
    normal_rfdetr_video: Path,
    output_video: Path,
    output_size: tuple[int, int],
    trt_yolo_video: Path | None = None,
    trt_rfdetr_video: Path | None = None,
) -> None:
    print("[6/6] 4'lu grid video olusturuluyor...")
    out_w, out_h = output_size
    cell_w = out_w // 2
    cell_h = out_h // 2

    cap_tl = cv2.VideoCapture(str(normal_yolo_video))
    cap_tr = cv2.VideoCapture(str(normal_rfdetr_video))
    if not cap_tl.isOpened() or not cap_tr.isOpened():
        raise RuntimeError("4'lu grid icin normal videolar acilamadi.")

    cap_bl = cv2.VideoCapture(str(trt_yolo_video)) if trt_yolo_video and trt_yolo_video.exists() else None
    cap_br = cv2.VideoCapture(str(trt_rfdetr_video)) if trt_rfdetr_video and trt_rfdetr_video.exists() else None

    fps_candidates = [cap_tl.get(cv2.CAP_PROP_FPS), cap_tr.get(cv2.CAP_PROP_FPS)]
    fps_candidates = [x for x in fps_candidates if x and x > 0]
    fps = min(fps_candidates) if fps_candidates else 30.0

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (out_w, out_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"4'lu grid cikti olusturulamadi -> {output_video}")

    placeholder_bl = make_placeholder_frame(cell_w, cell_h, "YOLO (TensorRT)")
    placeholder_br = make_placeholder_frame(cell_w, cell_h, "RF-DETR (TensorRT)")

    while True:
        ret_tl, frame_tl = cap_tl.read()
        ret_tr, frame_tr = cap_tr.read()
        if not ret_tl or not ret_tr:
            break

        top_left = fit_with_letterbox(frame_tl, cell_w, cell_h)
        top_right = fit_with_letterbox(frame_tr, cell_w, cell_h)
        put_panel_title(top_left, "YOLO (Normal)")
        put_panel_title(top_right, "RF-DETR (Normal)")

        if cap_bl and cap_bl.isOpened():
            ret_bl, frame_bl = cap_bl.read()
            if ret_bl:
                bottom_left = fit_with_letterbox(frame_bl, cell_w, cell_h)
                put_panel_title(bottom_left, "YOLO (TensorRT)")
            else:
                bottom_left = placeholder_bl
        else:
            bottom_left = placeholder_bl

        if cap_br and cap_br.isOpened():
            ret_br, frame_br = cap_br.read()
            if ret_br:
                bottom_right = fit_with_letterbox(frame_br, cell_w, cell_h)
                put_panel_title(bottom_right, "RF-DETR (TensorRT)")
            else:
                bottom_right = placeholder_br
        else:
            bottom_right = placeholder_br

        top_row = np.hstack([top_left, top_right])
        bottom_row = np.hstack([bottom_left, bottom_right])
        grid = np.vstack([top_row, bottom_row])
        writer.write(grid)

    cap_tl.release()
    cap_tr.release()
    if cap_bl:
        cap_bl.release()
    if cap_br:
        cap_br.release()
    writer.release()
    print(f"4'lu grid video hazir -> {output_video}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="YOLO ve RF-DETR ciktilarini karsilastirmali video olarak uretir."
    )
    parser.add_argument(
        "--input-video",
        type=Path,
        required=True,
        help="Giris video yolu",
    )
    parser.add_argument(
        "--yolo-weights",
        type=Path,
        default=Path("/home/tom/Downloads/epoch40.pt"),
        help="YOLO agirlik dosyasi (.pt)",
    )
    parser.add_argument(
        "--yolo-engine",
        type=Path,
        default=Path("/home/tom/Downloads/epoch40.engine"),
        help="YOLO TensorRT engine dosyasi (.engine)",
    )
    parser.add_argument(
        "--rfdetr-weights",
        type=Path,
        default=Path("/home/tom/Downloads/checkpoint_best_regular_2.pth"),
        help="RF-DETR agirlik dosyasi (.pth)",
    )
    parser.add_argument(
        "--rfdetr-engine",
        type=Path,
        default=Path("/home/tom/Documents/Projeler/DogFight/object_detection/inference_model.engine"),
        help="RF-DETR TensorRT engine dosyasi (.engine)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("compare_outputs"),
        help="Tum ciktilarin kaydedilecegi klasor",
    )
    parser.add_argument(
        "--output-size",
        type=parse_size,
        default=(1920, 1080),
        help="Nihai cikti boyutu, ornek: 1920x1080",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Guven skoru esigi",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Islem sirasinda model ciktilarini pencerede goster",
    )

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    yolo_out = args.output_dir / "01_yolo_normal.mp4"
    yolo_trt_out = args.output_dir / "02_yolo_tensorrt.mp4"
    rfdetr_out = args.output_dir / "03_rfdetr_normal.mp4"
    rfdetr_trt_out = args.output_dir / "04_rfdetr_tensorrt.mp4"
    side_by_side_out = args.output_dir / "05_yanyana_1920x1080.mp4"
    grid_out = args.output_dir / "06_grid_2x2_1920x1080.mp4"

    run_yolo(
        input_video=args.input_video,
        weights_path=args.yolo_weights,
        output_video=yolo_out,
        conf_threshold=args.conf,
        preview=args.preview,
    )

    # YOLO TensorRT (engine varsa)
    if args.yolo_engine and args.yolo_engine.exists():
        run_yolo_tensorrt(
            input_video=args.input_video,
            engine_path=args.yolo_engine,
            output_video=yolo_trt_out,
            conf_threshold=args.conf,
            preview=args.preview,
        )
    else:
        print("[2a/6] YOLO TensorRT atlanıyor (engine bulunamadı)...")
        yolo_trt_out = None

    run_rfdetr(
        input_video=args.input_video,
        weights_path=args.rfdetr_weights,
        output_video=rfdetr_out,
        conf_threshold=args.conf,
        preview=args.preview,
    )

    # RF-DETR TensorRT (engine varsa)
    if args.rfdetr_engine and args.rfdetr_engine.exists():
        run_rfdetr_tensorrt(
            input_video=args.input_video,
            engine_path=args.rfdetr_engine,
            output_video=rfdetr_trt_out,
            conf_threshold=args.conf,
            preview=args.preview,
        )
    else:
        print("[4/6] RF-DETR TensorRT atlanıyor (engine bulunamadı)...")
        rfdetr_trt_out = None

    combine_side_by_side(
        left_video=yolo_out,
        right_video=rfdetr_out,
        output_video=side_by_side_out,
        output_size=args.output_size,
    )

    combine_four_grid(
        normal_yolo_video=yolo_out,
        normal_rfdetr_video=rfdetr_out,
        trt_yolo_video=yolo_trt_out,
        trt_rfdetr_video=rfdetr_trt_out,
        output_video=grid_out,
        output_size=args.output_size,
    )

    print("\nTum islemler tamamlandi.")
    print(f"YOLO (Normal)           : {yolo_out}")
    if yolo_trt_out:
        print(f"YOLO (TensorRT)         : {yolo_trt_out}")
    print(f"RF-DETR (Normal)        : {rfdetr_out}")
    if rfdetr_trt_out:
        print(f"RF-DETR (TensorRT)      : {rfdetr_trt_out}")
    print(f"Yan yana cikti          : {side_by_side_out}")
    print(f"4'lu grid cikti         : {grid_out}")


if __name__ == "__main__":
    main()

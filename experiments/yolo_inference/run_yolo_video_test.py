import time

import cv2
from ultralytics import YOLO


def put_fps_top_right(frame, fps_text):
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


model = YOLO("/home/tom/Downloads/epoch40.pt")
cap = cv2.VideoCapture("/home/tom/Downloads/istockphoto-2179703365-640_adpp_is.mp4")

if not cap.isOpened():
    raise RuntimeError("Video acilamadi. Dosya yolunu kontrol edin.")

fps_in = cap.get(cv2.CAP_PROP_FPS)
fps_out = fps_in if fps_in and fps_in > 0 else 30.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    "output_video.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps_out,
    (width, height),
)
if not out.isOpened():
    raise RuntimeError("Cikti videosu olusturulamadi.")

score_threshold = 0.5
fps_smooth = 0.0
alpha = 0.9

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start = time.perf_counter()
    # Karsilastirma scripti ile ayni davranis icin track yerine predict kullaniyoruz.
    results = model.predict(frame, conf=score_threshold, verbose=False)

    annotated = frame.copy()
    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            if conf < score_threshold:
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

    cv2.imshow("YOLOv8 TensorRT Detection", annotated)
    out.write(annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

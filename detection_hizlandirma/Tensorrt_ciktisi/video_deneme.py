"""
RF-DETR TensorRT - Gerçek Zamanlı Webcam / Kamera Akışı
Adım 4 (isteğe bağlı): Kamera beslemesinde canlı nesne tespiti.

Kullanım:
    python step4_realtime.py                 # webcam (0)
    python step4_realtime.py --source rtsp://... # IP kamera
    python step4_realtime.py --source video.mp4  # video dosyası

Çıkış: 'q' tuşuna basarak pencereyi kapat.
"""

import argparse
import time
import cv2
import numpy as np
from modeli_dene import RFDETRTensorRT  # önceki modülden

try:
    from rfdetr.assets.coco_classes import COCO_CLASSES
    CLASS_NAMES = list(COCO_CLASSES.values())
except ImportError:
    CLASS_NAMES = None


def run(engine_path: str, source: int | str, threshold: float):
    detector = RFDETRTensorRT(engine_path)
    cap      = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Kaynak açılamadı: {source}")

    fps_buf = []
    print("Başlatılıyor... 'q' ile çık.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0         = time.perf_counter()
        detections = detector.predict(frame, threshold)
        fps_buf.append(1.0 / max(time.perf_counter() - t0, 1e-6))
        if len(fps_buf) > 30:
            fps_buf.pop(0)

        # Sonuçları çiz
        annotated = detector._annotate(frame, detections, CLASS_NAMES)

        # FPS göstergesi
        fps_text = f"FPS: {np.mean(fps_buf):.1f}"
        cv2.putText(
            annotated, fps_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
        )

        cv2.imshow("RF-DETR TensorRT", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nOrtalama FPS: {np.mean(fps_buf):.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine",    default="exports/rfdetr.engine")
    parser.add_argument("--source",    default=0, help="webcam id veya video/rtsp yolu")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    # Kaynak: kamera id mi yoksa yol mu?
    try:
        source = int(args.source)
    except (ValueError, TypeError):
        source = args.source

    run(args.engine, source, args.threshold)
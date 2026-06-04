#!/usr/bin/env python3
"""
RF-DETR + TCTrack entegre takip pipeline'i.

Akis:
1) RF-DETR ile hedef tespit edilir.
2) Secilen hedef bbox ile TCTrack baslatilir.
3) Sonraki frame'lerde TCTrack ile dusuk gecikmeli ve temporal tutarli takip surdurulur.
4) Tracker bbox'u kilitlenme dortgeni (AH) olarak cizilir ve PID hata sinyali hesaplanir.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, cast

import cv2 as _cv2
import numpy as np
import torch

from rfdetr import RFDETRLarge, RFDETRNano, RFDETRSmall

cv2 = cast(Any, _cv2)

LOCK_DISPLAY_SPEED = 10


@dataclass
class TrackState:
    bbox_xywh: Optional[tuple[float, float, float, float]] = None
    score: float = 0.0
    lock_elapsed: float = 0.0
    window_elapsed: float = 0.0


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    default_rfdetr_weights = project_root / "rf-detr-large-2026.pth"
    default_tc_config = project_root / "TCTrack" / "experiments" / "TCTrack" / "config.yaml"
    default_tc_snapshot = project_root / "TCTrack" / "snapshot" / "general_model.pth"

    parser = argparse.ArgumentParser(description="RF-DETR + TCTrack entegrasyonu")
    parser.add_argument("--source", type=str, default="0", help="Kamera index'i (0 gibi) veya video dosya yolu")
    parser.add_argument("--rfdetr-variant", type=str, default="large", choices=["nano", "small", "large"], help="RF-DETR model varyanti")
    parser.add_argument("--rfdetr-weights", type=Path, default=default_rfdetr_weights, help="RF-DETR agirlik yolu")
    parser.add_argument("--det-threshold", type=float, default=0.45, help="RF-DETR confidence threshold")
    parser.add_argument("--redetect-every", type=int, default=12, help="Her N frame'de bir RF-DETR ile yeniden dogrulama")

    parser.add_argument("--tctrack-config", type=Path, default=default_tc_config, help="TCTrack config.yaml yolu")
    parser.add_argument("--tctrack-snapshot", type=Path, default=default_tc_snapshot, help="TCTrack agirlik (.pth) yolu")
    parser.add_argument("--track-score-threshold", type=float, default=0.18, help="TCTrack skoru bunun altina inerse yeniden tespit denenir")

    parser.add_argument("--class-id", type=int, default=None, help="Yalnizca bu class_id takip edilir (opsiyonel)")
    parser.add_argument("--show", action="store_true", help="Canli pencere goster")
    parser.add_argument("--output", type=Path, default=None, help="Kayit cikti videosu (opsiyonel)")
    parser.add_argument("--max-frames", type=int, default=0, help="Test amacli frame limiti (0=sinirsiz)")
    return parser.parse_args()


def add_tctrack_to_path(project_root: Path) -> None:
    tctrack_root = project_root / "TCTrack"
    if str(tctrack_root) not in sys.path:
        sys.path.insert(0, str(tctrack_root))


def build_rfdetr(variant: str, weights: Path):
    if variant == "nano":
        return RFDETRNano(pretrain_weights=str(weights))
    if variant == "small":
        return RFDETRSmall(pretrain_weights=str(weights))
    return RFDETRLarge(pretrain_weights=str(weights))


def to_xywh(xyxy: np.ndarray) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    return (x1, y1, x2 - x1, y2 - y1)


def center_of_xywh(b: tuple[float, float, float, float]) -> tuple[float, float]:
    return (b[0] + b[2] * 0.5, b[1] + b[3] * 0.5)


def select_detection(detections, class_id: Optional[int], prev_bbox_xywh: Optional[tuple[float, float, float, float]]):
    if len(detections) == 0:
        return None

    best_idx = None
    best_value = -1e18

    prev_center = center_of_xywh(prev_bbox_xywh) if prev_bbox_xywh is not None else None

    for idx, (xyxy, conf, cid) in enumerate(zip(detections.xyxy, detections.confidence, detections.class_id)):
        cid_int = int(cid)
        if class_id is not None and cid_int != class_id:
            continue

        score = float(conf)
        if prev_center is not None:
            cx = float((xyxy[0] + xyxy[2]) * 0.5)
            cy = float((xyxy[1] + xyxy[3]) * 0.5)
            dist = np.hypot(cx - prev_center[0], cy - prev_center[1])
            score -= 0.0008 * dist

        if score > best_value:
            best_value = score
            best_idx = idx

    if best_idx is None:
        return None
    return best_idx


def draw_reference_zones(frame: np.ndarray) -> tuple[int, int, int, int]:
    h, w = frame.shape[:2]

    # AV: Hedef vurus alani (soldan/sagdan %25, ustten/alttan %10 offset)
    av_x1 = int(w * 0.25)
    av_y1 = int(h * 0.10)
    av_x2 = int(w * 0.75)
    av_y2 = int(h * 0.90)
    cv2.rectangle(frame, (av_x1, av_y1), (av_x2, av_y2), (0, 255, 255), 2)

    return (av_x1, av_y1, av_x2, av_y2)


def draw_track_overlay(frame: np.ndarray, bbox_xywh: tuple[float, float, float, float], av_rect: tuple[int, int, int, int]) -> dict[str, float]:
    h, w = frame.shape[:2]
    x, y, bw, bh = bbox_xywh
    x1, y1 = int(x), int(y)
    x2, y2 = int(x + bw), int(y + bh)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cx = x + bw * 0.5
    cy = y + bh * 0.5
    err_x = (cx - (w * 0.5)) / (w * 0.5)
    err_y = (cy - (h * 0.5)) / (h * 0.5)

    width_ratio = bw / max(w, 1)
    height_ratio = bh / max(h, 1)

    av_x1, av_y1, av_x2, av_y2 = av_rect
    inside_av = x1 >= av_x1 and y1 >= av_y1 and x2 <= av_x2 and y2 <= av_y2
    size_ok = (width_ratio >= 0.05) and (height_ratio >= 0.05)
    lock_ok = inside_av and size_ok

   
    return {
        "err_x": float(err_x),
        "err_y": float(err_y),
        "inside_av": float(1 if inside_av else 0),
        "size_ok": float(1 if size_ok else 0),
        "lock_ok": float(1 if lock_ok else 0),
        "width_ratio": float(width_ratio),
        "height_ratio": float(height_ratio),
    }


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    add_tctrack_to_path(project_root)

    from pysot.core.config import cfg  # type: ignore[import-not-found]
    from pysot.models.utile_tctrack.model_builder import ModelBuilder_tctrack  # type: ignore[import-not-found]
    from pysot.tracker.tctrack_tracker import TCTrackTracker  # type: ignore[import-not-found]
    from pysot.utils.model_load import load_pretrain  # type: ignore[import-not-found]

    if not args.rfdetr_weights.exists():
        raise FileNotFoundError(f"RF-DETR agirlik bulunamadi: {args.rfdetr_weights}")
    if not args.tctrack_config.exists():
        raise FileNotFoundError(f"TCTrack config bulunamadi: {args.tctrack_config}")
    if not args.tctrack_snapshot.exists():
        raise FileNotFoundError(f"TCTrack agirlik bulunamadi: {args.tctrack_snapshot}")

    detector = build_rfdetr(args.rfdetr_variant, args.rfdetr_weights)

    cfg.merge_from_file(str(args.tctrack_config))
    cfg.CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if cfg.CUDA else "cpu")

    tc_model = ModelBuilder_tctrack("test")
    tc_model = load_pretrain(tc_model, str(args.tctrack_snapshot)).eval().to(device)
    tracker = TCTrackTracker(tc_model)
    hp = [cfg.TRACK.PENALTY_K, cfg.TRACK.WINDOW_INFLUENCE, cfg.TRACK.LR]

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Kaynak acilamadi: {args.source}")

    writer = None
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    fps_ema = 0.0
    t_prev = time.perf_counter()

    state = TrackState()
    tracker_initialized = False

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1

        t_now = time.perf_counter()
        dt = t_now - t_prev
        if dt > 0:
            inst_fps = 1.0 / dt
            fps_ema = inst_fps if fps_ema <= 0 else (0.9 * fps_ema + 0.1 * inst_fps)
        t_prev = t_now

        need_detect = (not tracker_initialized) or (frame_idx % max(args.redetect_every, 1) == 0) or (state.score < args.track_score_threshold)

        if need_detect:
            detections = detector.predict(frame, threshold=args.det_threshold)
            selected = select_detection(detections, args.class_id, state.bbox_xywh)
            if selected is not None:
                new_bbox = to_xywh(detections.xyxy[selected])
                tracker.init(frame, new_bbox)
                tracker_initialized = True
                state = TrackState(
                    bbox_xywh=new_bbox,
                    score=float(detections.confidence[selected]),
                    lock_elapsed=0.0,
                    window_elapsed=0.0,
                )

        if tracker_initialized:
            tracked = tracker.track(frame, hp)
            state.bbox_xywh = tuple(float(v) for v in tracked["bbox"])
            state.score = float(tracked["best_score"])

        av_rect = draw_reference_zones(frame)
        if state.bbox_xywh is not None:
            pid_debug = draw_track_overlay(frame, state.bbox_xywh, av_rect)
            lock_ok = bool(pid_debug["lock_ok"])
            state.window_elapsed += dt
            if lock_ok:
                state.lock_elapsed += dt * LOCK_DISPLAY_SPEED

            if state.window_elapsed >= 10.0:
                state.window_elapsed = 0.0
                state.lock_elapsed = 0.0

            cv2.putText(frame, f"Kilit: {state.lock_elapsed:.1f} sn", (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            if lock_ok:
                cv2.putText(frame, "Kilitlenme aktif", (10, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        cv2.putText(frame, f"FPS: {fps_ema:.1f}", (10, frame.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        if writer is None and args.output is not None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
            if not writer.isOpened():
                raise RuntimeError(f"Cikti videosu olusturulamadi: {args.output}")

        if writer is not None:
            writer.write(frame)

        if args.show:
            cv2.imshow("RF-DETR + TCTrack", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        if args.max_frames > 0 and frame_idx >= args.max_frames:
            break

    cap.release()
    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

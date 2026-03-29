"""
RF-DETR TensorRT Inference
Adım 3: Derlenmiş .engine dosyasıyla gerçek zamanlı nesne tespiti.

Gereksinimler:
    pip install tensorrt pycuda numpy opencv-python Pillow supervision
"""

import os
import time
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt
import pycuda.autoinit          # GPU context başlatır
import pycuda.driver as cuda
from PIL import Image
import supervision as sv

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# RF-DETR giriş boyutu (eğitim sırasında kullanılan çözünürlük)
INPUT_H = 384
INPUT_W = 384

# ImageNet normalize sabitleri
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
class RFDETRTensorRT:
    """
    RF-DETR modelini TensorRT engine ile çalıştıran inference sınıfı.

    Kullanım:
        detector = RFDETRTensorRT("exports/rfdetr.engine")
        detections = detector.predict(image, threshold=0.5)
    """

    def __init__(self, engine_path: str):
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"Engine bulunamadı: {engine_path}")

        self.engine   = self._load_engine(engine_path)
        self.context  = self.engine.create_execution_context()
        self.stream   = cuda.Stream()
        self._allocate_buffers()
        print(f"✅ TensorRT engine yüklendi: {engine_path}")

    # ── Engine Yükleme ────────────────────────────────────────────────────
    @staticmethod
    def _load_engine(path: str) -> trt.ICudaEngine:
        runtime = trt.Runtime(TRT_LOGGER)
        with open(path, "rb") as f:
            return runtime.deserialize_cuda_engine(f.read())

    # ── Bellek Tahsisi ────────────────────────────────────────────────────
    def _allocate_buffers(self):
        """Giriş/çıkış tensor'ları için GPU/CPU belleği tahsis eder."""
        self.inputs  = []
        self.outputs = []
        self.bindings = []

        for i in range(self.engine.num_io_tensors):
            name  = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = self.engine.get_tensor_shape(name)
            size  = trt.volume(shape)

            # Sayfalanmış (pinned) host belleği → GPU'ya hızlı transfer
            host_mem   = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))
            info = {"name": name, "host": host_mem, "device": device_mem, "shape": shape}

            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.inputs.append(info)
            else:
                self.outputs.append(info)

        print("Tensörler:")
        for t in self.inputs:
            print(f"  Giriş  → {t['name']} {t['shape']}")
        for t in self.outputs:
            print(f"  Çıkış  → {t['name']} {t['shape']}")

    # ── Görüntü Ön İşleme ────────────────────────────────────────────────
    @staticmethod
    def preprocess(image: np.ndarray) -> np.ndarray:
        """
        BGR numpy görüntüsünü model girişine hazırlar.
        Çıkış: float32[1, 3, H, W]  (normalize edilmiş)
        """
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = (img - MEAN) / STD                  # ImageNet normalize
        img = img.transpose(2, 0, 1)              # HWC → CHW
        img = np.ascontiguousarray(img[np.newaxis])  # CHW → 1CHW
        return img

    # ── TensorRT Çalıştırma ───────────────────────────────────────────────
    def _infer(self, input_tensor: np.ndarray) -> list:
        """Ham TensorRT çalıştırma; ham çıkış listesi döner."""
        # Veriyi giriş buffer'ına kopyala
        np.copyto(self.inputs[0]["host"], input_tensor.ravel())

        # CPU → GPU
        cuda.memcpy_htod_async(
            self.inputs[0]["device"],
            self.inputs[0]["host"],
            self.stream
        )

        # Context'e tensor adreslerini bağla (TensorRT 10 API)
        for t in self.inputs + self.outputs:
            self.context.set_tensor_address(t["name"], int(t["device"]))

        # Inference
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # GPU → CPU
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)

        self.stream.synchronize()
        return [out["host"].reshape(out["shape"]) for out in self.outputs]

    # ── Son İşleme (Post-process) ─────────────────────────────────────────
    @staticmethod
    def postprocess(
        raw_outputs: list,
        orig_h: int,
        orig_w: int,
        threshold: float = 0.5,
    ) -> sv.Detections:
        """
        Model çıkışlarını supervision.Detections nesnesine dönüştürür.

        RF-DETR çıkış formatı:
          dets   : [1, num_queries, 4]  - cxcywh (normalize [0-1])
          labels : [1, num_queries, num_classes]  - sınıf logitleri
        """
        dets_raw   = raw_outputs[0][0]   # [num_queries, 4]
        labels_raw = raw_outputs[1][0]   # [num_queries, num_classes]

        # Sınıf olasılıkları (sigmoid)
        scores_per_class = 1 / (1 + np.exp(-labels_raw))   # sigmoid
        class_ids = scores_per_class.argmax(axis=-1)        # [num_queries]
        scores    = scores_per_class[np.arange(len(class_ids)), class_ids]

        # Eşik filtresi
        keep = scores >= threshold
        if keep.sum() == 0:
            return sv.Detections.empty()

        boxes_cxcywh = dets_raw[keep]    # normalize merkez-xywh
        class_ids    = class_ids[keep]
        scores       = scores[keep]

        # cxcywh (normalize) → xyxy (piksel)
        cx = boxes_cxcywh[:, 0] * orig_w
        cy = boxes_cxcywh[:, 1] * orig_h
        w  = boxes_cxcywh[:, 2] * orig_w
        h  = boxes_cxcywh[:, 3] * orig_h

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        xyxy = np.stack([x1, y1, x2, y2], axis=1)
        xyxy = np.clip(xyxy, [0, 0, 0, 0], [orig_w, orig_h, orig_w, orig_h])

        return sv.Detections(
            xyxy=xyxy,
            class_id=class_ids.astype(np.int32),
            confidence=scores,
        )

    # ── Tek Görüntü Tahmini ───────────────────────────────────────────────
    def predict(
        self,
        image: np.ndarray,
        threshold: float = 0.5,
    ) -> sv.Detections:
        """
        Bir görüntü üzerinde nesne tespiti yapar.

        Args:
            image     : BGR numpy array (OpenCV formatı)
            threshold : Güven eşiği (0-1 arası)

        Returns:
            supervision.Detections
        """
        orig_h, orig_w = image.shape[:2]
        tensor = self.preprocess(image)
        raw    = self._infer(tensor)
        return self.postprocess(raw, orig_h, orig_w, threshold)

    # ── Görüntü Dosyasında Tahmin ─────────────────────────────────────────
    def predict_image(
        self,
        image_path: str,
        threshold: float = 0.5,
        output_path: str | None = None,
        class_names: list[str] | None = None,
    ) -> sv.Detections:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Görüntü açılamadı: {image_path}")

        t0 = time.perf_counter()
        detections = self.predict(image, threshold)
        ms = (time.perf_counter() - t0) * 1000

        print(f"Tespit sayısı: {len(detections)}  |  Süre: {ms:.1f} ms")

        if output_path:
            annotated = self._annotate(image, detections, class_names)
            cv2.imwrite(output_path, annotated)
            print(f"Sonuç kaydedildi: {output_path}")

        return detections

    # ── Video İşleme ──────────────────────────────────────────────────────
    def predict_video(
        self,
        video_path: str,
        output_path: str,
        threshold: float = 0.5,
        class_names: list[str] | None = None,
    ):
        """Video üzerinde kare kare nesne tespiti."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Video açılamadı: {video_path}")

        fps    = cap.get(cv2.CAP_PROP_FPS)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_idx = 0
        total_ms  = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.perf_counter()
            detections = self.predict(frame, threshold)
            total_ms  += (time.perf_counter() - t0) * 1000

            annotated = self._annotate(frame, detections, class_names)
            writer.write(annotated)
            frame_idx += 1

            if frame_idx % 50 == 0:
                avg = total_ms / frame_idx
                print(f"  Kare {frame_idx}/{total}  |  Ort. {avg:.1f} ms/kare")

        cap.release()
        writer.release()
        avg = total_ms / max(frame_idx, 1)
        print(f"\n✅ Video kaydedildi: {output_path}")
        print(f"   Ortalama: {avg:.1f} ms/kare  |  FPS: {1000/avg:.1f}")

    # ── Görselleştirme ────────────────────────────────────────────────────
    @staticmethod
    def _annotate(
        image: np.ndarray,
        detections: sv.Detections,
        class_names: list[str] | None = None,
    ) -> np.ndarray:
        if len(detections) == 0:
            return image

        labels = None
        if class_names and detections.class_id is not None:
            labels = [
                f"{class_names[cid]} {conf:.2f}"
                for cid, conf in zip(detections.class_id, detections.confidence)
                if 0 <= cid < len(class_names)
            ]

        annotated = sv.BoxAnnotator().annotate(image.copy(), detections)
        if labels:
            annotated = sv.LabelAnnotator().annotate(annotated, detections, labels)
        return annotated

    # ── Bağlam Yöneticisi ────────────────────────────────────────────────
    def __enter__(self):
        return self

    def __exit__(self, *_):
        del self.context
        del self.engine


# ─────────────────────────────────────────────────────────────────────────────
# Hız Testi
# ─────────────────────────────────────────────────────────────────────────────
def benchmark(engine_path: str, n_runs: int = 200, warmup: int = 20):
    """TensorRT engine gecikme karşılaştırması."""
    from rfdetr.assets.coco_classes import COCO_CLASSES  # isteğe bağlı

    detector = RFDETRTensorRT(engine_path)
    dummy    = np.random.randint(0, 255, (INPUT_H, INPUT_W, 3), dtype=np.uint8)

    print(f"\nIsınma turu ({warmup} kez)...")
    for _ in range(warmup):
        detector.predict(dummy)

    print(f"Ölçüm ({n_runs} kez)...")
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        detector.predict(dummy)
        times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    print(f"\n{'─'*40}")
    print(f"Ortalama   : {times.mean():.2f} ms")
    print(f"Medyan     : {np.median(times):.2f} ms")
    print(f"P99        : {np.percentile(times, 99):.2f} ms")
    print(f"Min / Max  : {times.min():.2f} / {times.max():.2f} ms")
    print(f"FPS        : {1000/times.mean():.1f}")
    print(f"{'─'*40}")


# ─────────────────────────────────────────────────────────────────────────────
# Ana Program
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ENGINE = "exports/rfdetr.engine"

    # COCO sınıf isimlerini yükle (isteğe bağlı)
    try:
        from rfdetr.assets.coco_classes import COCO_CLASSES
        class_names = list(COCO_CLASSES.values())
    except ImportError:
        class_names = None

    # ── Görüntü üzerinde test ─────────────────────────────────────────────
    with RFDETRTensorRT(ENGINE) as detector:
        detections = detector.predict_image(
            image_path="test_image.jpg",
            threshold=0.5,
            output_path="result.jpg",
            class_names=class_names,
        )
        print(detections)

    # ── Hız testi ─────────────────────────────────────────────────────────
    benchmark(ENGINE)
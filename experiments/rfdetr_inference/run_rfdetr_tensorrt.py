"""
RF-DETR TensorRT Inference - Ekran Yakalama
TensorRT 11 / Python 10 uyumlu

Gereksinimler:
    pip install tensorrt cuda-python opencv-python-headless mss numpy
    (tensorrt: sistem kurulumunu .venv icine baglayabilirsin - bkz. asagidaki not)

Not: TRT Python modulu .venv icinde yoksa, sisteme kurulu olanı kullanmak icin:
    export PYTHONPATH=/usr/lib/python3/dist-packages:$PYTHONPATH
    ya da: pip install tensorrt-cu12 --extra-index-url https://pypi.nvidia.com

Kullanim:
    python run_rfdetr_tensorrt.py
    (q tusuna basarak cikis)
"""

import time
from pathlib import Path

import cv2
import mss
import numpy as np

# ─── Yapilandirma ────────────────────────────────────────────────────────────
ENGINE_PATH    = "object_detection/inference_model.engine"
CONF_THRESHOLD = 0.45
RESIZE_SCALE   = 0.8          # ekran boyutu carpani (performans/kalite dengesi)
MONITOR_INDEX  = 1            # 1 = ana ekran
RECORD_VIDEO   = True         # False = video kaydi yapma (daha hizli)
OUTPUT_FPS     = 50.0
OUTPUT_PATH    = f"/home/tom/Documents/Projeler/DogFight/ekran_tespit_trt_{time.strftime('%Y%m%d_%H%M%S')}.mp4"



# RF-DETR model input boyutu (export sirasinda kullanilan deger)
MODEL_INPUT_SIZE = 560  # RFDETRSmall default: 560x560

# Sinif tanimlari (egitim sirasindaki siniflar)
CLASS_NAMES = {
    0: "drone",
    1: "f16",
    2: "helicopter",
    3: "rocket",
    4: "missile",
}

# Annotasyon renkleri (BGR)
BOX_COLOR   = (0, 220, 90)
LABEL_COLOR = (255, 255, 255)
LABEL_BG    = (0, 150, 60)
# ─────────────────────────────────────────────────────────────────────────────


# ─── TensorRT Yukleme ────────────────────────────────────────────────────────
try:
    import tensorrt as trt
    import cuda
    from cuda import cudart
    HAS_CUDA_PYTHON = True
except ImportError:
    HAS_CUDA_PYTHON = False

try:
    import pycuda.driver as cuda_drv
    import pycuda.autoinit  # noqa: F401
    HAS_PYCUDA = True
except ImportError:
    HAS_PYCUDA = False

if not (HAS_CUDA_PYTHON or HAS_PYCUDA):
    raise ImportError(
        "CUDA Python binding bulunamadi!\n"
        "Kur: pip install pycuda   VEYA   pip install cuda-python"
    )

try:
    import tensorrt as trt
except ImportError:
    raise ImportError(
        "TensorRT Python modulu bulunamadi!\n"
        "Sisteme kuruluysa: export PYTHONPATH=/usr/lib/python3/dist-packages\n"
        "Ya da: pip install tensorrt-cu12 --extra-index-url https://pypi.nvidia.com"
    )
# ─────────────────────────────────────────────────────────────────────────────


class TRTInferencer:
    """TensorRT engine ile FP16/FP32 inference."""

    def __init__(self, engine_path: str):
        path = Path(engine_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Engine dosyasi bulunamadi: {path.resolve()}\n"
                "Once export_rfdetr_tensorrt.py ile export yapin."
            )

        self.logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(self.logger)

        print(f"[INFO] TRT engine yukleniyor: {path.resolve()}")
        with open(path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError("Engine deserialize edilemedi!")

        self.context = self.engine.create_execution_context()

        # Tensor isimlerini al
        self.input_names  = []
        self.output_names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        print(f"[INFO] Inputs : {self.input_names}")
        print(f"[INFO] Outputs: {self.output_names}")

        # CUDA bellegi ayir (pycuda yolu)
        if HAS_PYCUDA:
            self._init_pycuda()
        else:
            self._init_cuda_python()

        # Onislem icin normalizasyon sabitlerini onceden ayir
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self._inv_std = (1.0 / self._std).astype(np.float32)
        self._neg_mean_div_std = (-self._mean / self._std).astype(np.float32)

        # Onislem icin yeniden kullanilabilir buffer
        _, _, H, W = self._input_shape
        self._pre_buf = np.empty((1, 3, H, W), dtype=np.float32)

        # Cikti key'lerini onceden belirle (decode_outputs icin)
        self._dets_key = None
        self._labels_key = None

        print("[INFO] TRT Inferencer hazir.")

    # ── pycuda yolu ──────────────────────────────────────────────────────────
    def _init_pycuda(self):
        self._use_pycuda = True
        self._host_inputs  = {}
        self._host_outputs = {}
        self._cuda_inputs   = {}
        self._cuda_outputs  = {}

        # Input: [1, 3, H, W]
        in_name = self.input_names[0]
        shape = tuple(self.engine.get_tensor_shape(in_name))  # (1, 3, 560, 560)
        size  = int(np.prod(shape))
        self._input_shape = shape
        self._host_inputs[in_name]  = cuda_drv.pagelocked_empty(size, dtype=np.float32)
        self._cuda_inputs[in_name]  = cuda_drv.mem_alloc(self._host_inputs[in_name].nbytes)

        # Outputs
        for out_name in self.output_names:
            shape = tuple(self.engine.get_tensor_shape(out_name))
            size  = int(np.prod(np.abs(shape)))  # dinamik boyutlara karsi abs
            host  = cuda_drv.pagelocked_empty(size, dtype=np.float32)
            self._host_outputs[out_name] = host
            self._cuda_outputs[out_name] = cuda_drv.mem_alloc(host.nbytes)
            setattr(self, f"_shape_{out_name}", shape)

        self._stream = cuda_drv.Stream()

    # ── cuda-python yolu ─────────────────────────────────────────────────────
    def _init_cuda_python(self):
        self._use_pycuda = False
        raise NotImplementedError("cuda-python yolu henuz desteklenmiyor, pycuda kurun.")

    # ── Onislem ──────────────────────────────────────────────────────────────
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """BGR image -> float32 CHW [0,1] normalized, model input boyutuna boyutlandir."""
        _, _, H, W = self._input_shape
        img = cv2.resize(image, (W, H))
        # BGR->RGB + float32 + normalize: tek geciste
        img = img[:, :, ::-1].astype(np.float32)  # BGR->RGB (numpy slice, cvtColor'dan hizli)
        img *= (1.0 / 255.0)
        # (img - mean) / std  =  img * inv_std + neg_mean_div_std  (onceden hesaplandi)
        img *= self._inv_std
        img += self._neg_mean_div_std
        # HWC -> NCHW direkt buffer'a yaz
        self._pre_buf[0] = img.transpose(2, 0, 1)
        return self._pre_buf

    # ── Inference ────────────────────────────────────────────────────────────
    def infer(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """Tek bir tile/frame uzerinde inference yap. dict{output_name: array} doner."""
        inp = self._preprocess(image)

        if self._use_pycuda:
            return self._infer_pycuda(inp)

    def _infer_pycuda(self, inp: np.ndarray) -> dict[str, np.ndarray]:
        in_name = self.input_names[0]
        np.copyto(self._host_inputs[in_name], inp.ravel())

        # Host -> Device
        cuda_drv.memcpy_htod_async(self._cuda_inputs[in_name], self._host_inputs[in_name], self._stream)

        # Tensor adreslerini bağla
        self.context.set_tensor_address(in_name, int(self._cuda_inputs[in_name]))
        for out_name in self.output_names:
            self.context.set_tensor_address(out_name, int(self._cuda_outputs[out_name]))

        self.context.execute_async_v3(self._stream.handle)

        # Device -> Host
        for out_name in self.output_names:
            cuda_drv.memcpy_dtoh_async(self._host_outputs[out_name], self._cuda_outputs[out_name], self._stream)
        self._stream.synchronize()

        # Reshape sonucu view olarak don (.copy() kaldirildi - performans)
        results = {}
        for out_name in self.output_names:
            shape = getattr(self, f"_shape_{out_name}")
            results[out_name] = self._host_outputs[out_name].reshape(shape)

        return results

    def __del__(self):
        try:
            if hasattr(self, "_stream"):
                self._stream.synchronize()
        except Exception:
            pass


# ─── Sonuc Islemleri ─────────────────────────────────────────────────────────
def decode_outputs(outputs: dict, orig_h: int, orig_w: int, conf_threshold: float,
                   inferencer=None):
    """
    RF-DETR ONNX ciktilarini decode et.
    Cikti tensörleri:
        dets   : (1, 300, 4) - cxcywh formatinda, normalize edilmis [0,1]
        labels : (1, 300, num_classes) - logit skorlari
    """
    # Cikti isimlerini bul (onbellekli)
    if inferencer is not None and inferencer._dets_key is not None:
        dets_key = inferencer._dets_key
        labels_key = inferencer._labels_key
    else:
        dets_key   = next((k for k in outputs if "det" in k.lower() or "box" in k.lower()), list(outputs.keys())[0])
        labels_key = next((k for k in outputs if "label" in k.lower() or "class" in k.lower() or "logit" in k.lower()), list(outputs.keys())[1])
        if inferencer is not None:
            inferencer._dets_key = dets_key
            inferencer._labels_key = labels_key

    dets   = outputs[dets_key][0]    # (300, 4)
    logits = outputs[labels_key][0]  # (300, num_classes)

    # Sigmoid + argmax + confidence - vektorize
    scores    = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
    class_ids = np.argmax(scores, axis=-1)       # (300,)
    confs     = scores[np.arange(len(class_ids)), class_ids]  # (300,)

    # Filtrele
    mask  = confs >= conf_threshold
    dets  = dets[mask]
    class_ids = class_ids[mask]
    confs = confs[mask]

    if len(dets) == 0:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)

    # cxcywh -> xyxy, piksel koordinatlarina cevir
    half_w = dets[:, 2] * 0.5
    half_h = dets[:, 3] * 0.5
    x1 = (dets[:, 0] - half_w) * orig_w
    y1 = (dets[:, 1] - half_h) * orig_h
    x2 = (dets[:, 0] + half_w) * orig_w
    y2 = (dets[:, 1] + half_h) * orig_h

    xyxy = np.stack([x1, y1, x2, y2], axis=-1)
    return xyxy, confs, class_ids





# ─── Annotasyon ──────────────────────────────────────────────────────────────
def annotate(frame, xyxy, confs, class_ids):
    """Frame uzerine dogrudan cizer (in-place, kopya yok)."""
    for box, conf, cid in zip(xyxy, confs, class_ids):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        label = f"{CLASS_NAMES.get(int(cid), f'id:{cid}')} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), LABEL_BG, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, LABEL_COLOR, 1, cv2.LINE_AA)
    return frame


def put_stats(frame, fps, n_det, mode="TRT"):
    cv2.putText(frame, f"FPS: {fps:.1f}",       (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0),   2, cv2.LINE_AA)
    cv2.putText(frame, f"Nesne: {n_det}",        (10, 65),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Mod: {mode}",           (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
    return frame


# ─── Ana Dongu ───────────────────────────────────────────────────────────────
def main():
    inferencer = TRTInferencer(ENGINE_PATH)

    prev_time   = time.perf_counter()
    fps         = 0.0
    fps_alpha   = 0.9
    video_writer = None

    with mss.MSS() as sct:
        if MONITOR_INDEX >= len(sct.monitors):
            raise RuntimeError(
                f"Gecersiz monitor indexi: {MONITOR_INDEX}. "
                f"Mevcut: {len(sct.monitors) - 1}"
            )

        monitor = sct.monitors[MONITOR_INDEX]
        print(f"[INFO] Monitor: {monitor}")
        print(f"[INFO] Cikti: {OUTPUT_PATH}")
        print("[INFO] Cikis icin pencerede 'q' tusuna basin.")

        while True:
            screenshot = sct.grab(monitor)
            # BGRA -> BGR: numpy slice (cvtColor'dan ~2x hizli)
            frame = np.asarray(screenshot)[:, :, :3].copy()

            if RESIZE_SCALE != 1.0:
                new_w = int(frame.shape[1] * RESIZE_SCALE)
                new_h = int(frame.shape[0] * RESIZE_SCALE)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            now = time.perf_counter()
            dt  = now - prev_time
            if dt > 0:
                cur_fps = 1.0 / dt
                fps = (fps_alpha * fps + (1.0 - fps_alpha) * cur_fps) if fps > 0 else cur_fps
            prev_time = now

            h, w = frame.shape[:2]
            outputs = inferencer.infer(frame)
            xyxy, confs, class_ids = decode_outputs(outputs, h, w, CONF_THRESHOLD,
                                                     inferencer=inferencer)

            annotate(frame, xyxy, confs, class_ids)  # in-place
            put_stats(frame, fps, len(xyxy))

            # Video yazar (opsiyonel)
            if RECORD_VIDEO:
                if video_writer is None:
                    oh, ow = frame.shape[:2]
                    video_writer = cv2.VideoWriter(
                        OUTPUT_PATH,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        OUTPUT_FPS,
                        (ow, oh),
                    )
                    if not video_writer.isOpened():
                        raise RuntimeError(f"Video kaydi baslatilamadi: {OUTPUT_PATH}")
                video_writer.write(frame)

            cv2.imshow("RF-DETR TensorRT", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Kayit tamamlandi: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

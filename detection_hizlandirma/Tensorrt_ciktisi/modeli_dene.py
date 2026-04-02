"""
RF-DETR TensorRT Inference
Adım 3: Derlenmiş .engine dosyasıyla gerçek zamanlı nesne tespiti.

TensorRT, NVIDIA GPU'larında derin öğrenme modellerini maksimum performansla (düşük gecikme, yüksek verim) 
çalıştırmak için ağırlıkları, katmanları ve bellek kullanımını donanıma özel olarak optimize eden bir SDK'dır.
"""

import os
import time
from pathlib import Path

import cv2
import numpy as np
# NVIDIA'nın yüksek performanslı derin öğrenme çıkarım (inference) motoru.
import tensorrt as trt
# PyCUDA, Python üzerinden doğrudan GPU belleğini (VRAM) ve işlem birimlerini kontrol etmemizi sağlar.
import pycuda.autoinit          # GPU context'ini (çalışma ortamını) otomatik olarak başlatır.
import pycuda.driver as cuda
from PIL import Image
import supervision as sv

# TensorRT motorunun çalışma sırasındaki uyarılarını ve loglarını kontrol eder. 
# WARNING seviyesi, sadece önemli hataları veya uyarıları göstererek terminal kalabalığını önler.
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# ==========================================
# Model Sabitleri ve Normalizasyon Parametreleri
# ==========================================
# TensorRT motorları genellikle sabit giriş boyutlarıyla (statik shape) derlenir. 
# Bu boyutlar, RAM tahsisinin önceden yapılabilmesi ve maksimum hız sağlanması için sabittir.
INPUT_H = 384
INPUT_W = 384

# ImageNet veri setinin istatistiksel ortalama (MEAN) ve standart sapma (STD) değerleri.
# Model eğitilirken görüntüler bu değerlerle normalize edildiği için, 
# test (inference) sırasında da girdi görüntülerinin aynı matematiksel dağılıma çekilmesi şarttır.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
class RFDETRTensorRT:
    """
    RF-DETR modelini TensorRT engine ile çalıştıran inference sınıfı.
    """

    def __init__(self, engine_path: str):
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"Engine bulunamadı: {engine_path}")

        # Daha önceden derlenmiş (compile edilmiş) model yüklenir.
        self.engine   = self._load_engine(engine_path)
        # Modelin GPU üzerinde çalışabilmesi için bir yürütme bağlamı (execution context) oluşturulur.
        self.context  = self.engine.create_execution_context()
        # Asenkron (eşzamanlı olmayan) GPU işlemleri için bir 'Stream' başlatılır. 
        # Bu, CPU ve GPU'nun birbirini beklemeden paralel çalışmasına olanak tanır.
        self.stream   = cuda.Stream()
        # CPU ve GPU bellekleri arasında veri aktarımı için gerekli alanlar ayrılır.
        self._allocate_buffers()
        print(f"✅ TensorRT engine yüklendi: {engine_path}")

    # ── Engine Yükleme ────────────────────────────────────────────────────
    @staticmethod
    def _load_engine(path: str) -> trt.ICudaEngine:
        # Disk üzerindeki serileştirilmiş (baytlara çevrilmiş) motor dosyasını okur ve 
        # GPU'nun anlayacağı ICudaEngine nesnesine geri çevirir (deserialize).
        runtime = trt.Runtime(TRT_LOGGER)
        with open(path, "rb") as f:
            return runtime.deserialize_cuda_engine(f.read())

    # ── Bellek Tahsisi (Memory Allocation) ─────────────────────────────────
    def _allocate_buffers(self):
        """Giriş/çıkış tensor'ları için GPU ve CPU belleğini önden tahsis eder."""
        self.inputs  = []
        self.outputs = []
        self.bindings = [] # TensorRT'nin bellek adreslerini bulması için tutulan liste.

        # Modelin kaç adet girdi ve çıktı tensörü (veri bloğu) olduğu döngüye alınır.
        for i in range(self.engine.num_io_tensors):
            name  = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = self.engine.get_tensor_shape(name)
            size  = trt.volume(shape) # Tensörün toplam eleman sayısı (Örn: 1 * 3 * 384 * 384)

            # Pagelocked (Pinned) Host Memory: Standart RAM (CPU belleği) işletim sistemi tarafından 
            # diske (pagefile) kaydırılabilir. Pinned memory, belleğin fiziksel RAM'de sabit kalmasını sağlar.
            # Bu işlem, CPU'dan GPU'ya (DMA - Direct Memory Access) veri transfer hızını muazzam ölçüde artırır.
            host_mem   = cuda.pagelocked_empty(size, dtype)
            
            # GPU (VRAM) üzerinde veri için yer ayrılır.
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

    # ── Görüntü Ön İşleme (Preprocessing) ────────────────────────────────
    @staticmethod
    def preprocess(image: np.ndarray) -> np.ndarray:
        """
        Kamera/Video'dan gelen numpy dizisini modelin beklediği matematiksel tensöre dönüştürür.
        """
        # OpenCV varsayılan BGR renk uzayını RGB'ye çevirir (Model RGB ile eğitilmiştir).
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Görüntü, TensorRT motorunun beklediği sabit boyuta (384x384) yeniden boyutlandırılır.
        img = cv2.resize(img, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
        
        # Piksel değerleri [0, 255] aralığından [0.0, 1.0] aralığına normalize edilir.
        img = img.astype(np.float32) / 255.0
        
        # Z-Skoru normalizasyonu (Görüntü - Ortalama / Standart Sapma).
        img = (img - MEAN) / STD                  
        
        # OpenCV formatı HWC (Yükseklik, Genişlik, Kanal) şeklindedir.
        # PyTorch/TensorRT modelleri ise CHW (Kanal, Yükseklik, Genişlik) formatı bekler.
        img = img.transpose(2, 0, 1)              
        
        # Model, aynı anda birden fazla görüntü (batch) alabilecek şekilde tasarlandığı için
        # dizinin başına bir boyut daha eklenir (1, C, H, W).
        img = np.ascontiguousarray(img[np.newaxis])  
        return img

    # ── TensorRT Çalıştırma (Inference Execution) ─────────────────────────
    def _infer(self, input_tensor: np.ndarray) -> list:
        """CPU ve GPU arasındaki asıl veri trafiğini ve yapay zeka işlemini yönetir."""
        # 1. İşlenmiş görüntü verisini CPU'daki sabit belleğe (host) kopyala.
        np.copyto(self.inputs[0]["host"], input_tensor.ravel())

        # 2. CPU'daki veriyi asenkron olarak (işlemciyi dondurmadan) GPU'ya gönder (Host To Device).
        cuda.memcpy_htod_async(
            self.inputs[0]["device"],
            self.inputs[0]["host"],
            self.stream
        )

        # 3. GPU'ya "İşleyeceğin veri bu bellek adreslerinde" bilgisini ver.
        for t in self.inputs + self.outputs:
            self.context.set_tensor_address(t["name"], int(t["device"]))

        # 4. Asıl Tahmin: Modeli GPU üzerinde çalıştır.
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # 5. Modelin ürettiği sonuçları (tahminleri) GPU'dan tekrar CPU belleğine al (Device To Host).
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)

        # CPU, GPU'nun işlemlerini bitirmesini bekler (Senkronizasyon noktası).
        self.stream.synchronize()
        
        # Veriyi düz liste halinden tekrar modelin çıkış boyutlarına (shape) getirerek döndür.
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
        Modelin ürettiği ham sayıları (logits ve oranları), gerçek piksel koordinatlarına dönüştürür.
        """
        # Modelden dönen ham koordinatlar ve sınıf skorları.
        dets_raw   = raw_outputs[0][0]   # Kutu koordinatları [Merkez X, Merkez Y, Genişlik, Yükseklik]
        labels_raw = raw_outputs[1][0]   # Ham sınıf tahmin skorları (Logits)

        # Ham skorları (Logits), Sigmoid fonksiyonu kullanılarak [0, 1] arası olasılık değerlerine çeviririz.
        scores_per_class = 1 / (1 + np.exp(-labels_raw))   
        
        # En yüksek olasılığa sahip sınıfın ID'sini bulur.
        class_ids = scores_per_class.argmax(axis=-1)        
        scores    = scores_per_class[np.arange(len(class_ids)), class_ids]

        # Güven skoru bizim belirlediğimiz eşikten (örn %50) büyük olanları ayır (Thresholding).
        keep = scores >= threshold
        if keep.sum() == 0:
            return sv.Detections.empty()

        # Filtrelenmiş tahminler alınır.
        boxes_cxcywh = dets_raw[keep]    
        class_ids    = class_ids[keep]
        scores       = scores[keep]

        # Modelin çıktıları [0, 1] arasında normalize edilmiştir. 
        # Gerçek piksel boyutlarını bulmak için orijinal resmin Genişlik (orig_w) ve Yüksekliği (orig_h) ile çarpılır.
        cx = boxes_cxcywh[:, 0] * orig_w
        cy = boxes_cxcywh[:, 1] * orig_h
        w  = boxes_cxcywh[:, 2] * orig_w
        h  = boxes_cxcywh[:, 3] * orig_h

        # Merkez X, Y ve Genişlik, Yükseklik değerleri kullanılarak 
        # kutunun Sol-Üst (x1, y1) ve Sağ-Alt (x2, y2) piksel koordinatları hesaplanır.
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        xyxy = np.stack([x1, y1, x2, y2], axis=1)
        # Tahmin edilen kutunun görüntünün sınırları dışına taşmasını engeller.
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
        Ön işleme, çıkarım ve son işlemenin tek bir fonksiyonda birleştiği ana çağrı.
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
        ms = (time.perf_counter() - t0) * 1000 # İşlem süresi milisaniyeye çevrilir.

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

    # ── Bağlam Yöneticisi (Context Manager - 'with' Kullanımı) ────────────
    # Bu sınıftan oluşturulan nesne 'with' bloğu içinden çıktığında,
    # '__exit__' otomatik tetiklenerek bellek sızıntısını engellemek için GPU kaynaklarını serbest bırakır.
    def __enter__(self):
        return self

    def __exit__(self, *_):
        del self.context
        del self.engine


# ─────────────────────────────────────────────────────────────────────────────
# Hız Testi (Benchmarking)
# ─────────────────────────────────────────────────────────────────────────────
def benchmark(engine_path: str, n_runs: int = 200, warmup: int = 20):
    """
    TensorRT motorunun kararlı çalışma hızını test eder.
    Isınma (warmup) çok önemlidir; GPU ilk çalıştığında bellek transferleri ve 
    saat hızları (clock speeds) optimum seviyede olmaz.
    """
    from rfdetr.assets.coco_classes import COCO_CLASSES  # İsteğe bağlı

    detector = RFDETRTensorRT(engine_path)
    # Rastgele gürültüden oluşan sahte bir görüntü (dummy) üretilir.
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
    print(f"Ortalama   : {times.mean():.2f} ms") # Ortalama işlem süresi
    print(f"Medyan     : {np.median(times):.2f} ms") # Aykırı değerlerden arındırılmış orta nokta süresi
    print(f"P99        : {np.percentile(times, 99):.2f} ms") # İşlemlerin %99'unun altında kaldığı maksimum gecikme
    print(f"Min / Max  : {times.min():.2f} / {times.max():.2f} ms")
    print(f"FPS        : {1000/times.mean():.1f}")
    print(f"{'─'*40}")


# ─────────────────────────────────────────────────────────────────────────────
# Ana Program Akışı
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
    # 'with' kullanımı, işlem bitince GPU hafızasının temizlenmesini garanti eder.
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
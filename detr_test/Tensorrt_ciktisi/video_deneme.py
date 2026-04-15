"""
RF-DETR TensorRT - Gerçek Zamanlı Webcam / Kamera Akışı
Adım 4 (isteğe bağlı): Kamera beslemesinde canlı nesne tespiti.

Bu betik, önceki adımda donanıma özel derlenmiş olan TensorRT motorunu (.engine) 
kullanarak sürekli bir video akışı (stream) üzerinden gerçek zamanlı çıkarım (inference) yapar.

Kullanım:
    python step4_realtime.py                 # webcam (0)
    python step4_realtime.py --source rtsp://... # IP kamera
    python step4_realtime.py --source video.mp4  # video dosyası

Çıkış: 'q' tuşuna basarak pencereyi kapat.
"""

# Komut satırı argümanlarını (parametrelerini) ayrıştırmak için kullanılan kütüphane.
import argparse
import time
import cv2
import numpy as np

# Önceki modülde oluşturduğumuz ve TensorRT bellek yönetimini halleden sınıfı içe aktarıyoruz.
from modeli_dene import RFDETRTensorRT  

# Modelin tahmin ettiği ID numaralarını (0, 1, 2...) anlamlı etiketlere ("insan", "araba")
# dönüştürmek için kullanılan sınıf listesi.
try:
    COCO_CLASSES = {
        1: "drone",
        2: "f16",
        3: "helicopter",
        4: "rocket",
        5: "missile"}
    CLASS_NAMES = list(COCO_CLASSES.values())
except ImportError:
    CLASS_NAMES = None


# ==========================================
# Ana Çalıştırma Fonksiyonu (Pipeline)
# ==========================================
def run(engine_path: str, source: int | str, threshold: float):
    """
    Kameradan veya videodan sürekli kare (frame) okuyan ve modeli besleyen döngü.
    """
    # 1. Modelin GPU'ya Yüklenmesi
    # TensorRT motoru belleğe (VRAM) yüklenir ve CPU-GPU arasındaki köprüler kurulur.
    detector = RFDETRTensorRT(engine_path)
    
    # 2. Veri Akışının (Stream) Başlatılması
    # OpenCV'nin VideoCapture sınıfı, donanım sensöründen (webcam), bir ağ protokolünden (RTSP) 
    # veya yerel bir dosyadan (MP4) gelen veri paketlerini yakalar ve çözümler (decode).
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Kaynak açılamadı: {source}")

    # FPS (Saniyedeki Kare Sayısı) değerini stabilize etmek için bir tampon (buffer) listesi.
    # Anlık hesaplanan FPS çok dalgalanır (jitter). Son 30 karenin ortalamasını almak 
    # ekrandaki yazının okunabilir olmasını sağlar.
    fps_buf = []
    print("Başlatılıyor... 'q' ile çık.\n")

    # 3. Sonsuz Okuma Döngüsü (Real-time Loop)
    while True:
        # Sensörden anlık veriyi çek. 'ret' (return) verinin bütünlüğünü doğrular.
        ret, frame = cap.read()
        if not ret:
            break

        # Kronometreyi başlat. Sadece yapay zeka çıkarım süresini ölçeceğiz.
        t0 = time.perf_counter()
        
        # Görüntüyü modele gönder ve sonuçları al.
        detections = detector.predict(frame, threshold)
        
        # Geçen süreyi (Delta Time) hesapla. 
        # 1e-6 (0.000001) kullanımı, çok hızlı sistemlerde 'sıfıra bölünme' (ZeroDivisionError)
        # hatasını engellemek için eklenmiş matematiksel bir güvenlik önlemidir.
        fps_buf.append(1.0 / max(time.perf_counter() - t0, 1e-6))
        
        # Tampon belleğin (liste) boyutunu son 30 kare ile sınırla (Kuyruk mantığı - FIFO).
        if len(fps_buf) > 30:
            fps_buf.pop(0)

        # 4. Görselleştirme (Annotation)
        # RFDETRTensorRT sınıfı içindeki (önceki adımda yazdığımız) çizim fonksiyonunu çağırır.
        annotated = detector._annotate(frame, detections, CLASS_NAMES)

        # FPS Göstergesinin Çizilmesi
        # np.mean(fps_buf) ile son 30 karenin FPS ortalaması hesaplanır.
        fps_text = f"FPS: {np.mean(fps_buf):.1f}"
        cv2.putText(
            annotated, fps_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
        )

        # 5. Ekran Çıktısı ve Döngü Kontrolü
        cv2.imshow("Algilama V3", annotated)
        
        # Klavye dinleyicisi. Her kare çizildikten sonra 1 milisaniye bekler.
        # Bu bekleme süresi, işletim sisteminin pencereyi güncellemesi (render) için zorunludur.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Döngü kırıldığında (kullanıcı 'q' tuşuna bastığında veya video bittiğinde) 
    # donanım kilitlerini serbest bırak (Memory Management).
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nOrtalama FPS: {np.mean(fps_buf):.1f}")


# ==========================================
# Programın Giriş Noktası (Entry Point)
# ==========================================
if __name__ == "__main__":
    # Kullanıcının terminalden (komut satırı) girebileceği parametrelerin tanımlanması.
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine",    default="exports/rfdetr.engine")
    parser.add_argument("--source",    default=0, help="webcam id veya video/rtsp yolu")
    parser.add_argument("--threshold", type=float, default=0.5)
    
    # Kullanıcının terminale yazdığı komutları parse (ayrıştır) edip bir objeye dönüştürür.
    args = parser.parse_args()

    # Kaynak Tipi Tespiti:
    # OpenCV, yerel USB kameralara donanım ID'si ile erişir (örneğin 0, 1, 2 = Integer).
    # Ancak IP kameralara (RTSP) veya videolara dosya yolu ile erişir ("video.mp4" = String).
    # Terminalden gelen girdiler her zaman String (metin) formatındadır. 
    # Bu try-except bloğu, eğer girdi "0" gibi bir rakamsa onu Integer'a çevirerek OpenCV'nin 
    # USB kamerayı doğru tanımasını sağlar. Metinse (örn: "rtsp://...") olduğu gibi bırakır.
    try:
        source = int(args.source)
    except (ValueError, TypeError):
        source = args.source

    # İşlemi başlat.
    run(args.engine, source, args.threshold)
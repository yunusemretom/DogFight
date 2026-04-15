import cv2
import time
import supervision as sv
from rfdetr import RFDETRLarge,RFDETRNano,RFDETRBase,RFDETRSmall
from rfdetr.assets.coco_classes import COCO_CLASSES
#
# 1. Modelin Başlatılması
# RF-DETR mimarisi, transformer tabanlı bir nesne tespit modelidir.
model = RFDETRSmall(pretrain_weights="/home/tom/Downloads/checkpoint_best_regular_colab_4.pth")


# Görselleştirme (OpenCV/Supervision) için numpy dizisi gereklidir (BGR)
cap = cv2.VideoCapture("/home/tom/Downloads/snapsave-app_3868097202723797224.mp4")

if not cap.isOpened():
    raise RuntimeError("Video acilamadi. Dosya yolunu kontrol edin.")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
source_fps = cap.get(cv2.CAP_PROP_FPS)
output_fps = source_fps if source_fps and source_fps > 0 else 30.0
output_path = "/home/tom/Documents/DogFight/cikti.mp4"

# MP4 icin yaygin codec: mp4v
video_writer = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    output_fps,
    (frame_width, frame_height),
)

if not video_writer.isOpened():
    raise RuntimeError("Cikti videosu olusturulamadi. Yazma izinlerini kontrol edin.")

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# FPS hesaplama icin zaman tutucu
prev_time = time.perf_counter()
fps = 0.0
fps_smoothing = 0.9

while True:
    ret, image_cv2 = cap.read()
    if not ret:
        print("Video bitti veya kare okunamadi.")
        break

    now = time.perf_counter()
    dt = now - prev_time
    if dt > 0:
        current_fps = 1.0 / dt
        fps = (fps_smoothing * fps) + ((1.0 - fps_smoothing) * current_fps) if fps > 0 else current_fps
    prev_time = now

    # 3. İnferans (Tahmin) İşlemi
    # Threshold 0.5: Modelin %50'den az emin olduğu tahminler filtrelenir.
    detections = model.predict(image_cv2, threshold=0.5)

    # 4. Etiketlerin Hazırlanması (Düzeltilmiş Kısım)
    # detections.class_id ve detections.confidence listelerini 'zip' ile eşleştiriyoruz.
    # Bu sayede her tespite tam olarak 1 etiket karşılık gelir (N=N eşleşmesi).
    labels = [
        f"{COCO_CLASSES[class_id]} {confidence:.2f}" 
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]
    print(f"Algilanan nesneler: {labels}")
    # 6. Görüntü Üzerine Çizim Yapılması
    # .copy() kullanımı orijinal görselin bozulmasını engeller.
    annotated_image = box_annotator.annotate(
        scene=image_cv2.copy(), 
        detections=detections
    )

    annotated_image = label_annotator.annotate(
        scene=annotated_image, 
        detections=detections, 
        labels=labels
    )

    cv2.putText(
        annotated_image,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    # 7. Sonucu Ekranda Gösterme
    # sv.plot_image() fonksiyonu görüntüyü uygun bir pencerede (matplotlib tabanlı) açar.
    print(f"Toplam {len(detections)} nesne algılandı.")

    video_writer.write(annotated_image)
    
    cv2.imshow("RF-DETR Tespit Sonucu", annotated_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # Opsiyonel: Sonucu kaydetmek isterseniz alttaki satırı aktif edebilirsiniz.
    # cv2.imwrite("tespit_sonucu.jpg", annotated_image)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
print(f"Video kaydedildi: {output_path}")
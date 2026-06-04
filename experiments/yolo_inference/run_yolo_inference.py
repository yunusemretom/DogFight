import cv2
from ultralytics import YOLO
import time
import torch
import numpy as np
from collections import deque
import threading

# GPU kullanımını etkinleştir ve model ayarları
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO("model.pt").to(device)
if device == 'cuda':
    torch.backends.cudnn.benchmark = True

# Video yakalama ve yazma ayarları
cap = cv2.VideoCapture("denemevideo2.mp4")
cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

# Video boyutlarını al ve yeniden boyutlandırma faktörünü ayarla
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
scale_factor = 0.4
new_width = int(frame_width * scale_factor)
new_height = int(frame_height * scale_factor)

# Video yazıcı ayarları
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, 30.0, (frame_width, frame_height))

# Performans parametreleri
score_threshold = 0.5
fps_list = deque(maxlen=50)
prev_time = time.time()
start_time = time.time()
frame_count = 0

# Frame ve sonuç buffer'ları
frame_buffer = deque(maxlen=2)
results_buffer = deque(maxlen=2)
processing_lock = threading.Lock()
running = True

def process_frame(frame):
    # Frame'i yeniden boyutlandır
    small_frame = cv2.resize(frame, (new_width, new_height))
    
    # BGR'den RGB'ye dönüştür
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    # YOLO tespiti
    results = model(rgb_frame, stream=True)
    return next(results)

def draw_fps_info(frame, current_fps, fps_list):
    if not fps_list:
        return frame
        
    avg_fps = sum(fps_list) / len(fps_list)
    min_fps = min(fps_list)
    max_fps = max(fps_list)
    
    # Arka plan dikdörtgeni çiz
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (250, 120), (0, 0, 0), -1)
    alpha = 0.6
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    # FPS bilgilerini yaz
    cv2.putText(frame, f"Current FPS: {int(current_fps)}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Avg FPS: {int(avg_fps)}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Min FPS: {int(min_fps)}", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Max FPS: {int(max_fps)}", (10, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return frame

def process_detection_thread():
    while running:
        with processing_lock:
            if len(frame_buffer) > 0:
                frame = frame_buffer.popleft()
                try:
                    result = process_frame(frame)
                    results_buffer.append((frame, result))
                except Exception as e:
                    print(f"Detection error: {e}")
            else:
                time.sleep(0.001)

# Detection thread'ini başlat
detection_thread = threading.Thread(target=process_detection_thread, daemon=True)
detection_thread.start()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        current_time = time.time()
        
        # Frame'i buffer'a ekle
        with processing_lock:
            frame_buffer.append(frame.copy())
        
        # İşlenmiş sonuçları al
        if results_buffer:
            _, result = results_buffer.popleft()
            
            # Tespitleri çiz
            for box in result.boxes:
                if box.conf > score_threshold:
                    # Koordinatları orijinal boyuta ölçeklendir
                    x1, y1, x2, y2 = map(int, box.xyxy[0] / scale_factor)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = f"Class: {cls}, Score: {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # FPS hesaplama
        elapsed_time = current_time - prev_time
        current_fps = 1 / elapsed_time if elapsed_time > 0 else 0
        fps_list.append(current_fps)
        
        # FPS bilgilerini ekrana çiz
        frame = draw_fps_info(frame, current_fps, fps_list)
        
        prev_time = current_time
        
        cv2.imshow('YOLOv8 Live Detection', frame)
        out.write(frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Thread'i durdur
    running = False
    detection_thread.join(timeout=1.0)
    
    # Son performans istatistiklerini göster
    total_time = time.time() - start_time
    print(f"\nPerformans İstatistikleri:")
    print(f"Toplam süre: {total_time:.2f} saniye")
    print(f"Toplam frame: {frame_count}")
    print(f"Ortalama FPS: {frame_count/total_time:.2f}")
    
    # Temizlik
    cap.release()
    out.release()
    cv2.destroyAllWindows() 
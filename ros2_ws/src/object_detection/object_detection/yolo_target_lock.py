import cv2
import time
from collections import deque
from ultralytics import YOLO

# ============================= MODEL ================================
model = YOLO("/home/tom/Downloads/best(1).pt")  # sen kendi eğittiğin modeli buraya koyabilirsin örn: best.pt

# ============================= AYARLAR ==============================
FRAME_RATE = 30                 # Kamera FPS
REQUIRED_LOCK = 4               # Minimum kilit süresi
TOLERANCE = 1                   # ±1sn tolerans
WINDOW = 2                      # İncelenecek zaman aralığı
MIN_AREA = 0.05                 # Ekranın %5'ini kapsamalı (min şart)
frames = deque(maxlen=FRAME_RATE * WINDOW)

# ===================================================================
def draw_zones(frame):
    h, w, _ = frame.shape

    # Sarı dış bölge (Vuruş Alanı)
    x1 = int(w * 0.25);  y1 = int(h * 0.10)
    x2 = int(w * 0.75);  y2 = int(h * 0.90)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,255), 2)

    # İç kırmızı kilit kutusu (AH)
    mw = int((x2 - x1) * 0.05)
    mh = int((y2 - y1) * 0.05)
    AH = (x1+mw, y1+mh, x2-mw, y2-mh)
    cv2.rectangle(frame, (AH[0], AH[1]), (AH[2], AH[3]), (0,0,255), 2)

    return AH

# ===================================================================
def check_lock():
    now = time.time()
    last = [f for f in frames if now - f["time"] <= WINDOW]
    locked_time = sum([f["locked"] for f in last]) / FRAME_RATE

    if REQUIRED_LOCK - TOLERANCE <= locked_time <= REQUIRED_LOCK + TOLERANCE:
        return True, locked_time
    return False, locked_time

# ===================================================================
cap = cv2.VideoCapture(0)    # Kamerayı aç

while cap.isOpened():

    ret, frame = cap.read()
    if not ret: break

    AH = draw_zones(frame)                     # Şekil 2’ye göre alanları çiz
    h, w, _ = frame.shape

    # ==================== YOLO İLE TESPİT ============================
    results = model.predict(frame, conf=0.50, verbose=False)

    detected = False
    for r in results:
        for box in r.boxes:

            x1,y1,x2,y2 = box.xyxy[0].cpu().numpy().astype(int)
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            # YOLO kutusunu çiz
            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
            cv2.putText(frame,f"{cls} {conf:.2f}",(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)

            area = ((x2-x1)*(y2-y1)) / (w*h)
            inside = (x1>AH[0] and y1>AH[1] and x2<AH[2] and y2<AH[3])

            locked = area >= MIN_AREA and inside
            detected = True

            frames.append({"time": time.time(), "locked":1 if locked else 0})

    # Eğer hiçbir bbox bulunmazsa frame ekle (kilit = 0)
    if not detected:
        frames.append({"time": time.time(), "locked":0})

    ok, lt = check_lock()

    cv2.putText(frame,f"LOCK: {'YES' if ok else 'NO'}",(25,40),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0) if ok else (0,0,255),3)

    cv2.putText(frame,f"LOCKED_TIME: {lt:.2f}sn",(25,85),
                cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2)

    if ok:
        cv2.putText(frame,"🚀 TARGET HIT 🚀",(60,150),
                    cv2.FONT_HERSHEY_SIMPLEX,1.4,(0,255,0),4)

    cv2.imshow("YOLO Lock System",frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()

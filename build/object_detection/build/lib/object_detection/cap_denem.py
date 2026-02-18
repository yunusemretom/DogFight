import cv2

cap = cv2.VideoCapture(3, cv2.CAP_V4L2)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)

alpha = 1.3   # Kontrast (1.0 = değişmez)
beta = 30     # Parlaklık (0 = değişmez)

if not cap.isOpened():
    print("Kamera açılamadı")
    exit()

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, -1)
    frame = cv2.flip(frame, 1)

    frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    if not ret:
        print("Frame yok")
        break

    cv2.imshow("DC880", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

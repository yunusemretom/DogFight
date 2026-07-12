# DogFight — Dosya Yapısı

> Nerede ne var, hangi dosya ne iş yapar. Kardeş dosyalar: `PROJE_OZETI.md`, `GOREVLER.md`.
> Son güncelleme: 2026-07-12

```
DogFight/
├── PROJE_OZETI.md / DOSYA_YAPISI.md / GOREVLER.md   # AI/geliştirici hızlı bağlam dosyaları
├── README.md                        # Genel tanıtım (Türkçe, mimari diyagramlı)
├── LICENSE                          # BSD 3-Clause
│
├── ros2_ws/                         # ★ ROS 2 Humble çalışma alanı (asıl kod burada)
│   └── src/
│       ├── px4_msgs/                # PX4 v1.16 mesaj tanımları (git submodule — firmware ile uyumlu tutulmalı)
│       ├── px4_ros_com/             # PX4-ROS2 köprü örnekleri (git submodule)
│       │
│       ├── dogfight_tracking/       # ★ HEDEF TAKİP PAKETİ (ana iş)
│       │   └── dogfight_tracking/
│       │       ├── l1_pursuit_node.py      # ★ Mesafe korumalı L1/formasyon takip kontrolcüsü
│       │       │                           #   (SITL doğrulamalı; rehberi: docs/guides/l1-takip-rehberi.md)
│       │       ├── gps_tracker_node.py     # İki aracın GPS'ini izleyip CSV loglayan yardımcı
│       │       └── visual_tracker_node.py  # Görsel takip (kamera tabanlı) — entegrasyonu tamamlanmadı
│       │
│       ├── dogfight_control/        # Offboard arayüz TEST node'ları (2026-07-12 temiz yazım; üretim takibi l1_pursuit_node'da)
│       │   ├── archive/
│       │   │   └── fw_pursuit_node.py          # ESKİ pure-pursuit — arşiv, entry point'i yok
│       │   ├── test/                           # pytest birim + smoke testleri (SITL gerektirmez)
│       │   │   ├── test_control_math.py
│       │   │   └── test_nodes_smoke.py
│       │   └── dogfight_control/
│       │       ├── control_math.py             # Saf matematik (PID, follow_point, NED dönüşümü) — ROS'suz test edilir
│       │       ├── offboard_base.py            # Ortak taban: durum makinesi, v1.16 topic varyantları, arm/offboard
│       │       ├── attitude_setpoint_node.py   # VehicleAttitudeSetpoint testi (SITL doğrulamalı)
│       │       ├── trajectory_velocity_node.py # TrajectorySetpoint.velocity testi (FW'de mode manager devrede — gözlem amaçlı)
│       │       ├── trajectory_position_node.py # TrajectorySetpoint.position testi (aynı sınırlama)
│       │       ├── visual_offboard_node.py     # Görsel tespit → FW lateral/longitudinal köprüsü
│       │       └── px4_status_monitor.py       # Durum monitörü (eski deneme.py'nin düzenlenmişi)
│       │
│       ├── dogfight_detection/      # Nesne tespiti paketi
│       │   └── dogfight_detection/
│       │       ├── rfdetr_detection_node.py    # RF-DETR ROS2 düğümü
│       │       └── yolo_detection_node.py      # YOLO ROS2 düğümü
│       │
│       └── dogfight_bringup/        # Launch + parametre dosyaları
│           ├── launch/              # detection / tracking / full_system launch'ları
│           └── config/              # control/detection/tracking_params.yaml
│
├── simulation/
│   └── scripts/
│       ├── launch_multi_aircraft.sh # ★ Çoklu araç PX4 SITL + Gazebo başlatıcı
│       └── install_dependencies.sh  # Bağımlılık kurulumu
│
├── docs/                            # Jekyll dokümantasyon sitesi (GitHub Pages)
│   ├── guides/                      # Rehberler (otomatik listelenir; front-matter: title/icon/order)
│   │   ├── kurulum-rehberi.md       #   order 1
│   │   ├── kullanim-rehberi.md      #   order 2
│   │   ├── rfdetr-rehberi.md        #   order 3
│   │   ├── yolo-rehberi.md          #   order 4
│   │   ├── l1-takip-rehberi.md      #   order 5 — ★ L1 algoritması + güvenlik önlemleri
│   │   ├── dogfight-control-rehberi.md   #   order 6 — offboard test node'ları mimarisi + SITL sonuçları
│   │   └── px4_ros2_fixedwing_rehber.md  # PX4+ROS2 fixed-wing genel rehberi (front-matter'sız)
│   ├── experiments/ · tools/        # Deney ve araç dokümanları
│   └── _layouts/ · _includes/ · assets/  # Jekyll tema
│
├── experiments/                     # Model deneyleri (ROS dışı, bağımsız scriptler)
│   ├── rfdetr_inference/            # RF-DETR: video/ekran testi, ONNX + TensorRT ihracı ve koşumu
│   ├── rfdetr_tctrack/              # RF-DETR tespit + TCTrack takip birleşik pipeline
│   ├── yolo_inference/              # YOLO çıkarım denemeleri
│   ├── model_benchmark/             # compare_models.py — model karşılaştırma
│   └── training/                    # rfdetr_training.ipynb — eğitim not defteri
│
├── TCTrack/                         # TCTrack görsel takip repo'su (git submodule)
├── object_detection/                # İhraç edilmiş çıkarım modelleri (.onnx, .engine)
├── tools/                           # convert_video_format.py, gps_bearing_calculator.py
│
├── rf-detr-base.pth / rf-detr-large-2026.pth   # Model ağırlıkları (büyük; git'e girmemeli)
├── video/ · Video_2/                # Test videoları / kayıtlar
└── gps_log_*.csv                    # gps_tracker_node çıktı logları
```

## Önemli dış bağımlılık

- **`~/PX4-Autopilot`** (repo dışında): PX4 v1.16 kaynak + SITL build. `build/px4_sitl_default/bin/px4-param` gibi araçlarla çalışan SITL instance'larına komut verilebilir (instance 0 = takipçi, socket `/tmp/px4-sock-0`).

## Nereye ne eklenir?

| Yapılacak iş | Yer |
|---|---|
| Yeni takip/güdüm algoritması | `ros2_ws/src/dogfight_tracking/` + `setup.py` entry_points |
| Yeni tespit modeli düğümü | `ros2_ws/src/dogfight_detection/` |
| Launch/param değişikliği | `ros2_ws/src/dogfight_bringup/` |
| ROS dışı model deneyi | `experiments/<konu>/` |
| Dokümantasyon | `docs/guides/*.md` (front-matter ile) |

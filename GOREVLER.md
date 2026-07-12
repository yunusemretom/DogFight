# DogFight — Görevler

> Projenin durum panosu: ne bitti, ne sürüyor, sırada ne var.
> Yeni bir oturuma başlayan yapay zeka: önce `PROJE_OZETI.md`'yi oku, sonra buradaki "Devam eden / Sıradaki" bölümlerine bak.
> Son güncelleme: 2026-07-12 — durumlar değiştikçe bu dosyayı güncelle.

## ✅ Tamamlanan

- [x] Çoklu araç PX4 SITL + Gazebo ortamı (`launch_multi_aircraft.sh`; takipçi `/fmu`, hedef `/px4_1`)
- [x] PX4 v1.16 fixed-wing offboard arayüzünün çözülmesi (lateral/longitudinal setpoint reçetesi — bkz. `docs/guides/l1-takip-rehberi.md` §1)
- [x] **L1 takip düğümü** (`l1_pursuit_node.py`): uzakta L1 (yay-üzerinde takip noktası), yakında formasyon modu (course + cross-track + PID mesafe)
  - [x] SITL doğrulaması: 50 m hedef → ~57 m (std 0.9 m); 5 m hedef → 5.7 m (std 0.28 m), dönen hedefe karşı
  - [x] Güvenlik katları: min ayrılma (3 m), hedef timeout→HOLD, offboard kaybında yeniden bağlanma, irtifa kilidi
  - [x] Çalışma anında ayarlanabilir ROS parametreleri
- [x] SITL failsafe düzeni (takipçi: `NAV_DLL_ACT=0`, `COM_RCL_EXCEPT=4`)
- [x] RF-DETR / YOLO tespit düğümleri + TensorRT/ONNX ihraç scriptleri (`experiments/rfdetr_inference/`)
- [x] **`dogfight_control` temiz yeniden yazımı** (2026-07-12): ortak taban `offboard_base.py` (durum makinesi, v1.16 topic varyantları, çerçeve-güvenli bağıl konum) + ROS'suz test edilebilir `control_math.py`
  - [x] `attitude_setpoint_node` — SITL doğrulamalı: 20 m hedefte ort 20.5 m (std 4.2 m), irtifa |hata| 0.48 m
  - [x] `trajectory_velocity_node` / `trajectory_position_node` — tuzak #6 ÖLÇÜLDÜ: velocity'de mesafe komuttan bağımsız ~4 m'ye kilitlendi, position'da 30 m hedefe karşı ~75 m'ye açıldı (FW mode manager kendi guidance'ını işletiyor; FW'de bu arayüzlerle hassas takip yok)
  - [x] `visual_offboard_node` — FW lateral/longitudinal reçetesine taşındı; sahte tespitle doğrulandı (+100 px → +5 m/s² yanal, timeout → HOLD düz uçuş)
  - [x] 32 birim test + 5 smoke test (`dogfight_control/test/`, pytest ile SITL'siz koşuyor)
  - [x] Eski bozuk node'lar silindi (`attitude/velocity/position_controller_node.py`), `fw_pursuit_node.py` → `archive/`, `deneme.py` → `px4_status_monitor` entry point'i
  - [x] Rehber yazıldı: `docs/guides/dogfight-control-rehberi.md` (mimari, kontrol yasaları, SITL sonuçları, ortam tuzakları)
- [x] Jekyll dokümantasyon sitesi + 6 rehber (`docs/guides/`)
- [x] AI bağlam dosyaları (`PROJE_OZETI.md`, `DOSYA_YAPISI.md`, `GOREVLER.md`)

## 🔄 Devam eden

- [ ] L1 takibinin farklı senaryolarda testi (hedef düz uçuş, ani manevra, irtifa değişimi — şimdiye dek ağırlıkla loiter'daki hedefle test edildi)
- [ ] Kalan ~+0.7 m mesafe sapmasının giderilmesi (`ki_distance` ayarı ile)
- [ ] Görsel takip zinciri: RF-DETR/TCTrack çıktısının takip düğümüne bağlanması (`visual_tracker_node.py` entegrasyonu eksik)

## 📋 Sıradaki

- [ ] **GPS + görsel füzyon:** Yakın mesafede (< 15 m) GPS bağıl hatası yetersiz — kamera tabanlı bağıl konum ile füzyon gerekli
- [ ] `dogfight_bringup` launch dosyalarına `l1_pursuit_node`'un eklenmesi (tracking_launch.py güncel değil)
- [ ] Gerçek uçuş hazırlığı:
  - [ ] `auto_arm_takeoff=false`, `auto_offboard=false` profili + pilot devralma prosedürü
  - [ ] Failsafe parametrelerinin gerçek araç için yeniden planlanması (SITL'de kapatılanlar geri alınacak)
  - [ ] Telemetri linki üzerinden hedef konum aktarımı ve gecikme ölçümü (`target_timeout` ayarı)
  - [ ] Hız/ivme limitlerinin gerçek uçağın zarfına göre daraltılması
- [ ] Hedef manevra kestirimi (dönüş hızı tahmini var; ivme kestirimi ile takip noktası öngörüsü geliştirilebilir)
- [ ] Depo temizliği: kök dizindeki model ağırlıkları (`rf-detr-*.pth`) ve video klasörlerinin git-lfs/dışarı taşınması (deneme.py/files.zip/fw_pursuit temizliği 2026-07-12'de yapıldı)
- [ ] `VEHICLE_CMD_NAV_TAKEOFF`'un ROS'tan işlememe sorunu (aşağıda) — l1_pursuit_node'un auto_arm_takeoff'u da aynı komutu kullanıyor, doğrulanmalı
- [ ] `gps_tracker_node.py`'deki namespace düzeninin (`/px4_1`+`/px4_3`) güncel sim düzeniyle (`/fmu`+`/px4_1`) uyumlanması

## ⚠️ Bilinen sorunlar

| Sorun | Durum / geçici çözüm |
|---|---|
| `colcon build --symlink-install` setuptools 83 ile kırık | Symlink'siz derle; gerekirse `rm -rf build/<pkg> install/<pkg>` |
| GCS'siz SITL'de datalink failsafe offboard'u düşürüyor | Takipçide `NAV_DLL_ACT=0` yapıldı (yalnızca SITL) |
| 5 m standoff gerçek uçuşta güvensiz (GPS bağıl hatası 2–5 m) | Görsel/RTK olmadan ≥15 m kullan |
| Kök dizinde büyük ikili dosyalar (`.pth`, videolar) | Git deposunu şişiriyor; taşınacak (bkz. Sıradaki) |
| ROS'tan `VEHICLE_CMD_NAV_TAKEOFF` SITL'de kalkış tetiklemiyor (2026-07-12'de gözlendi; ARM ve DO_SET_MODE işliyor) | `px4-commander --instance N takeoff` ile kalkış ver; node offboard'u 15 m'de devralıyor |
| `launch_multi_aircraft.sh` instance 2+3 spawn ediyor, dokümandaki düzen instance 0 (`/fmu`) + 1 (`/px4_1`) | Script güncellenmeli; testlerde instance 0+1 elle başlatıldı |
| rc_cessna SITL'de Hold/loiter modunda spiralle yere iniyor; pist dışına spawn (ör. 20,20) uçağı deviriyor | Hedef araç offboard FW lateral/longitudinal ile uçurulmalı (sabit yanal ivme = daire); pist üzeri spawn (5,-10 / -5,10) kullan |
| SITL failsafe paramları (`NAV_DLL_ACT` vb.) build/rootfs temizlenince sıfırlanıyor | Her yeni SITL kurulumunda iki instance'ta da yeniden set et |

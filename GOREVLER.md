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
- [ ] Depo temizliği: kök dizindeki model ağırlıkları (`rf-detr-*.pth`) ve video klasörlerinin git-lfs/dışarı taşınması; `dogfight_control/deneme.py`, `files.zip` silinmesi; eski `fw_pursuit_node.py`'nin arşivlenmesi
- [ ] `gps_tracker_node.py`'deki namespace düzeninin (`/px4_1`+`/px4_3`) güncel sim düzeniyle (`/fmu`+`/px4_1`) uyumlanması

## ⚠️ Bilinen sorunlar

| Sorun | Durum / geçici çözüm |
|---|---|
| `colcon build --symlink-install` setuptools 83 ile kırık | Symlink'siz derle; gerekirse `rm -rf build/<pkg> install/<pkg>` |
| GCS'siz SITL'de datalink failsafe offboard'u düşürüyor | Takipçide `NAV_DLL_ACT=0` yapıldı (yalnızca SITL) |
| 5 m standoff gerçek uçuşta güvensiz (GPS bağıl hatası 2–5 m) | Görsel/RTK olmadan ≥15 m kullan |
| Kök dizinde büyük ikili dosyalar (`.pth`, videolar) | Git deposunu şişiriyor; taşınacak (bkz. Sıradaki) |

# RF-DETR + TCTrack Entegrasyon Projesi

Bu klasor, DogFight icinde RF-DETR tespit modelini TCTrack ile frame-to-frame takipte birlestirir.

## Tasarim Ozeti

- Hedef tespiti: RF-DETR
- Hedef takibi: TCTrack (Temporal Consistent Tracking)
- Takip cikti bbox'u: AH (kilitlenme dortgeni)
- Kontrol sinyali: AH merkezinden uretilen normalize PID hata sinyali (`ex`, `ey`)

Metindeki gereksinime uygun olarak pipeline su sekilde calisir:

1. RF-DETR ile hedef ilk kez tespit edilir.
2. Tespit bbox'u ile TCTrack `init()` edilir.
3. Sonraki frame'lerde TCTrack `track()` ile hedef kimligini temporal olarak korur.
4. Tracker skoru duserse veya periyodik olarak RF-DETR yeniden devreye girer.

## Ekran Overlay Bilesenleri

- AK: Kamera gorus alani (dis cerceve)
- AV: Hedef vurus alani (soldan/sagdan %25, ustten/alttan %10 offset)
- AH: TCTrack cikti bbox'u (kilitlenme dortgeni)
- Merkez hedef karesi (goruntu merkezinde)
- AH oran yazilari:
  - AH yatay: `%` olarak frame genisligine oran (>= %5 kontrolu)
  - AH dikey: `%` olarak frame yuksekligine oran (>= %5 kontrolu)

## Kullanimi

Proje kokunden calistirin:

```bash
python rfdetr_tctrack_integration/run_rfdetr_tctrack.py \
  --source 0 \
  --rfdetr-variant large \
  --rfdetr-weights /home/tom/Documents/Projeler/DogFight/rf-detr-large-2026.pth \
  --tctrack-config /home/tom/Documents/Projeler/DogFight/TCTrack/experiments/TCTrack/config.yaml \
  --tctrack-snapshot /home/tom/Documents/Projeler/DogFight/TCTrack/snapshot/general_model.pth \
  --show \
  --output /home/tom/Documents/Projeler/DogFight/rfdetr_tctrack_integration/outputs/demo.mp4
```

Video dosyasi ile:

```bash
python rfdetr_tctrack_integration/run_rfdetr_tctrack.py \
  --source /path/to/video.mp4 \
  --show
```

## Notlar

- `--class-id` vererek yalnizca belirli bir class takip ettirilebilir.
- `--redetect-every` degeri kuculdukce RF-DETR daha sik devreye girer.
- `--track-score-threshold` degeri yukseldikce takipten detections'a geri donus daha erken olur.

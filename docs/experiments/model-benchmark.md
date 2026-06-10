---
layout: page
title: "Model Benchmark: YOLO vs RF-DETR"
description: "YOLO ve RF-DETR modellerinin performans karşılaştırması — FPS, doğruluk ve TensorRT optimizasyonu"
icon: "📊"
order: 1
---

## Genel Bakış

DogFight projesinde iki farklı nesne tespit modeli kullanılmaktadır. Bu sayfa, her iki modelin performans karşılaştırmasını içerir.

## Modeller

| Model | Mimari | Ağırlık Dosyası | Boyut |
|-------|--------|-----------------|-------|
| YOLO | Ultralytics YOLOv8/11 | `best.pt` | ~20 MB |
| RF-DETR | Transformer (DETR tabanlı) | `rf-detr-base.pth` | ~370 MB |
| RF-DETR Large | Transformer (büyük model) | `rf-detr-large.pth` | ~136 MB |

## Karşılaştırma Kriterleri

### FPS (Kare/Saniye)
- Yüksek FPS gerçek zamanlı takip için kritik
- Hedef: minimum 30 FPS (video akışı hızı)

### Doğruluk (mAP)
- Hedef tespiti doğruluğu
- Farklı mesafe ve açılarda performans

### TensorRT Optimizasyonu
- GPU üzerinde model hızlandırma
- FP16 ve INT8 precision modları

## Test Ortamı

- **GPU**: NVIDIA (CUDA 11.8+)
- **Giriş**: 640x480 çözünürlük video
- **Senaryo**: RC Cessna uçak görsel tespiti

## Deneyler Klasörü

| Klasör | İçerik |
|--------|--------|
| `experiments/model_benchmark/` | YOLO vs RF-DETR karşılaştırmalı video çıktısı |
| `experiments/rfdetr_tctrack/` | RF-DETR tespit + TCTrack temporal takip entegrasyonu |
| `experiments/yolo_inference/` | YOLO inferans, video test, TensorRT dönüştürme |
| `experiments/rfdetr_inference/` | RF-DETR inferans, ekran yakalama, SAHI, ONNX/TensorRT |
| `experiments/training/` | RF-DETR model eğitim notebook'u |

> **Not:** Detaylı benchmark sonuçları deneyler tamamlandıkça bu sayfaya eklenecektir.

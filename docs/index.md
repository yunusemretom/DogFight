---
layout: default
title: Ana Sayfa
description: "PX4 + ROS 2 ile RC Uçak Hedef Takip Sistemi"
---

<section class="hero fade-in">
  <img src="{{ '/assets/images/logo.png' | relative_url }}" alt="DogFight Logo" class="hero-logo">
  <div class="hero-subtitle">TEKNOFEST · Otonom Hava Muharebesi</div>
  <h1>DogFight</h1>
  <p class="hero-desc">
    ROS 2 Humble ve PX4 Autopilot tabanlı, yapay zeka destekli hedef takip ve dogfight (hava muharebesi) platformu. RC uçağı otonom olarak kontrol edip, GPS ve görüntü işleme verilerine dayanarak rakip aracı takip eder.
  </p>
  <div class="hero-actions">
    <a href="{{ '/guides/' | relative_url }}" class="btn btn-primary">📖 Başlangıç Rehberi</a>
    <a href="https://github.com/yunusemretom/DogFight" target="_blank" class="btn btn-secondary">⭐ GitHub'da Gör</a>
  </div>
</section>

<div class="stats-row fade-in">
  <div class="stat-card">
    <div class="stat-value">4</div>
    <div class="stat-label">ROS 2 Paketi</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">8+</div>
    <div class="stat-label">ROS 2 Node</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">3</div>
    <div class="stat-label">AI Modeli</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">3</div>
    <div class="stat-label">Kontrol Stratejisi</div>
  </div>
</div>

<div class="quick-links fade-in">
  <a href="{{ '/architecture/' | relative_url }}" class="quick-card">
    <span class="quick-card-icon">📐</span>
    <h3>Sistem Mimarisi</h3>
    <p>Katmanlı mimari, veri akışı diyagramları ve ROS 2 topic haritası</p>
  </a>
  <a href="{{ '/guides/' | relative_url }}" class="quick-card">
    <span class="quick-card-icon">📖</span>
    <h3>Kurulum Rehberleri</h3>
    <p>Sıfırdan sistemi kurmak için adım adım talimatlar</p>
  </a>
  <a href="{{ '/experiments/' | relative_url }}" class="quick-card">
    <span class="quick-card-icon">🧪</span>
    <h3>Deneyler</h3>
    <p>Model karşılaştırmaları, benchmark sonuçları ve test verileri</p>
  </a>
  <a href="{{ '/tools/' | relative_url }}" class="quick-card">
    <span class="quick-card-icon">🔧</span>
    <h3>Araçlar</h3>
    <p>Aerodinamik simülatör, GPS hesaplayıcı ve yardımcı araçlar</p>
  </a>
</div>

<div class="fade-in">

## 📸 ROS 2 Paketleri

| Paket | Sorumluluk | Node Sayısı |
|-------|-----------|-------------|
| `dogfight_detection` | Kamera görüntüsünden nesne tespiti (YOLO, RF-DETR) | 2 |
| `dogfight_tracking` | Hedef konum takibi (GPS / görsel) | 2 |
| `dogfight_control` | Offboard uçuş kontrolü (Attitude, Velocity, Position, Visual) | 4 |
| `dogfight_bringup` | Launch dosyaları ve merkezi konfigürasyon | — |

## 🌍 Simülasyon

Gazebo Harmonic simülasyonu, 3 adet RC Cessna uçak ile TEKNOFEST senaryosunu çalıştırır:

- **HERO** — Ana uçak (kameralı), kontrol ettiğimiz araç
- **ENEMY1** — Rakip uçak 1
- **ENEMY2** — Rakip uçak 2

**GPS Home:** Şanlıurfa GAP Havalimanı (LTCS) — Pist 04/22

</div>

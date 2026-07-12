---
layout: page
title: "L1 Takip Algoritması Rehberi"
description: "Mesafe korumalı L1 guidance takip node'unun mimarisi, çalışma mantığı, güvenlik önlemleri ve parametre ayarları"
icon: "🎯"
order: 5
---

Bu rehber, `dogfight_tracking` paketindeki **`l1_pursuit_node`** düğümünün çalışma mantığını, PX4 ile entegrasyon mimarisini ve güvenlik önlemlerini açıklar. Düğüm, bir sabit kanat İHA'nın başka bir İHA'yı **belirli bir mesafeyi koruyarak** takip etmesini sağlar.

- **Kod:** `ros2_ws/src/dogfight_tracking/dogfight_tracking/l1_pursuit_node.py`
- **Doğrulama:** Gazebo SITL, sürekli dönüş yapan (loiter) hedefe karşı — 5 m hedef mesafede ortalama 5.7 m, standart sapma 0.28 m

---

## 1. PX4 Entegrasyon Mimarisi

### Neden attitude veya velocity setpoint değil?

PX4 v1.16, sabit kanat offboard kontrolü için yeni bir arayüz sunar:

| Topic | İçerik | PX4 tarafında işleyen |
|---|---|---|
| `/fmu/in/fixed_wing_lateral_setpoint` | `lateral_acceleration` [m/s²] veya `course` [rad] | `FwLateralLongitudinalControl` → roll = atan(a/g) |
| `/fmu/in/fixed_wing_longitudinal_setpoint` | `altitude` (AMSL) + `equivalent_airspeed` | **TECS** (pitch + throttle) |

Bu arayüzün avantajı: **L1 guidance'ı biz hesaplarız**, ama roll dönüşümü, stall koruması, irtifa/hız enerji yönetimi (TECS) PX4 içinde kalır. Ham attitude setpoint kullansaydık pitch+throttle döngüsünü de kendimiz yazmak zorunda kalırdık; velocity setpoint ise sabit kanatta PX4'ün kendi guidance'ından geçer ve bizim L1'imiz devre dışı kalırdı.

### Kritik püf noktası: OffboardControlMode

```
OffboardControlMode.velocity = true   → FwLateralLongitudinalControl çalışır
TrajectorySetpoint YAYINLANMAZ        → FixedWingModeManager pasif kalır
                                        (FW_POSCTRL_MODE_OTHER)
```

`TrajectorySetpoint` yayınlanırsa PX4'ün kendi mode manager'ı devreye girer ve bizim lateral/longitudinal setpoint'lerimizle **çakışan** komutlar üretir. Bu düzen PX4 v1.16 kaynak kodundan doğrulanmıştır (`FixedWingModeManager.cpp` mod seçimi + `FwLateralLongitudinalControl.cpp` setpoint tüketimi).

### Koordinat çerçevesi

Her aracın NED orijini kendi kalkış noktasıdır; local pozisyonlar **doğrudan çıkarılamaz**. Düğüm:

- **Pozisyonları** `VehicleGlobalPosition` (lat/lon/alt AMSL) üzerinden karşılaştırır (flat-earth yaklaşımı, <~5 km için yeterli),
- **Hızları** `VehicleLocalPosition`'dan alır (NED eksenleri yerel olarak paraleldir, çıkarma güvenlidir).

---

## 2. Algoritmanın Çalışma Mantığı

Düğüm mesafeye göre **iki kontrol rejimi** arasında histerezisle geçiş yapar:

```
                 dist > formation_exit_dist (45 m)
        ┌────────────────────────────────────────────┐
        ▼                                            │
   ┌─────────┐                                  ┌──────────┐
   │ L1 MODU │                                  │ FORMASYON│
   │ (uzak)  │                                  │  (yakın) │
   └─────────┘                                  └──────────┘
        │                                            ▲
        └────────────────────────────────────────────┘
                 dist < formation_enter_dist (30 m)
```

### 2.1 Takip noktası: teğet değil, yay üzerinde

Her iki modda da hedeflenen nokta, hedefin **izlediği yolun** `standoff_distance` kadar gerisidir. Hedef dönüyorsa bu nokta hedefin hız vektörünün teğeti üzerinde değil, **dönüş yayı üzerinde** hesaplanır:

1. Hedefin dönüş hızı ω, course açısının filtreli türeviyle tahmin edilir.
2. Dönüş yarıçapı `R = V_hedef / |ω|`, dönüş merkezi hız vektörünün 90° yanında bulunur.
3. Takip noktası, hedefin konumu merkez etrafında `θ = standoff / R` kadar geriye döndürülerek elde edilir.

> **Neden önemli:** 80 m yarıçaplı dönüşte teğet üzerinde 50 m geri konulan nokta dairenin **14 m dışına** düşer ve kalıcı mesafe sapması yaratır. SITL testinde bu düzeltme kalıcı hatayı ~20 m'den ~6 m'ye indirdi.

### 2.2 L1 modu (uzak yakalama)

Klasik L1 guidance ile takip noktasına yönelinir:

```
L1_dist = ζ · T · V / π          (ζ: l1_damping, T: l1_period, V: yer hızı)
η       = ∠(hız vektörü → takip noktası),  ±90°'ye sınırlı
a_lat   = 2 V² sin(η) / L1_dist   → fixed_wing_lateral_setpoint.lateral_acceleration
```

Hız komutu mesafe hatasıyla orantılıdır: `V_cmd = V_hedef + kp_distance · (mesafe − standoff)`.

- `l1_period` küçüldükçe takip agresifleşir. **15 s bu uçaklar için çok gevşektir** (hedefin dairesinin dışında kilitlenen limit döngüsü üretir); 8 s doğrulanmış iyi bir değerdir.

### 2.3 Formasyon modu (yakın takip)

**L1 yakın mesafede kullanılamaz:** lookahead mesafesi (~40–70 m) takip noktasından çok uzakta kaldığı için küçük konum hataları büyük yön sıçramalarına (S-çizme / weaving) dönüşür. `formation_enter_dist` altında düğüm eksen ayrışımına geçer:

**Yanal kanal** — hedefin rotasını izler:
```
e_cross    = takip noktasının rotaya dik hatası
course_cmd = hedef_course + atan(kp_cross · e_cross / V)   (±max_course_corr_deg)
a_ff       = V · ω_hedef        ← viraj eşleme feedforward'u
```
`FixedWingLateralSetpoint`'te `course` sonluysa PX4, `lateral_acceleration` alanını feedforward olarak toplar — dönen hedefin arkasında aynı bank açısıyla oturmayı bu sağlar.

**Boylamsal kanal** — mesafe yalnızca hız komutuyla tutulur (**PID**):
```
e_along = takip noktasının rota boyu hatası
ė       = V_hedef − (kendi hızımın rota-boyu bileşeni)     ← analitik türev
V_cmd   = V_hedef + kp·e_along + ki·∫e_along + kd·ė
```

Üç terimin de gerekçesi SITL'de yaşanmış somut arızalardır:

| Terim | Yokluğunda ne oldu |
|---|---|
| **P** | Temel yakınsama yok |
| **I** | TECS'in ~0.5 m/s airspeed takip sapması, 5 m standoff'u **~1 m'ye** kaydırdı (çarpışma sınırı) |
| **D** | TECS gecikmesi (~2–3 s) yüzünden 2↔11 m osilasyon; analitik türev erken fren sağlayıp söndürdü |

Yanal ve boylamsal eksenler ayrıştığı için mesafe hatası artık yön komutunu bozmaz — salınımın kökten çözümü budur.

---

## 3. Durum Makinesi

```
WAIT_DATA ──veri hazır──▶ TAKEOFF ──irtifa > engage_altitude──▶ ENGAGE
                                                                  │ offboard aktif
              hedef verisi kesildi (> target_timeout)             ▼
   HOLD ◀──────────────────────────────────────────────────── PURSUIT
     └──────────────────veri geldi───────────────────────────────▲
                                                                  │
              offboard kaybı (failsafe / pilot) ──▶ ENGAGE ───────┘
```

- **WAIT_DATA/TAKEOFF/ENGAGE**'de de setpoint akışı sürer — PX4, offboard'a geçişten önce kesintisiz sinyal ister.
- `auto_arm_takeoff` ve `auto_offboard` yalnızca SITL kolaylığıdır; gerçek uçuşta ikisi de `false` yapılmalıdır.

---

## 4. Güvenlik Önlemleri

### 4.1 Düğümün içindeki katmanlar

1. **Minimum ayrılma katı** (`min_separation`, varsayılan 3 m, 3D mesafe): İhlal edilirse uçak `min_airspeed`'e yavaşlar ve **hedefin 5 m üstüne** irtifa açar; mesafe +2 m histerezisle geri kazanılınca takibe döner. İntegral bu sırada dondurulur (windup önleme).
2. **Hedef verisi zaman aşımı** (`target_timeout`, 2 s): Veri kesilirse HOLD — düz uçuş, mevcut irtifa, `cruise_airspeed`. Veri dönünce otomatik devam.
3. **Offboard kaybı toparlama:** Failsafe veya pilot mod değişimi algılanırsa (yalnızca `auto_offboard=true` iken) yeniden offboard istenir.
4. **Kalkışta irtifa kilidi** (`engage_altitude`, 15 m AGL): Bu irtifanın altında offboard'a geçilmez.
5. **Hız zarfı:** Tüm hız komutları `[min_airspeed, max_airspeed]` aralığına kırpılır; PX4 tarafında ayrıca `FW_AIRSPD_MIN/MAX` ve TECS stall koruması devrededir.
6. **Yanal ivme limiti** (`max_lateral_accel`, 8 m/s² ≈ 39° roll); PX4 kendi tarafında `FW_R_LIM` ile ikinci kez kırpar.
7. **İntegral sınırı:** ±2 m/s (windup önleme).

### 4.2 PX4 failsafe etkileşimi — ÖNEMLİ

SITL'de GCS (QGroundControl) bağlı değilse **datalink-loss failsafe'i offboard'u ~40 s sonra RTL'e düşürür**. Simülasyon için takipçi aracında şu değişiklik yapılmıştır:

```
param set NAV_DLL_ACT 0        # datalink kaybı failsafe kapalı (yalnızca SITL!)
param set COM_RCL_EXCEPT 4     # RC kaybı offboard'da failsafe tetiklemez
```

> ⚠️ **Gerçek uçuşta bu parametreler bu şekilde bırakılmamalıdır.** Telemetri linki canlı tutulmalı, failsafe davranışları (RTL irtifası, geofence) görev öncesi ayrıca planlanmalıdır.

### 4.3 Gerçek uçuş kısıtları

- **5 m standoff yalnızca simülasyon içindir.** İki bağımsız GPS/EKF'in bağıl konum hatası 2–5 m mertebesindedir; görsel servo veya RTK olmadan **15 m altına inilmemelidir**.
- Hedef konumu tek yönlü telemetriyle geliyorsa gecikme (latency) mesafe korumasını doğrudan bozar; `target_timeout` gerçek link kalitesine göre ayarlanmalıdır.
- `auto_arm_takeoff=false`, `auto_offboard=false` yapılmalı; offboard'a geçiş pilot elinde olmalıdır. Pilot her an mod anahtarıyla kontrolü geri alabilir — düğüm buna direnmez (yeniden istek yalnızca `auto_offboard=true` iken atılır).

---

## 5. Kullanım

```bash
# Varsayılan: takipçi /fmu (sysid 1), hedef /px4_1
ros2 run dogfight_tracking l1_pursuit_node

# Örnek: farklı araçlar + parametreler
ros2 run dogfight_tracking l1_pursuit_node --ros-args \
  -p follower_ns:=/px4_2 -p target_ns:=/px4_1 \
  -p standoff_distance:=25.0 -p l1_period:=8.0
```

Tüm sayısal parametreler **uçuş sırasında** değiştirilebilir:

```bash
ros2 param set /l1_pursuit_node standoff_distance 40.0
```

### Parametre tablosu

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `follower_ns` / `target_ns` | `''` / `/px4_1` | Araç namespace'leri |
| `standoff_distance` | 5.0 | Korunacak takip mesafesi [m] |
| `l1_period` / `l1_damping` | 15.0 / 0.75 | L1 agresifliği (8 s önerilir) |
| `max_lateral_accel` | 8.0 | Yanal ivme limiti [m/s²] |
| `min/max/cruise_airspeed` | 10 / 30 / 15 | Hız zarfı [m/s] |
| `kp_distance` / `ki_distance` / `kd_distance` | 0.15 / 0.08 / 0.6 | Mesafe PID kazançları |
| `formation_enter_dist` / `formation_exit_dist` | 30 / 45 | Mod geçiş histerezisi [m] |
| `kp_cross` / `max_course_corr_deg` | 0.4 / 35 | Formasyon yanal kazanç/limit |
| `min_separation` | 3.0 | Emniyet mesafesi (3D) [m] |
| `target_timeout` | 2.0 | Hedef verisi zaman aşımı [s] |
| `auto_arm_takeoff` / `auto_offboard` | true / true | SITL otomasyonu (gerçekte `false`!) |
| `engage_altitude` | 15.0 | Offboard'a geçiş için min AGL [m] |
| `system_id` | 0 | VehicleCommand hedefi (0 = ns'den türet) |

### Ayar (tuning) ipuçları

- **Mesafe sürekli hedefin üstünde kalıyorsa:** `ki_distance`'ı 0.12'ye çıkarın (kalan sapma TECS bias'ıdır).
- **Yaklaşırken taşma (overshoot) varsa:** `kd_distance`'ı artırın (0.8–1.0).
- **Uzak yakalamada geniş tur atıyorsa:** `l1_period`'u düşürün (6–8 s).
- **Formasyonda yanal sürüklenme varsa:** `kp_cross`'u artırın; titremeye başlarsa geri alın.

---

## 6. Doğrulanmış Test Sonuçları (Gazebo SITL)

| Senaryo | Sonuç |
|---|---|
| 50 m standoff, loiter'daki hedef | ~57 m'de kararlı, std 0.9 m |
| 5 m standoff, loiter'daki hedef | ort **5.68 m**, min 5.0 / max 6.3, **std 0.28 m** (2 dk) |
| Hedef verisi kesintisi | HOLD'a geçiş + otomatik devam ✓ |
| Failsafe ile offboard kaybı | Otomatik yeniden bağlanma ✓ |
| Minimum ayrılma ihlali (3 m) | Yavaşlama + irtifa açma, çarpışmasız kurtarma ✓ |

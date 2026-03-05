import math

def hesapla(lat1, lon1, lat2, lon2):
    # Dereceyi radyana çevir
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # --- MESAFe (Haversine) ---
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    mesafe_km = 6371 * 2 * math.asin(math.sqrt(a))

    # --- BEARING (hangi yöne uçmalısın) ---
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    bearing = math.degrees(math.atan2(x, y))
    bearing = (bearing + 360) % 360  # 0-360 arası normalize et

    return mesafe_km, bearing

# Kullanım
target_lat = 41.0
target_lon = 29.0

for i in range(100):
    my_lat = 48.8 + i * 0.01  # Paris'e doğru biraz hareket
    my_lon = 2.3 + i * 0.01
    mesafe, yon = hesapla(my_lat, my_lon, target_lat, target_lon)
    print(f"Mesafe: {mesafe:.1f} km")
    print(f"Uçman gereken yön: {yon:.1f}°")

print(round(math.degrees(1.4946233034133911)))  # 90 derece, yani doğuya bakıyor
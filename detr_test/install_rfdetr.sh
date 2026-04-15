#!/bin/bash
# ============================================================
# RF-DETR Bağımlılık Kurulum Scripti
# Python >= 3.10 gereklidir
# ============================================================

echo "========================================="
echo "  RF-DETR Kurulum Başlıyor..."
echo "========================================="

# Python versiyonunu kontrol et
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Mevcut Python versiyonu: $python_version"

# pip'i güncelle
echo ""
echo "[1/5] pip güncelleniyor..."
pip install --upgrade pip


# RF-DETR ana paketi
echo ""
echo "[3/5] RF-DETR kuruluyor..."
pip install rfdetr

# Görüntü işleme kütüphaneleri
echo ""
echo "[4/5] Yardımcı kütüphaneler kuruluyor..."
pip install \
    opencv-python \
    Pillow \
    numpy \
    matplotlib \
    supervision

# İsteğe bağlı: webcam / video için
echo ""
echo "[5/5] İsteğe bağlı paketler kuruluyor..."
pip install requests tqdm

echo ""
echo "========================================="
echo "  Kurulum Tamamlandı!"
echo "========================================="
echo ""
echo "Test etmek için:"
echo "  python test_rfdetr.py"

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WEBM to MP4 Video Converter
Bu script webm formatındaki video dosyalarını mp4 formatına dönüştürür.
"""

import subprocess
import sys
import os
from pathlib import Path


def convert_webm_to_mp4(input_file, output_file=None):
    """
    WEBM dosyasını MP4'e çevirir.
    
    Args:
        input_file (str): Giriş webm dosyasının yolu
        output_file (str, optional): Çıkış mp4 dosyasının yolu. 
                                     Belirtilmezse otomatik oluşturulur.
    
    Returns:
        bool: Dönüşüm başarılıysa True, değilse False
    """
    input_path = Path(input_file)
    
    # Dosyanın var olup olmadığını kontrol et
    if not input_path.exists():
        print(f"Hata: {input_file} dosyası bulunamadı!")
        return False
    
    # Çıkış dosyasını belirle
    if output_file is None:
        output_file = input_path.with_suffix('.mp4')
    
    output_path = Path(output_file)
    
    # FFmpeg komutunu hazırla (mobil cihazlar için optimize)
    # -i: input dosya
    # -c:v libx264: H.264 video codec kullan
    # -profile:v baseline: Mobil cihazlarla maksimum uyumluluk için baseline profile
    # -level 3.0: Uyumluluk seviyesi
    # -pix_fmt yuv420p: Mobil cihazlarla uyumlu pixel format
    # -c:a aac: AAC audio codec kullan
    # -b:a 128k: Audio bitrate (mobil için optimize)
    # -movflags +faststart: Web ve mobil oynatma için optimize et
    # -preset medium: Kodlama hızı/kalite dengesi
    cmd = [
        'ffmpeg',
        '-i', str(input_path),
        '-c:v', 'libx264',
        '-profile:v', 'baseline',
        '-level', '3.0',
        '-pix_fmt', 'yuv420p',
        '-preset', 'medium',
        '-c:a', 'aac',
        '-b:a', '128k',
        '-movflags', '+faststart',
        '-y',  # Çıkış dosyası varsa üzerine yaz
        str(output_path)
    ]
    
    try:
        print(f"Dönüştürülüyor: {input_path.name} -> {output_path.name}")
        print("FFmpeg çalışıyor...")
        
        # FFmpeg'i çalıştır
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        print(f"✓ Başarıyla dönüştürüldü: {output_path}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Hata: FFmpeg dönüştürme hatası!")
        print(f"Hata mesajı: {e.stderr}")
        return False
    except FileNotFoundError:
        print("✗ Hata: FFmpeg bulunamadı!")
        print("FFmpeg'i yüklemeniz gerekiyor:")
        print("  Ubuntu/Debian: sudo apt-get install ffmpeg")
        print("  Fedora: sudo dnf install ffmpeg")
        print("  MacOS: brew install ffmpeg")
        return False


def convert_directory(directory, recursive=False):
    """
    Bir klasördeki tüm webm dosyalarını dönüştürür.
    
    Args:
        directory (str): Klasör yolu
        recursive (bool): Alt klasörlere de bakılsın mı?
    """
    dir_path = Path(directory)
    
    if not dir_path.is_dir():
        print(f"Hata: {directory} geçerli bir klasör değil!")
        return
    
    # Webm dosyalarını bul
    if recursive:
        pattern = '**/*.webm'
    else:
        pattern = '*.webm'
    
    webm_files = list(dir_path.glob(pattern))
    
    if not webm_files:
        print(f"Klasörde webm dosyası bulunamadı: {directory}")
        return
    
    print(f"{len(webm_files)} adet webm dosyası bulundu.\n")
    
    success_count = 0
    fail_count = 0
    
    for webm_file in webm_files:
        if convert_webm_to_mp4(str(webm_file)):
            success_count += 1
        else:
            fail_count += 1
        print()  # Boş satır
    
    print(f"\n{'='*50}")
    print(f"Toplam: {len(webm_files)} dosya")
    print(f"Başarılı: {success_count}")
    print(f"Başarısız: {fail_count}")


def main():
    """Ana fonksiyon - komut satırı argümanlarını işler"""
    
    if len(sys.argv) < 2:
        print("WEBM to MP4 Video Converter")
        print("=" * 50)
        print("\nKullanım:")
        print(f"  {sys.argv[0]} <dosya.webm>")
        print(f"  {sys.argv[0]} <dosya.webm> <çıkış.mp4>")
        print(f"  {sys.argv[0]} --dir <klasör>")
        print(f"  {sys.argv[0]} --dir <klasör> --recursive")
        print("\nÖrnekler:")
        print(f"  {sys.argv[0]} video.webm")
        print(f"  {sys.argv[0]} video.webm output.mp4")
        print(f"  {sys.argv[0]} --dir ./videos")
        print(f"  {sys.argv[0]} --dir ./videos --recursive")
        sys.exit(1)
    
    # Klasör modu
    if sys.argv[1] == '--dir':
        if len(sys.argv) < 3:
            print("Hata: Klasör yolu belirtilmedi!")
            sys.exit(1)
        
        directory = sys.argv[2]
        recursive = '--recursive' in sys.argv or '-r' in sys.argv
        
        convert_directory(directory, recursive)
    
    # Tek dosya modu
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        
        success = convert_webm_to_mp4(input_file, output_file)
        sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

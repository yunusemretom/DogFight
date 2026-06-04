from ultralytics import YOLO

# Model yükleme
model = YOLO("/home/tom/Downloads/epoch40.pt")

# TensorRT'ye dönüştürme
model.export(
    format="engine",
    device=0,
    half=True,
    simplify=True,
    workspace=4,
    verbose=True,
    task='detect'  # Task'ı açıkça belirt
)

print("Model dönüştürme tamamlandı. 'model.engine' dosyası oluşturuldu.")

# Dönüştürülen modeli test et
try:
    # Modeli doğrudan task parametresi ile yükle
    engine_model = YOLO("model.engine", task='detect')
    print("Engine model başarıyla yüklendi!")
    
    # Test için basit bir görüntü kullan (video yerine)
    import cv2
    import numpy as np
    
    # Test görüntüsü oluştur veya yükle
    test_image = np.zeros((640, 640, 3), dtype=np.uint8)
    results = engine_model(test_image)
    print(f"Test başarılı: {len(results[0].boxes)} nesne tespit edildi")
    
except Exception as e:
    print(f"Hata oluştu: {e}")
    print("\nÖnerilen çözümler:")
    print("1. TensorRT'yi kaldırıp yeniden yükleyin:")
    print("   pip uninstall tensorrt")
    print("   pip install tensorrt==8.6.1")
    print("\n2. CUDA sürümünüzü kontrol edin:")
    print("   nvidia-smi")
    print("\n3. Gerekli sürümler:")
    print("   - CUDA 11.8+")
    print("   - TensorRT 8.6.1")
    print("   - PyTorch 2.0+")
    print("\n4. Eğer sorun devam ederse, model.pt dosyasını kullanmaya devam edin.")

# Run inference
model = YOLO("best.engine")
results = model("https://ultralytics.com/images/bus.jpg")

"""
def quantize_onnx_model(onnx_model_path, quantized_model_path):
    from onnxruntime.quantization import quantize_dynamic, QuantType
    import onnx
    onnx_opt_model = onnx.load(onnx_model_path)
    quantize_dynamic(onnx_model_path,
    quantized_model_path,
    weight_type=QuantType.QUInt8) #chnage QInt8 to QUInt8 
    
quantize_onnx_model("best.onnx","best1.onnx")
"""

"""
from ultralytics import YOLO

model = YOLO("best.pt")
model.export(
    format="engine",
    dynamic=True,  
      
    int8=True  
)

# Load the exported TensorRT INT8 model
model = YOLO("yolov8n.engine", task="detect")

# Run inference
result = model.predict("https://ultralytics.com/images/bus.jpg")
"""
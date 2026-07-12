"""
RF-DETR ONNX -> TensorRT Engine Export
TensorRT 11+ uyumlu (--fp16 flag'i kaldirildi, Python API ile FP16 destegi)
"""
import subprocess
from pathlib import Path

ONNX_PATH = "object_detection/inference_model.onnx"
ENGINE_PATH = "object_detection/inference_model.engine"
WORKSPACE_MB = 4096  # MB cinsinden workspace bellegi
VERBOSE = True

def build_engine_python_api() -> None:
    """
    TensorRT 11 Python API ile FP16 engine olusturur.
    trtexec'te --fp16 kaldirildi; bu yontem daha esnek ve tam destekli.
    """
    try:
        import tensorrt as trt
    except ImportError:
        raise ImportError(
            "TensorRT Python modulu bulunamadi. "
            "Kur: pip install tensorrt-cu12 veya sistem TRT kurulumunu kontrol et."
        )

    onnx_path = Path(ONNX_PATH)
    engine_path = Path(ENGINE_PATH)

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model bulunamadi: {onnx_path.resolve()}")

    engine_path.parent.mkdir(parents=True, exist_ok=True)

    logger = trt.Logger(trt.Logger.VERBOSE if VERBOSE else trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(0)  # strongly typed (TRT11 default)
    parser = trt.OnnxParser(network, logger)

    print(f"[INFO] ONNX model yukleniyor: {onnx_path.resolve()}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"[ERROR] {parser.get_error(i)}")
            raise RuntimeError("ONNX parse hatasi!")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_MB * (1 << 20))

    # FP16 destegi - GPU destekliyorsa etkinlestir
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("[INFO] FP16 precision etkin.")
    else:
        print("[WARNING] GPU FP16 desteklemiyor, FP32 kullanilacak.")

    print("[INFO] Engine derleniyor... (bu birkaç dakika surebilir)")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Engine derleme basarisiz!")

    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    size_mb = engine_path.stat().st_size / (1024 ** 2)
    print(f"[INFO] Engine kaydedildi: {engine_path.resolve()} ({size_mb:.1f} MB)")


def build_engine_trtexec() -> None:
    """
    trtexec CLI ile engine olusturur (TRT 11 uyumlu, --fp16 olmadan).
    Python API basarisiz olursa yedek olarak kullanilir.
    """
    onnx_path = Path(ONNX_PATH)
    engine_path = Path(ENGINE_PATH)
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    # TRT 11'de kaldirilan deprecated flag'ler: --fp16, --useCudaGraph, --useSpinWait
    cmd_parts = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--memPoolSize=workspace:{WORKSPACE_MB}",
        "--warmUp=200",
        "--avgRuns=100",
        "--duration=5",
    ]
    if VERBOSE:
        cmd_parts.append("--verbose")

    command = " ".join(cmd_parts)
    print(f"[INFO] Komut: {command}")

    result = subprocess.run(command, shell=True, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"trtexec hatasi (exit code {result.returncode})")

    print(f"[INFO] Engine kaydedildi: {engine_path.resolve()}")


if __name__ == "__main__":
    print("=== RF-DETR TensorRT Export ===")
    print("Yontem: TensorRT Python API (TRT 11 uyumlu)")
    try:
        build_engine_python_api()
    except ImportError as e:
        print(f"[WARNING] Python API kullanilamadi: {e}")
        print("trtexec CLI yontemi deneniyor...")
        build_engine_trtexec()
    print("=== Export tamamlandi ===")
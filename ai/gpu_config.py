import torch
from torch.amp import GradScaler

# RTX 3060 (Ampere, compute capability 8.6) 기준 배치 크기
BATCH = {
    'lstm': 256,
    'bert': 16,
}
NUM_WORKERS = 4
PIN_MEMORY  = True

_device: torch.device | None = None


def setup_rtx3060() -> torch.device:
    """CUDA 디바이스 초기화. GPU 없으면 CPU로 폴백."""
    global _device
    if _device is not None:
        return _device

    if torch.cuda.is_available():
        _device = torch.device('cuda')
        props   = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024 ** 3)

        # Ampere TF32 매트멀 가속 + cuDNN autotune
        torch.backends.cudnn.benchmark        = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True

        print(f"[GPU] {torch.cuda.get_device_name(0)} 감지 | "
              f"VRAM={vram_gb:.1f}GB | cudnn.benchmark=True, TF32=True")
    else:
        _device = torch.device('cpu')
        print("[GPU] CUDA 사용 불가 -> CPU로 폴백")

    return _device


def amp_dtype() -> torch.dtype:
    """
    Ampere(3060)는 bf16 지원 -> overflow 위험 없는 bf16 우선 사용.
    CUDA가 없는 CPU 폴백에서는 float32를 반환한다. autocast(device_type='cuda', ...)는
    CUDA 미존재 시 자동으로 비활성화(no-op)되는데, 그 상태에서 입력만 fp16/bf16으로
    캐스팅하면 fp32 파라미터를 가진 레이어(LSTM 등)와 dtype이 어긋나 바로 크래시한다.
    """
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def get_scaler() -> GradScaler:
    """bf16 사용 시 GradScaler는 불필요(오버플로우 위험 없음) -> 그 경우 no-op으로 비활성화."""
    return GradScaler('cuda', enabled=(amp_dtype() == torch.float16))


def print_vram(label: str = ''):
    if not torch.cuda.is_available():
        return
    alloc    = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved()  / (1024 ** 3)
    tag = f' - {label}' if label else ''
    print(f"[VRAM{tag}] allocated={alloc:.2f}GB reserved={reserved:.2f}GB")

import torch

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


def print_vram(label: str = ''):
    if not torch.cuda.is_available():
        return
    alloc    = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved()  / (1024 ** 3)
    tag = f' - {label}' if label else ''
    print(f"[VRAM{tag}] allocated={alloc:.2f}GB reserved={reserved:.2f}GB")

import torch

print("PyTorch Version:", torch.__version__)
print("CUDA verfügbar?:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA Geräteanzahl:", torch.cuda.device_count())
    print("CUDA Gerät:", torch.cuda.get_device_name(0))
else:
    print("[ERROR] Keine CUDA-Unterstützung erkannt!")


import torch

print(f"Device available: {torch.cuda.is_available()}")

print(f"Number of devices: {torch.cuda.device_count()}")

print(f"Current device: {torch.cuda.current_device()}")

print(f"Name of device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
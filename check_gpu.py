import torch

# Check if MPS (Metal Performance Shaders) is available
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    print("✅ Success! M4 Pro GPU (MPS) is detected.")
    print(f"Device set to: {mps_device}")
else:
    print("❌ Warning! MPS not detected. CPU will be used (Slower).")
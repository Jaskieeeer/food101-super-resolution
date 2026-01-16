import torch
import os
import platform

print(f"Python Version: {platform.python_version()}")
print(f"PyTorch Version: {torch.__version__}")

# Check for Apple Silicon GPU (MPS)
if torch.backends.mps.is_available():
    print("✅ Apple M4 GPU (MPS) is available! Acceleration is ON.")
    device = torch.device("mps")
    
    # Test a small calculation on the GPU
    x = torch.ones(1, device=device)
    print(f"   Test tensor created on: {x.device}")
else:
    print("⚠️ MPS not detected. You are running on CPU (Slower).")

required_folders = ['data', 'src', 'reports', 'weights']
print("\nChecking folders...")
for folder in required_folders:
    if os.path.exists(folder):
        print(f"✅ {folder} exists")
    else:
        print(f"❌ {folder} missing")

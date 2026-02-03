import torch
import sys

def sanity_check():
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    print("-" * 20)

    # 1. Check if CUDA is available
    if torch.cuda.is_available():
        print("✅ SUCCESS: CUDA is available!")
        
        # 2. Get Device Details
        device_count = torch.cuda.device_count()
        print(f"   GPU Count: {device_count}")
        
        for i in range(device_count):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            
        # 3. Check Memory (Optional but good to know)
        prop = torch.cuda.get_device_properties(0)
        print(f"   Total VRAM: {prop.total_memory / 1e9:.2f} GB")
        
    else:
        print("❌ FAILURE: CUDA is NOT available.")
        print("   Training will run on CPU (very slow).")
        print("   Check your NVIDIA drivers and PyTorch installation.")

if __name__ == "__main__":
    sanity_check()
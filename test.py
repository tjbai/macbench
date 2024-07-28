import torch
import time
import platform
import psutil
from tqdm import tqdm

def log_system_info():
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Processor: {platform.processor()}")
    print(f"Python version: {platform.python_version()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")

def run(device, size):
    torch.manual_seed(42)
    
    s = time.time()
    for _ in tqdm(range(100)):
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        c = torch.matmul(a, b)

    torch.cuda.synchronize() if device == 'cuda' else torch.mps.synchronize() if device == 'mps' else None
    return time.time() - s

def main():
    log_system_info()
    
    devices = ['cpu', 'mps'] if torch.backends.mps.is_available() else ['cpu']
    sizes = [1000, 2000, 4000]
    
    for device in devices:
        print(f"\nRunning tests on {device.upper()}:")
        for size in sizes:
            time_taken = run(device, size)
            print(f"Matrix multiplication ({size}x{size}) - Time: {time_taken:.4f} seconds")
    
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"\nMemory Usage: {memory_info.rss / (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    main()

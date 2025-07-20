import torch
import psutil

def print_memory_usage():
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    print(f"RAM Usage: {psutil.virtual_memory().percent}%")

if __name__ == "__main__":
    print_memory_usage()
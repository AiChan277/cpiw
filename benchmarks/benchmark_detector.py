import argparse
import time
import numpy as np

def benchmark_detector(device, model, iterations, resolution):
    print(f"Benchmarking detector on {device} with {iterations} iterations...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="CPU")
    parser.add_argument("--model", type=str, default="default")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--resolution", type=str, default="1280x720")
    args = parser.parse_args()
    
    benchmark_detector(args.device, args.model, args.iterations, args.resolution)

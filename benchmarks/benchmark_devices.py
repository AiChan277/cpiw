import argparse

def benchmark_devices(model, iterations, resolution):
    print("Comparing devices...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="default")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--resolution", type=str, default="1280x720")
    args = parser.parse_args()
    
    benchmark_devices(args.model, args.iterations, args.resolution)

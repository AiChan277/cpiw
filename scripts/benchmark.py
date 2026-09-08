import argparse
import subprocess

def run_benchmark():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="CPU")
    parser.add_argument("--resolution", type=str, default="1280x720")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()

    print("Running benchmarks...")

if __name__ == "__main__":
    run_benchmark()

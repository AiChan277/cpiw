import argparse
import subprocess

def package_app(onefile, output_dir):
    print("Packaging PrivacyCam...")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onefile", action="store_true")
    parser.add_argument("--output-dir", type=str, default="dist/")
    args = parser.parse_args()
    
    package_app(args.onefile, args.output_dir)

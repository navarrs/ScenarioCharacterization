import os
import shutil
import sys
import random

def copy_sample_files(input_folder, output_folder, sample_size=10):
    if not os.path.isdir(input_folder):
        print(f"Input folder '{input_folder}' does not exist.")
        return

    os.makedirs(output_folder, exist_ok=True)
    files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    print(files)
    if len(files) < sample_size:
        print(f"Not enough files in input folder. Found {len(files)}, need {sample_size}.")
        return

    sample_files = random.sample(files, sample_size)
    for filename in sample_files:
        src = os.path.join(input_folder, filename)
        dst = os.path.join(output_folder, filename)
        shutil.copy2(src, dst)
    print(f"Copied {sample_size} files to '{output_folder}'.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python get_sample.py <input_folder> <output_folder>")
        sys.exit(1)
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    copy_sample_files(input_folder, output_folder)
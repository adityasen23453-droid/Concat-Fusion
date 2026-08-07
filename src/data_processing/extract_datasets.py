import zipfile
import os
from tqdm import tqdm

def extract_zip(zip_path, extract_dir):
    print(f"Extracting {zip_path} to {extract_dir}...")
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        files = zip_ref.namelist()
        for file in tqdm(files, desc=os.path.basename(zip_path)):
            zip_ref.extract(file, extract_dir)
    print(f"Successfully extracted {zip_path}!")

if __name__ == "__main__":
    # Extract face dataset
    extract_zip("Face dataset/FEI face data.zip", "Face dataset/extracted")
    
    # Extract iris dataset
    extract_zip("iris dataset/CASIA-Iris-Thousand.zip", "iris dataset/extracted")

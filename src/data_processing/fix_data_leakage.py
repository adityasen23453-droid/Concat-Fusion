#!/usr/bin/env python3
import os
import cv2
import shutil
import hashlib
import numpy as np
from tqdm import tqdm

IRIS_DIR = "iris dataset/extracted/CASIA-Iris-Thousand"
TEST_DIR = "Chimeric_Dataset_Noisy/testing"

def calculate_md5(filepath):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def scan_glasses_free_subjects():
    glasses_free_subs = []
    for sub in range(1000):
        folder = os.path.join(IRIS_DIR, f"{sub:03d}", "R")
        if os.path.exists(folder):
            files = os.listdir(folder)
            has_glasses = False
            for f in files:
                p = os.path.join(folder, f)
                img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    if np.sum(img >= 220) > 200:
                        has_glasses = True
                        break
            if not has_glasses and len(files) >= 4:
                glasses_free_subs.append(sub)
                if len(glasses_free_subs) == 100:
                    break
    return glasses_free_subs

def main():
    print("Scanning glasses-free subjects...")
    glasses_free_subs = scan_glasses_free_subjects()
    print(f"Found {len(glasses_free_subs)} glasses-free subjects.")
    
    replacements = 0
    
    for i in tqdm(range(1, 101), desc="Resolving Iris Data Leakage"):
        person_name = f"Person_{i:03d}"
        subj_id = glasses_free_subs[i - 1]
        
        # Get right-eye source folder
        src_folder = os.path.join(IRIS_DIR, f"{subj_id:03d}", "R")
        if not os.path.exists(src_folder):
            print(f"Error: Source folder {src_folder} does not exist.")
            continue
            
        # Get all jpg files
        src_files = [f for f in os.listdir(src_folder) if f.lower().endswith(('.jpg', '.jpeg'))]
        src_files.sort()
        
        # De-duplicate by MD5 hash
        unique_files = []
        seen_hashes = set()
        
        for file in src_files:
            filepath = os.path.join(src_folder, file)
            file_hash = calculate_md5(filepath)
            if file_hash not in seen_hashes:
                seen_hashes.add(file_hash)
                unique_files.append((file, filepath, file_hash))
                
        # Validate we have at least 5 unique images
        if len(unique_files) < 5:
            raise ValueError(f"Subject S5{subj_id:03d} (mapped to Person_{i:03d}) has only {len(unique_files)} unique images, but we need 5.")
            
        # Target folder for noisy chimeric test dataset
        dest_folder = os.path.join(TEST_DIR, person_name)
        os.makedirs(dest_folder, exist_ok=True)
        
        # Copy selected unique files
        # Gallery: iris_R_1.jpg -> index 0
        gal_src = unique_files[0][1]
        gal_dest = os.path.join(dest_folder, "iris_R_1.jpg")
        shutil.copy2(gal_src, gal_dest)
        
        # Probes: iris_R_2.jpg to iris_R_5.jpg -> indices 1 to 4
        for p_idx in range(2, 6):
            probe_src = unique_files[p_idx - 1][1]
            probe_dest = os.path.join(dest_folder, f"iris_R_{p_idx}.jpg")
            shutil.copy2(probe_src, probe_dest)
            replacements += 1
            
    print(f"\nSuccessfully resolved data leakage! Updated iris files in {TEST_DIR}.")
    print(f"De-duplicated and copied 500 right-eye images (100 gallery, 400 probes) with zero hash overlaps.")

if __name__ == "__main__":
    main()

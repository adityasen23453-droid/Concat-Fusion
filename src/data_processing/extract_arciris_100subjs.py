#!/usr/bin/env python3
"""
Extract 512-D ArcIris ResNet100 Embeddings for all 100 Subjects.
Saves gallery and probe embeddings to arciris_templates_cache_100.pkl.
"""

import os
import sys
import glob
import time
import pickle
import numpy as np
import cv2
import torch
from tqdm import tqdm

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("OpenSourceIrisRecognition/methods/ArcIris/Python"))

from open_iris_pipeline import OpenIrisPipelineManager
from modules.network import iresnet100

def extract_arciris_embedding(img_path, iris_mgr, model, device):
    try:
        _, _ = iris_mgr.generate_biometric_template(img_path, eye_side="right")
        norm_img = iris_mgr.last_normalized_image
        if norm_img is None:
            return None
        
        if norm_img.ndim == 2:
            img_3ch = cv2.cvtColor((norm_img * 255).astype(np.uint8) if norm_img.max() <= 1.0 else norm_img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        else:
            img_3ch = norm_img

        img_resized = cv2.resize(img_3ch, (512, 64))
        tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        tensor = (tensor - 0.5) / 0.5
        tensor = tensor.to(device)

        with torch.no_grad():
            emb = model(tensor).cpu().numpy().flatten()

        norm_val = np.linalg.norm(emb)
        if norm_val > 1e-9:
            emb = emb / norm_val
        return emb
    except Exception as e:
        print(f"Warning: ArcIris extraction failed for {img_path}: {e}")
        return None

def main():
    print("=" * 70)
    print("      EXTRACTING 512-D ARCIRIS EMBEDDINGS FOR 100 SUBJECTS      ")
    print("=" * 70)

    cache_output = "arciris_templates_cache_100.pkl"
    if os.path.exists(cache_output):
        print(f"ArcIris cache already exists at {cache_output}. Skipping extraction.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device: {device}")

    # Load Model
    print("Loading ArcIris ResNet100 Model...")
    model = iresnet100(num_features=512)
    weights_path = "OpenSourceIrisRecognition/methods/ArcIris/Python/models/ResNet100_154000.pt"
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    iris_mgr = OpenIrisPipelineManager()

    gallery_dict = {}
    probes_dict = {}

    dataset_noisy_dir = "Chimeric_Dataset_Noisy"
    dataset_clean_dir = "Chimeric_Dataset"

    print("Extracting ArcIris embeddings across Persons 001 to 100...")
    start_time = time.time()

    for i in tqdm(range(1, 101), desc="Extracting 100 Subjects"):
        person_name = f"Person_{i:03d}"
        
        # Paths
        test_dir = os.path.join(dataset_noisy_dir, "testing", person_name)
        if not os.path.exists(test_dir):
            test_dir = os.path.join(dataset_clean_dir, "testing", person_name)

        train_dir = os.path.join(dataset_noisy_dir, "training", person_name)
        if not os.path.exists(train_dir):
            train_dir = os.path.join(dataset_clean_dir, "training", person_name)

        # 1. Gallery
        gal_img_path = os.path.join(train_dir, "iris_right.jpg")
        if not os.path.exists(gal_img_path):
            gal_img_path = os.path.join(test_dir, "iris_R_1.jpg")

        gal_emb = extract_arciris_embedding(gal_img_path, iris_mgr, model, device)
        gallery_dict[i] = {
            "embedding": gal_emb,
            "filename": os.path.basename(gal_img_path),
            "quality": 0.85 if gal_emb is not None else 0.0
        }

        # 2. Probes (iris_R_1.jpg to iris_R_5.jpg)
        iris_files = sorted(glob.glob(os.path.join(test_dir, "iris_R_*.jpg")))
        if len(iris_files) == 0:
            iris_files = sorted(glob.glob(os.path.join(test_dir, "iris_right*.jpg")))

        probes_dict[i] = []
        for file_path in iris_files:
            fn = os.path.basename(file_path)
            emb = extract_arciris_embedding(file_path, iris_mgr, model, device)
            probes_dict[i].append({
                "embedding": emb,
                "filename": fn,
                "quality": 0.80 if emb is not None else 0.0
            })

    print(f"Saving extracted ArcIris templates to {cache_output}...")
    with open(cache_output, "wb") as f:
        pickle.dump({"gallery": gallery_dict, "probes": probes_dict}, f)

    print(f"Successfully cached ArcIris embeddings for {len(gallery_dict)} subjects in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()

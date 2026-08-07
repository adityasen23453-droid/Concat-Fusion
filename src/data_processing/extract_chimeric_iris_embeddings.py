import os
import sys
import argparse
import numpy as np
import cv2
from tqdm import tqdm
from open_iris_pipeline import OpenIrisPipelineManager

def main():
    parser = argparse.ArgumentParser(description="Chimeric Iris Embedding Extraction")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="Chimeric_Dataset",
        help="Path to chimeric dataset root folder"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="iris_embeddings",
        help="Directory to save extracted embeddings"
    )
    args = parser.parse_args()

    print("Initialising Open-IRIS Pipeline for Chimeric Training Data...")
    irisRec = OpenIrisPipelineManager()
    print("Models loaded successfully!")
    
    # Iterate through Chimeric Dataset training folders (Person_001 to Person_100)
    train_dir = os.path.join(args.dataset_dir, "training")
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Chimeric training directory not found at {train_dir}")
        
    image_tasks = []
    for i in range(1, 101):
        person_name = f"Person_{i:03d}"
        img_path = os.path.join(train_dir, person_name, "iris_right.jpg")
        if os.path.exists(img_path):
            image_tasks.append({
                'path': img_path,
                'subject_id': person_name,
                'side': 'R',
                'filename': 'iris_right.jpg'
            })
        else:
            print(f"Warning: Missing training iris for {person_name} at {img_path}")
            
    print(f"Found {len(image_tasks)} training iris images to process.")
    
    processed_count = 0
    failed_count = 0
    
    # Loop over images
    for task in tqdm(image_tasks, desc="Extracting Chimeric Iris Embeddings"):
        img_path = task['path']
        subj_id = task['subject_id']
        side = task['side']
        filename = task['filename']
        
        try:
            # Extract template using Open-IRIS
            iris_code, noise_mask = irisRec.generate_biometric_template(img_path, eye_side=side)
            
            # Save separate .npy file matching the fingerprint/face structure
            save_dir = os.path.join(args.output_dir, "training", subj_id)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{subj_id}_iris.npy")
            np.save(save_path, {
                "iris_codes": iris_code,
                "mask_codes": noise_mask,
                "version": "v0.1"
            }, allow_pickle=True)
            
            processed_count += 1
            
        except Exception as ex:
            print(f"\nError processing {img_path}: {ex}")
            failed_count += 1
            
    print(f"\nExtraction complete! Processed: {processed_count}, Failed: {failed_count}")
    print(f"Embeddings saved to individual files under directory: {args.output_dir}")

if __name__ == "__main__":
    main()


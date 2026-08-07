import os
import sys
import subprocess
import numpy as np
import cv2
from tqdm import tqdm
from open_iris_pipeline import OpenIrisPipelineManager

def main():
    print("Initialising Open-IRIS Pipeline...")
    irisRec = OpenIrisPipelineManager()
    print("Models loaded successfully!")
    
    # Dataset Iteration
    dataset_dir = "iris dataset/extracted/CASIA-Iris-Thousand"
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found at {dataset_dir}")
        
    # Gather subjects (all 1000 subjects: 000 to 999)
    subjects = [f"{i:03d}" for i in range(1000)]
    print(f"Iterating through subjects: {len(subjects)} subjects")
    
    # Collect all image paths to process
    image_tasks = []
    for subj in subjects:
        subj_path = os.path.join(dataset_dir, subj)
        if not os.path.exists(subj_path):
            print(f"Warning: Subject folder {subj_path} does not exist. Skipping.")
            continue
        for side in ['R']:
            side_path = os.path.join(subj_path, side)
            if not os.path.exists(side_path):
                continue
            # Get all JPEGs
            filenames = [f for f in os.listdir(side_path) if f.lower().endswith(('.jpg', '.jpeg'))]
            for fn in filenames:
                full_path = os.path.join(side_path, fn)
                image_tasks.append({
                    'path': full_path,
                    'subject_id': subj,
                    'side': side,
                    'filename': fn
                })
                
    print(f"Found {len(image_tasks)} iris images to process.")
    
    # Storage arrays
    embeddings = []
    subject_ids = []
    sides = []
    image_ids = []
    
    # We will save normalized polar images for visual verification (first few samples)
    # Save directly to the brain's artifact directory to keep them visible
    artifact_dir = "C:/Users/adity/.gemini/antigravity-ide/brain/ac09a221-18a1-4321-97f2-20817fd0429c"
    os.makedirs(artifact_dir, exist_ok=True)
    visual_save_count = 5
    
    processed_count = 0
    failed_count = 0
    
    # Loop over images
    for task in tqdm(image_tasks, desc="Extracting Iris Embeddings"):
        img_path = task['path']
        subj_id = task['subject_id']
        side = task['side']
        filename = task['filename']
        
        try:
            # Extract template using Open-IRIS
            iris_code, noise_mask = irisRec.generate_biometric_template(img_path, eye_side=side)
            
            template_dict = {
                "iris_codes": iris_code,
                "mask_codes": noise_mask,
                "version": "v0.1"
            }
            
            # Store results
            embeddings.append(template_dict)
            subject_ids.append(subj_id)
            sides.append(side)
            image_ids.append(filename)
            
            processed_count += 1
            
        except Exception as ex:
            print(f"\nError processing {img_path}: {ex}")
            failed_count += 1
            
    print(f"\nExtraction complete! Processed: {processed_count}, Failed: {failed_count}")
    
    # 8. Save compressed NPZ file
    output_filename = "casia_iris_embeddings.npz"
    np.savez_compressed(
        output_filename,
        embeddings=np.array(embeddings, dtype=object),
        subject_ids=np.array(subject_ids, dtype=object),
        sides=np.array(sides, dtype=object),
        image_ids=np.array(image_ids, dtype=object)
    )
    print(f"Embeddings saved to {output_filename} successfully!")
    
    # Print quick shape and verification checks
    data = np.load(output_filename, allow_pickle=True)
    print("Verification of saved NPZ:")
    print("  Keys:", list(data.keys()))
    print("  embeddings shape:", data['embeddings'].shape)
    print("  subject_ids shape:", data['subject_ids'].shape)
    print("  sides shape:", data['sides'].shape)
    print("  image_ids shape:", data['image_ids'].shape)

if __name__ == "__main__":
    main()

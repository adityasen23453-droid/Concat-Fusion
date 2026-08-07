#!/usr/bin/env python3
"""
DeepPrint Fingerprint Feature Extraction Script.

This script extracts 512-dimensional fingerprint embeddings using a pre-trained
DeepPrint_TexMinu model (8,000 subjects) on chimeric gallery enrollments and probe variations.
It incorporates Gabor baborization preprocessing to match the training modality,
L2-normalizes the final concatenated embeddings, and stores them as float32 NumPy files.

Author: Expert Biometrics & Machine Learning Engineer
Date: June 24, 2026
"""

import os
import shutil
import argparse
import logging
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time

# Import DeepPrint architecture and preprocessing helpers from flx package
from flx.models.deep_print_arch import DeepPrint_TexMinu
from flx.data.image_helpers import pad_and_resize_to_deepprint_input_size
from flx.image_processing.binarization import LazilyAllocatedBinarizer

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class ChimericFingerprintDataset(Dataset):
    """
    PyTorch Dataset wrapper for chimeric fingerprints.
    Performs image loading, padding/resizing, and binarization.
    """
    def __init__(self, file_list, binarizer):
        self.file_list = file_list
        self.binarizer = binarizer

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path, save_path = self.file_list[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to read image at: {path}")
        
        # Pad and resize with white background fill
        preprocessed = pad_and_resize_to_deepprint_input_size(img, fill=1.0)
        
        # Binarize ridges and valleys to match DeepPrint model training modality
        if self.binarizer is not None:
            preprocessed = self.binarizer(preprocessed)
            
        return preprocessed, save_path


def main():
    parser = argparse.ArgumentParser(description="DeepPrint Fingerprint Feature Extraction")
    parser.add_argument(
        "--dataset_dir", 
        type=str, 
        default="Chimeric_Dataset", 
        help="Path to chimeric dataset root folder"
    )
    parser.add_argument(
        "--fvc_dir", 
        type=str, 
        default="FVC2002/Dbs/Db1_a", 
        help="Path to raw FVC2002 database"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="fingerprint_embeddings", 
        help="Path to save extracted embeddings"
    )
    parser.add_argument(
        "--weights_path", 
        type=str, 
        default="best_model.pyt", 
        help="Path to pretrained DeepPrint weights (.pyt file)"
    )
    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=0, 
        help="Number of workers for data loader (0 is optimal on CPU to prevent thread thrashing)"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=32, 
        help="Batch size for model inference"
    )
    args = parser.parse_args()

    # Limit PyTorch to a reasonable number of threads to prevent OpenMP CPU thrashing
    torch.set_num_threads(8)

    # Route device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Execution Device: {device}")

    # Step 1: Directory Purge & Isolated Initialization
    if os.path.exists(args.output_dir):
        logger.info(f"Purging existing directory: {args.output_dir}")
        # Security Safeguard: Prevent deleting face/iris directories
        out_lower = args.output_dir.lower()
        if "face" in out_lower or "iris" in out_lower:
            logger.critical("Security safeguard triggered: Output directory contains face/iris assets.")
            raise ValueError("Extraction directory must not contain face or iris assets.")
        shutil.rmtree(args.output_dir)
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "gallery"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "probes"), exist_ok=True)

    # Prepare file list for Gallery and Probes
    file_list = []
    
    # Process Gallery mapping (100 clean baseline images)
    for i in range(1, 101):
        person_name = f"Person_{i:03d}"
        gallery_img_path = os.path.join(args.dataset_dir, "training", person_name, "fingerprint_right_thumb.jpg")
        save_path = os.path.join(args.output_dir, "gallery", f"{person_name}.npy")
        file_list.append((gallery_img_path, save_path))
        
    # Process Probes mapping (800 degraded variations from FVC2002)
    for i in range(1, 101):
        person_name = f"Person_{i:03d}"
        probe_person_dir = os.path.join(args.output_dir, "probes", person_name)
        os.makedirs(probe_person_dir, exist_ok=True)
        for j in range(1, 9):
            probe_img_path = os.path.join(args.fvc_dir, f"{i}_{j}.tif")
            save_path = os.path.join(probe_person_dir, f"variation_{j}.npy")
            file_list.append((probe_img_path, save_path))

    # Initialize DeepPrint_TexMinu model
    logger.info("Initializing DeepPrint_TexMinu model (8000 training subjects, 256 texture, 256 minutiae)")
    model = DeepPrint_TexMinu(num_fingerprints=8000, texture_embedding_dims=256, minutia_embedding_dims=256)
    
    # Load state dict
    if not os.path.exists(args.weights_path):
        logger.critical(f"Pretrained weights file not found at: {args.weights_path}")
        raise FileNotFoundError(f"Model checkpoint {args.weights_path} did not exist.")
        
    logger.info(f"Loading weights from: {args.weights_path}")
    checkpoint = torch.load(args.weights_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    logger.info("Pretrained model loaded successfully in eval mode.")

    # Initialize Gabor binarizer with standard FVC ridge width parameter
    binarizer = LazilyAllocatedBinarizer(1.8)

    # Initialize PyTorch dataset and dataloader
    dataset = ChimericFingerprintDataset(file_list, binarizer)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=False
    )

    logger.info("Starting embedding extraction...")
    start_time = time.time()
    
    with torch.no_grad():
        for batch_imgs, batch_save_paths in tqdm(dataloader, desc="Extracting embeddings"):
            # Workaround for single-sample squeeze bugs in the architecture forward pass
            if batch_imgs.shape[0] == 1:
                batch_imgs = torch.cat([batch_imgs, batch_imgs], dim=0)
                duplicated = True
            else:
                duplicated = False

            batch_imgs = batch_imgs.to(device)
            output = model(batch_imgs)
            
            tex_emb = output.texture_embeddings
            minu_emb = output.minutia_embeddings
            
            # Array Vectorization (Concatenate texture and minutiae into 512-dim vector)
            combined_emb = torch.cat([tex_emb, minu_emb], dim=1)
            
            # Hypersphere Projection (L2 Normalization)
            norm = torch.norm(combined_emb, p=2, dim=1, keepdim=True)
            normalized_emb = combined_emb / torch.clamp(norm, min=1e-12)
            
            if duplicated:
                normalized_emb = normalized_emb[0:1]
                
            embs_np = normalized_emb.cpu().numpy().astype(np.float32)
            
            # Save files
            for idx, save_path in enumerate(batch_save_paths):
                np.save(save_path, embs_np[idx])

    logger.info(f"Successfully completed fingerprint embedding extraction in {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()

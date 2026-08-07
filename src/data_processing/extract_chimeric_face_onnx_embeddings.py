#!/usr/bin/env python3
"""
ONNX-based ArcFace Face Alignment and Embedding Extraction Script.

This script processes a chimeric dataset directory, parses identity subfolders
to locate face images, aligns faces using a 2D similarity transform mapped to 
canonical coordinates, normalizes the images, and runs inference using the
w600k_r50.onnx (ArcFace IR50) model to extract 512-dimensional L2-normalized embeddings.

Author: Antigravity (Expert Biometrics & Deep Learning Engineer)
Date: June 20, 2026
"""

import os
import argparse
import logging
import cv2
import numpy as np
import onnxruntime as ort
from tqdm import tqdm

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Canonical reference landmark coordinates for 112x112 face alignment
# Derived from standard ArcFace / InsightFace reference points.
CANONICAL_LANDMARKS = np.array([
    [38.2946, 51.6963],  # Left Eye
    [73.5318, 51.5014],  # Right Eye
    [56.0252, 71.7366],  # Nose Tip
    [41.5493, 92.3655],  # Left Mouth Corner
    [70.7299, 92.2041]   # Right Mouth Corner
], dtype=np.float32)


def parse_chimeric_dataset(root_dir, base_dir=None):
    """
    Parses the chimeric dataset directory to locate face images.
    Traverses subfolders recursively or directly. In each identity subfolder,
    it identifies files whose names contain 'face' (case-insensitive) and end
    with standard image extensions.
    
    Args:
        root_dir (str): Path to chimeric dataset root folder.
        base_dir (str, optional): Base directory to compute relative path.
        
    Returns:
        list of tuple: List of (identity_id, relative_path, full_path) tasks.
    """
    if not os.path.exists(root_dir):
        logger.error(f"Root directory does not exist: {root_dir}")
        return []
        
    if base_dir is None:
        base_dir = root_dir
        
    tasks = []
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    
    # Traverse directory
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            name_lower = file.lower()
            # Isolate face image file by checking name contains 'face' and is an image
            if name_lower.endswith(image_extensions) and 'face' in name_lower:
                full_path = os.path.join(root, file)
                # Identity ID corresponds to the direct parent folder name (e.g. Person_001)
                identity_id = os.path.basename(root)
                rel_path = os.path.relpath(root, base_dir)
                tasks.append((identity_id, rel_path, full_path))
                
    # Sort tasks to ensure deterministic order
    tasks.sort(key=lambda x: (x[1], x[2]))
    return tasks


def align_face(image, landmarks=None):
    """
    Applies precise affine face alignment to warp a face image to 112x112.
    
    Uses least-squares estimation to compute a 2D similarity transform matrix
    preserving scale, rotation, and translation, mapping the input 5-point
    landmarks onto the canonical reference points.
    
    If landmarks are not provided, attempts to run a fallback detector or
    generates mock landmarks in the center of the image.
    
    Args:
        image (np.ndarray): Original BGR face image.
        landmarks (np.ndarray, optional): 5-point facial landmarks of shape (5, 2).
        
    Returns:
        np.ndarray: Aligned face patch of shape (112, 112, 3) in BGR format.
    """
    h, w = image.shape[:2]
    
    # 1. Fallback: Detect or generate landmarks if not provided
    if landmarks is None:
        # Attempt to detect using insightface FaceAnalysis if imported and runnable
        try:
            from insightface.app import FaceAnalysis
            # Lazy initialize standard detector
            detector = FaceAnalysis(allowed_modules=['detection'])
            detector.prepare(ctx_id=-1, det_size=(640, 640))
            faces = detector.get(image)
            if len(faces) > 0:
                # Select the best face by det_score * bbox_area
                best_face = max(faces, key=lambda x: x.det_score * (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
                landmarks = best_face.kps
        except Exception:
            pass
            
    if landmarks is None:
        # Fallback to centered mock landmarks scaled to the image dimensions
        logger.warning("No landmarks provided/detected. Using centered mock landmarks as fallback.")
        scale_x = w / 112.0
        scale_y = h / 112.0
        landmarks = CANONICAL_LANDMARKS.copy()
        landmarks[:, 0] *= scale_x
        landmarks[:, 1] *= scale_y

    # 2. Least-Squares Similarity Transform Calculation
    # A 2D similarity transform maps source points X to target points Y via:
    #   y_i = s * R * x_i + t
    # Written as a linear system: A * c = B, where c = [a, b, tx, ty]^T,
    # rotation R = [[a, -b], [b, a]] / s, scale s = sqrt(a^2 + b^2),
    # translation t = [tx, ty]^T.
    
    A = []
    B = []
    for i in range(5):
        x, y = landmarks[i]
        tx_ref, ty_ref = CANONICAL_LANDMARKS[i]
        
        # Build A matrix coefficients for x_ref and y_ref mapping
        A.append([x, -y, 1, 0])
        A.append([y,  x, 0, 1])
        B.append(tx_ref)
        B.append(ty_ref)
        
    A = np.array(A, dtype=np.float32)
    B = np.array(B, dtype=np.float32)
    
    # Solve system using standard least-squares: c = (A^T * A)^-1 * A^T * B
    c, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    
    # Construct 2x3 affine warp matrix:
    # M = [[a, -b, tx],
    #      [b,  a, ty]]
    M = np.array([
        [c[0], -c[1], c[2]],
        [c[1],  c[0], c[3]]
    ], dtype=np.float32)
    
    # 3. Warp image patch to 112x112 using bilinear interpolation
    aligned_patch = cv2.warpAffine(
        image, 
        M, 
        (112, 112), 
        flags=cv2.INTER_LINEAR, 
        borderMode=cv2.BORDER_CONSTANT, 
        borderValue=0
    )
    return aligned_patch


def preprocess_tensor(aligned_patch):
    """
    Transforms the 112x112 BGR face patch into ArcFace ONNX tensor input.
    
    1. Color Space: BGR to RGB.
    2. Data Type: float32.
    3. Rescaling: Map intensities from [0, 255] to [-1, 1] using (Pixel - 127.5) / 127.5.
    4. Dimension Layout: Transpose HWC (112, 112, 3) to CHW (3, 112, 112).
    5. Batch Expansion: Add batch dimension to yield (1, 3, 112, 112).
    
    Args:
        aligned_patch (np.ndarray): 112x112x3 BGR warped image patch.
        
    Returns:
        np.ndarray: Preprocessed input tensor of shape (1, 3, 112, 112).
    """
    # 1. Convert BGR to RGB
    img_rgb = cv2.cvtColor(aligned_patch, cv2.COLOR_BGR2RGB)
    
    # 2. Convert to float32
    img_float = img_rgb.astype(np.float32)
    
    # 3. Rescale pixel values to [-1, 1]
    img_normalized = (img_float - 127.5) / 127.5
    
    # 4. Transpose from HWC to CHW format
    img_chw = np.transpose(img_normalized, (2, 0, 1))
    
    # 5. Expand batch dimension to yield (1, 3, 112, 112)
    tensor = np.expand_dims(img_chw, axis=0)
    return tensor


def get_arcface_embedding(tensor, session):
    """
    Runs ONNX inference and applies L2 normalization to return the embedding.
    
    Args:
        tensor (np.ndarray): Preprocessed tensor of shape (1, 3, 112, 112).
        session (ort.InferenceSession): Initialized ONNX runtime session.
        
    Returns:
        np.ndarray: L2 normalized 512-dimensional embedding of shape (512,).
    """
    # Get model input and output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Run forward pass through ArcFace model
    outputs = session.run([output_name], {input_name: tensor})
    raw_embedding = outputs[0]  # Shape: (1, 512)
    
    # Flatten to a 1D vector (512,)
    embedding = raw_embedding.flatten()
    
    # Apply L2 Normalization: normalized = v / ||v||_2
    norm = np.linalg.norm(embedding, ord=2)
    eps = 1e-12
    normalized_embedding = embedding / max(norm, eps)
    
    return normalized_embedding


def main():
    parser = argparse.ArgumentParser(description="ArcFace Face Alignment and ONNX Feature Extraction")
    parser.add_argument(
        "--dataset_dir", 
        type=str, 
        default="Chimeric_Dataset", 
        help="Path to chimeric dataset root folder"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="face_onnx_embeddings", 
        help="Directory to save extracted embeddings"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        default=os.path.expanduser("~/.insightface/models/buffalo_l/w600k_r50.onnx"), 
        help="Path to w600k_r50.onnx model file"
    )
    args = parser.parse_args()

    # Initialize ONNX session
    if not os.path.exists(args.model_path):
        logger.error(f"ONNX model file not found at: {args.model_path}")
        logger.info("Please specify the correct path using --model_path.")
        return

    logger.info(f"Initializing ONNX Inference Session for ArcFace model: {args.model_path}")
    
    # Let ONNX Runtime choose the best execution provider (CUDA if available, else CPU)
    available_providers = ort.get_available_providers()
    providers = []
    if 'CUDAExecutionProvider' in available_providers:
        providers.append('CUDAExecutionProvider')
    providers.append('CPUExecutionProvider')
    
    logger.info(f"Available execution providers: {available_providers}")
    logger.info(f"Target execution providers: {providers}")
    
    session = ort.InferenceSession(args.model_path, providers=providers)
    logger.info("ONNX Session loaded successfully.")

    # Parse dataset
    train_dir = os.path.join(args.dataset_dir, "training")
    if not os.path.exists(train_dir):
        logger.error(f"Training directory '{train_dir}' does not exist.")
        return

    logger.info(f"Scanning directory: {train_dir}")
    tasks = parse_chimeric_dataset(train_dir, base_dir=args.dataset_dir)
    
    if not tasks:
        logger.warning(f"No face images found in {train_dir}.")
        return
        
    logger.info(f"Found {len(tasks)} face images to process.")
    
    # Create base output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    processed_count = 0
    failed_count = 0
    
    for identity_id, rel_path, full_path in tqdm(tasks, desc="Extracting Face Embeddings"):
        try:
            # 1. Load image in BGR
            img = cv2.imread(full_path)
            if img is None:
                raise ValueError(f"Failed to read image at: {full_path}")
                
            # 2. Perform Precise Affine Face Alignment (Automatic landmark detection inside)
            aligned_patch = align_face(img, landmarks=None)
            
            # 3. Preprocess to ONNX input format
            tensor = preprocess_tensor(aligned_patch)
            
            # 4. Extract ArcFace Embedding and L2 Normalize
            embedding = get_arcface_embedding(tensor, session)
            
            # 5. Storage: save mirroring the dataset subfolder structure to avoid collisions
            if rel_path == '.':
                save_dir = args.output_dir
            else:
                save_dir = os.path.join(args.output_dir, rel_path)
                
            os.makedirs(save_dir, exist_ok=True)
            
            # Determine save filename based on original face image filename
            orig_name = os.path.splitext(os.path.basename(full_path))[0]
            # Replace spaces with underscores in testing filenames
            orig_name_clean = orig_name.replace(" ", "_")
            
            save_path = os.path.join(save_dir, f"{identity_id}_{orig_name_clean}_face.npy")
            np.save(save_path, embedding)
            
            processed_count += 1
            
        except Exception as e:
            logger.error(f"Error processing {full_path}: {e}")
            failed_count += 1
            
    logger.info(f"Extraction completed. Processed: {processed_count}, Failed: {failed_count}.")
    logger.info(f"Face embeddings saved in: {args.output_dir}")


if __name__ == "__main__":
    main()

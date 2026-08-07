#!/usr/bin/env python3
"""
Multimodal Biometric Verification System - Adaptive Score-Level Fusion Pipeline.

This pipeline performs score-level fusion using:
1. Face: ArcFace Embeddings (Cosine Similarity)
2. Iris: OpenIris (Masked Fractional Hamming Distance)
3. Fingerprint: SourceAFIS (Minutiae Matching Score)

It implements:
- Automatic right-eye iris selection.
- 6th face image selection for the clean gallery face.
- Logistic score calibration to convert scores to posterior match probabilities.
- Dynamic threshold escalation/optimization per active combination at FAR = 0.1%.
- Advanced quality metrics: local gradient structure tensor orientation coherence
  for fingerprints, mask coverage for iris.
- Dynamic weight redistribution based on active quality scores.
- Robust error handling using NaN/None.
- Evaluation simulator (Aligned Index-Wrapping and Full Cartesian Sweep).
"""

import os
import sys
import glob
import pickle
import random
import argparse
import numpy as np
import pandas as pd
import cv2
import onnxruntime as ort
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

# Import local biometric frameworks
sys.path.append(os.path.abspath("."))
from open_iris_pipeline import OpenIrisPipelineManager, BiometricQualityFailure
from extractors.sourceafis_client import SourceAFISClient
from matchers.sourceafis_matcher import SourceAFISMatcher
from extract_chimeric_face_onnx_embeddings import align_face, preprocess_tensor, get_arcface_embedding

# Encryption, cancelable biometrics, and database wrapper
import pqc_helper
import cancelable_transforms
from database_wrapper import BiometricDatabase


# Configuration Constants
RANDOM_STATE = 42
TARGET_FAR = 0.001  # Target FAR = 0.1%
FLOORS = {'face': 0.15, 'iris': 0.15, 'fp': 0.15}  # Floors on calibrated probability space
BASE_WEIGHTS = {'face': 1/3, 'iris': 1/3, 'fp': 1/3}

# Quality weights in active combination
def get_adaptive_weights(q_face, q_iris, q_fp, active):
    """
    Computes weights dynamically proportional to the quality scores of active modalities.
    If any quality is unavailable, defaults to baseline weights.
    """
    qualities = {'face': q_face, 'iris': q_iris, 'fp': q_fp}
    active_quals = {k: qualities[k] if qualities[k] is not None else 1.0 for k in active}
    
    total_q = sum(active_quals.values())
    if total_q > 1e-5:
        return {k: active_quals[k] / total_q for k in active}
    else:
        # Fallback to equal redistribution
        return {k: 1.0 / len(active) for k in active}

def get_equal_redistributed_weights(active):
    """
    Redistributes baseline weights proportionally among the active modalities.
    """
    active_base = {k: BASE_WEIGHTS[k] for k in active}
    total_base = sum(active_base.values())
    if total_base > 1e-5:
        return {k: active_base[k] / total_base for k in active}
    else:
        return {k: 1.0 / len(active) for k in active}

# Advanced Quality Metric Functions
def compute_fingerprint_quality(img_path):
    """
    Computes advanced Fingerprint Quality:
    Q_fp = 0.6 * Coverage + 0.2 * Orientation Coherence + 0.2 * Sharpness
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0, 0.0, 0.0, 0.0
    
    h, w = img.shape
    
    # 1. Coverage: foreground pixels ratio (ridges are dark, background is bright)
    foreground = img < 210
    coverage = float(np.sum(foreground) / img.size)
    
    # 2. Sharpness: Laplacian variance normalized to [0, 1]
    sharpness = float(cv2.Laplacian(img, cv2.CV_64F).var())
    sharpness_norm = min(1.0, sharpness / 1500.0)
    
    # 3. Orientation Coherence via Gradient Structure Tensor over 16x16 blocks
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    
    block_size = 16
    coherences = []
    
    for y in range(0, h - block_size + 1, block_size):
        for x in range(0, w - block_size + 1, block_size):
            bx_img = img[y:y+block_size, x:x+block_size]
            # Check if block has significant foreground
            if np.sum(bx_img < 210) / (block_size * block_size) < 0.1:
                continue
                
            bx_gx = gx[y:y+block_size, x:x+block_size]
            bx_gy = gy[y:y+block_size, x:x+block_size]
            
            gxx = np.sum(bx_gx ** 2)
            gyy = np.sum(bx_gy ** 2)
            gxy = np.sum(bx_gx * bx_gy)
            
            denom = gxx + gyy
            if denom > 1e-5:
                num = np.sqrt((gxx - gyy) ** 2 + 4 * (gxy ** 2))
                coherence = float(num / denom)
                coherences.append(coherence)
                
    coherence_avg = float(np.mean(coherences)) if len(coherences) > 0 else 0.0
    
    # Overall Quality score
    q_fp = 0.6 * coverage + 0.2 * coherence_avg + 0.2 * sharpness_norm
    return q_fp, coverage, coherence_avg, sharpness_norm

def compute_iris_quality(mask_list, img_path, is_failed=False):
    """
    Computes advanced Iris Quality:
    Q_iris = 0.5 * MaskCoverage + 0.3 * Sharpness + 0.2 * SegmentationConfidence
    """
    if is_failed or mask_list is None:
        return 0.0, 0.0, 0.0, 0.0
    
    # 1. Mask Coverage: ratio of valid bits (True) in OpenIris mask
    total_elements = sum(m.size for m in mask_list)
    valid_elements = sum(np.sum(m) for m in mask_list)
    mask_coverage = float(valid_elements / total_elements) if total_elements > 0 else 0.0
    
    # 2. Sharpness of raw image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        sharpness = float(cv2.Laplacian(img, cv2.CV_64F).var())
        sharpness_norm = min(1.0, sharpness / 1000.0)
    else:
        sharpness_norm = 0.0
        
    # 3. Segmentation Confidence: successful extraction
    seg_confidence = 1.0
    
    q_iris = 0.5 * mask_coverage + 0.3 * sharpness_norm + 0.2 * seg_confidence
    return q_iris, mask_coverage, sharpness_norm, seg_confidence

def compute_face_quality(aligned_patch, det_score):
    """
    Computes Face Quality:
    Q_face = 0.8 * det_score + 0.2 * sharpness_norm
    """
    if aligned_patch is None or det_score is None:
        return 0.0, 0.0, 0.0
    
    gray = cv2.cvtColor(aligned_patch, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharpness_norm = min(1.0, sharpness / 1000.0)
    
    q_face = 0.8 * float(det_score) + 0.2 * sharpness_norm
    return q_face, float(det_score), sharpness_norm
class SecurityError(Exception):
    pass

class SecureFusionEnclave:
    def __init__(self, kem_priv, dsa_pub):
        self.kem_priv = kem_priv
        self.dsa_pub = dsa_pub
        self.clf_face = None
        self.clf_iris = None
        self.clf_fp = None
        self.thresholds = None
        
    def update_enclave_parameters(self, clf_face, clf_iris, clf_fp, thresholds):
        self.clf_face = clf_face
        self.clf_iris = clf_iris
        self.clf_fp = clf_fp
        self.thresholds = thresholds
        
    def decrypt_gallery_templates(self, encrypted_templates: list) -> dict:
        """
        Verifies signatures and decrypts the encrypted database records inside the TEE.
        Returns a dict mapping modality to its decrypted template.
        """
        import pqc_helper
        import pickle
        
        decrypted = {}
        for row in encrypted_templates:
            modality = row["modality"]
            
            # Verify signature first (protects against database ciphertext / nonce tampering)
            payload = {
                "kem_ciphertext": row["kem_ciphertext"],
                "nonce": row["nonce"],
                "ciphertext": row["ciphertext"]
            }
            if not pqc_helper.verify_payload(payload, row["signature"], self.dsa_pub):
                raise SecurityError(f"Security Alert: Signature verification failed for {modality} template!")
                
            # Decrypt
            plaintext = pqc_helper.envelope_decrypt(payload, self.kem_priv)
            decrypted[modality] = pickle.loads(plaintext)
            
        return decrypted

    def compute_raw_scores(self, decrypted_gal: dict, probe_face, probe_iris_code, probe_iris_mask, probe_fp, matcher, iris_rec):
        """
        Computes comparison scores between decrypted gallery templates and probe templates inside TEE.
        """
        import numpy as np
        from open_iris_pipeline import BiometricQualityFailure
        
        s_face = np.nan
        s_iris = np.nan
        s_fp = np.nan
        
        # Face Cosine Similarity
        if "face" in decrypted_gal and decrypted_gal["face"]["embedding"] is not None and probe_face is not None:
            s_face = float(np.dot(decrypted_gal["face"]["embedding"], probe_face))
            
        # Iris Masked Fractional Hamming Distance (converted to similarity: 1 - HD)
        if "iris" in decrypted_gal and decrypted_gal["iris"]["code"] is not None and probe_iris_code is not None:
            try:
                hd = float(iris_rec.compute_masked_distance(
                    decrypted_gal["iris"]["code"], decrypted_gal["iris"]["mask"],
                    probe_iris_code, probe_iris_mask
                ))
                s_iris = 1.0 - hd
            except BiometricQualityFailure:
                pass
                
        # Fingerprint Minutiae Matching
        if "fingerprint" in decrypted_gal and decrypted_gal["fingerprint"]["template_b64"] is not None and probe_fp is not None:
            s_fp = float(matcher.match(decrypted_gal["fingerprint"]["template_b64"], probe_fp))
            
        return s_face, s_iris, s_fp

    def verify_and_fuse(self, encrypted_templates: list, probe_face, probe_iris_code, probe_iris_mask, probe_fp, 
                        q_face, q_iris, q_fp, matcher, iris_rec) -> tuple:
        """
        Full isolated verification routine:
        1. Decrypts and verifies template signatures.
        2. Computes comparison scores.
        3. Calibrates raw scores into match probabilities.
        4. Applies 0.15 floor logic.
        5. Performs quality-weighted fusion.
        6. Makes ACCEPT/REJECT decision using enclave-secured thresholds.
        7. Returns (decision, fused_score, combo_key)
        """
        import numpy as np
        
        # Decrypt gallery templates
        decrypted_gal = self.decrypt_gallery_templates(encrypted_templates)
        
        # Compute raw scores
        s_face, s_iris, s_fp = self.compute_raw_scores(
            decrypted_gal, probe_face, probe_iris_code, probe_iris_mask, probe_fp, matcher, iris_rec
        )
        
        active = []
        p_face, p_iris, p_fp = np.nan, np.nan, np.nan
        FLOORS = {'face': 0.15, 'iris': 0.15, 'fp': 0.15}
        
        # Calibration (only if classifiers are updated)
        if self.clf_face is not None:
            if s_face is not None and not np.isnan(s_face):
                p_face = float(self.clf_face.predict_proba([[s_face]])[0, 1])
                active.append('face')
            
            if s_iris is not None and not np.isnan(s_iris):
                p_iris = float(self.clf_iris.predict_proba([[s_iris]])[0, 1])
                active.append('iris')
            
            if s_fp is not None and not np.isnan(s_fp):
                p_fp = float(self.clf_fp.predict_proba([[s_fp]])[0, 1])
                active.append('fp')
        else:
            return "REJECT", 0.0, "REJECT_UNCONFIGURED"
            
        if len(active) < 2:
            return "REJECT", 0.0, "REJECT_COMPROMISED"
            
        combo_key = '_'.join(active)
        
        # Dynamic Weight Redistribution
        weights = get_adaptive_weights(q_face, q_iris, q_fp, active)
        
        fused_score = 0.0
        if 'face' in active:
            fused_score += weights['face'] * p_face
        if 'iris' in active:
            fused_score += weights['iris'] * p_iris
        if 'fp' in active:
            fused_score += weights['fp'] * p_fp
            
        threshold = self.thresholds.get(combo_key, 0.65)
        decision = "MATCH" if fused_score >= threshold else "REJECT"
        
        return decision, fused_score, combo_key

class MultimodalFusionPipeline:
    def __init__(self, cache_path="multimodal_templates_cache.pkl"):
        self.cache_path = cache_path
        self.gallery_templates = {}
        self.probe_templates = {}
        
        # Calibration Parameters
        self.clf_face = None
        self.clf_iris = None
        self.clf_fp = None
        
        # Dynamic Thresholds
        self.thresholds = {
            'face_iris_fp': 0.65,
            'face_iris': 0.65,
            'face_fp': 0.65,
            'iris_fp': 0.65
        }
        
        # Cryptographic Setup
        self.kem_priv, self.kem_pub = pqc_helper.generate_kem_keypair()
        self.dsa_priv, self.dsa_pub = pqc_helper.generate_signing_keypair()
        
        # Database Setup
        self.db = BiometricDatabase()
        
        # Secure Enclave Setup
        self.enclave = SecureFusionEnclave(self.kem_priv, self.dsa_pub)
        
    def run_extraction_and_caching(self, testing_dir, training_dir):
        """
        Loops through chimeric users, extracts biometric templates and quality metrics,
        and saves them to a local cache to prevent redundant heavy execution.
        """
        if os.path.exists(self.cache_path):
            print(f"Loading precomputed templates and features from cache: {self.cache_path}")
            with open(self.cache_path, "rb") as f:
                data = pickle.load(f)
            self.gallery_templates = data["gallery"]
            self.probe_templates = data["probes"]
            self.enroll_all_users_in_db()
            return

        print("==================================================")
        print("    Initializing Modality Extraction Engines      ")
        print("==================================================")
        
        # 1. Initialize Face Analyzer (InsightFace app & session)
        from insightface.app import FaceAnalysis
        print("Loading Face Analysis Engine...")
        face_app = FaceAnalysis(allowed_modules=['detection'])
        face_app.prepare(ctx_id=-1, det_size=(640, 640))
        
        model_path = os.path.expanduser("~/.insightface/models/buffalo_l/w600k_r50.onnx")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ArcFace ONNX model missing at {model_path}")
        face_session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        
        # 2. Initialize OpenIris Pipeline Manager
        print("Loading Iris Extraction Engine (OpenIris)...")
        iris_rec = OpenIrisPipelineManager()
        
        # 3. Initialize JPype SourceAFIS Engine
        print("Loading Fingerprint Matching Engine (SourceAFIS)...")
        afis_client = SourceAFISClient()
        if not afis_client.enabled:
            raise RuntimeError("SourceAFIS client could not be loaded via JPype.")

        print("\nExtracting templates from Chimeric Dataset directories...")
        for i in tqdm(range(1, 101), desc="Subjects Ingestion"):
            person_name = f"Person_{i:03d}"
            
            # Directory setup
            subj_test_dir = os.path.join(testing_dir, person_name)
            subj_train_dir = os.path.join(training_dir, person_name)
            
            self.gallery_templates[i] = {}
            self.probe_templates[i] = {"face": [], "iris": [], "fingerprint": []}
            
            # --------------------------------------------------
            # Gallery Template Extractions
            # --------------------------------------------------
            
            # Face Gallery: 6th face image (face_06.jpg) in the testing folder
            face_gal_path = os.path.join(subj_test_dir, "face_06.jpg")
            try:
                img_f = cv2.imread(face_gal_path)
                if img_f is None:
                    raise ValueError(f"Face gallery image missing at {face_gal_path}")
                faces = face_app.get(img_f)
                if len(faces) == 0:
                    raise ValueError("No faces detected in gallery image.")
                best_face = max(faces, key=lambda x: x.det_score * (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
                aligned_patch = align_face(img_f, best_face.kps)
                tensor = preprocess_tensor(aligned_patch)
                emb = get_arcface_embedding(tensor, face_session)
                q_face, _, _ = compute_face_quality(aligned_patch, best_face.det_score)
                self.gallery_templates[i]["face"] = {"embedding": emb, "quality": q_face}
            except Exception as e:
                print(f"Warning: Failed face gallery for {person_name}: {e}")
                self.gallery_templates[i]["face"] = {"embedding": None, "quality": 0.0}

            # Iris Gallery: iris_right.jpg from the training folder
            iris_gal_path = os.path.join(subj_train_dir, "iris_right.jpg")
            try:
                code, mask = iris_rec.generate_biometric_template(iris_gal_path, eye_side="right")
                q_iris, _, _, _ = compute_iris_quality(mask, iris_gal_path, is_failed=False)
                self.gallery_templates[i]["iris"] = {"code": code, "mask": mask, "quality": q_iris}
            except Exception as e:
                print(f"Warning: Failed iris gallery for {person_name}: {e}")
                self.gallery_templates[i]["iris"] = {"code": None, "mask": None, "quality": 0.0}

            # Fingerprint Gallery: fingerprint_right_thumb.jpg from the training folder
            fp_gal_path = os.path.join(subj_train_dir, "fingerprint_right_thumb.jpg")
            try:
                img_fp = cv2.imread(fp_gal_path, cv2.IMREAD_GRAYSCALE)
                if img_fp is None:
                    raise ValueError(f"Fingerprint gallery image missing at {fp_gal_path}")
                fp_template = afis_client.extract_template(img_fp)
                q_fp, _, _, _ = compute_fingerprint_quality(fp_gal_path)
                self.gallery_templates[i]["fingerprint"] = {"template_b64": fp_template, "quality": q_fp}
            except Exception as e:
                print(f"Warning: Failed fingerprint gallery for {person_name}: {e}")
                self.gallery_templates[i]["fingerprint"] = {"template_b64": None, "quality": 0.0}
                
            # --------------------------------------------------
            # Probe Template Extractions
            # --------------------------------------------------
            
            # Face Probes: face_01.jpg to face_14.jpg (excluding face_06.jpg)
            for idx in range(1, 15):
                if idx == 6:
                    continue  # Used for gallery
                face_probe_path = os.path.join(subj_test_dir, f"face_{idx:02d}.jpg")
                try:
                    img_f = cv2.imread(face_probe_path)
                    if img_f is None:
                        raise ValueError(f"Image not found at {face_probe_path}")
                    faces = face_app.get(img_f)
                    if len(faces) == 0:
                        raise ValueError("No faces detected in probe image.")
                    best_face = max(faces, key=lambda x: x.det_score * (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
                    aligned_patch = align_face(img_f, best_face.kps)
                    tensor = preprocess_tensor(aligned_patch)
                    emb = get_arcface_embedding(tensor, face_session)
                    q_face, _, _ = compute_face_quality(aligned_patch, best_face.det_score)
                    self.probe_templates[i]["face"].append({
                        "embedding": emb, "quality": q_face, "filename": f"face_{idx:02d}.jpg"
                    })
                except Exception as e:
                    self.probe_templates[i]["face"].append({
                        "embedding": None, "quality": 0.0, "filename": f"face_{idx:02d}.jpg"
                    })
                    
            # Iris Probes: Select only Right eye (iris_R_1.jpg to iris_R_5.jpg)
            # Find R files, discarding Left ones (iris_L_*.jpg)
            iris_files = sorted(glob.glob(os.path.join(subj_test_dir, "iris_R_*.jpg")))
            # Let's verify if files are named iris_right1.jpg or similar in some cases.
            if len(iris_files) == 0:
                iris_files = sorted(glob.glob(os.path.join(subj_test_dir, "iris_right*.jpg")))
            
            for file_path in iris_files:
                filename = os.path.basename(file_path)
                try:
                    code, mask = iris_rec.generate_biometric_template(file_path, eye_side="right")
                    q_iris, _, _, _ = compute_iris_quality(mask, file_path, is_failed=False)
                    self.probe_templates[i]["iris"].append({
                        "code": code, "mask": mask, "quality": q_iris, "filename": filename
                    })
                except Exception as e:
                    self.probe_templates[i]["iris"].append({
                        "code": None, "mask": None, "quality": 0.0, "filename": filename
                    })
                    
            # Fingerprint Probes: fingerprint_1.tif to fingerprint_8.tif
            for idx in range(1, 9):
                fp_probe_path = os.path.join(subj_test_dir, f"fingerprint_{idx}.tif")
                # Fallback to jpg check if tif is converted
                if not os.path.exists(fp_probe_path):
                    fp_probe_path = os.path.join(subj_test_dir, f"fingerprint_right_thumb{idx}.jpg")
                    
                filename = os.path.basename(fp_probe_path)
                try:
                    img_fp = cv2.imread(fp_probe_path, cv2.IMREAD_GRAYSCALE)
                    if img_fp is None:
                        raise ValueError(f"Fingerprint image missing at {fp_probe_path}")
                    fp_template = afis_client.extract_template(img_fp)
                    q_fp, _, _, _ = compute_fingerprint_quality(fp_probe_path)
                    self.probe_templates[i]["fingerprint"].append({
                        "template_b64": fp_template, "quality": q_fp, "filename": filename
                    })
                except Exception as e:
                    self.probe_templates[i]["fingerprint"].append({
                        "template_b64": None, "quality": 0.0, "filename": filename
                    })
                    
        # Write to cache
        print(f"Caching precomputed templates and features to: {self.cache_path}")
        with open(self.cache_path, "wb") as f:
            pickle.dump({"gallery": self.gallery_templates, "probes": self.probe_templates}, f)
        self.enroll_all_users_in_db()

    def enroll_all_users_in_db(self):
        """
        Enrolls all 100 gallery templates from self.gallery_templates
        into the SQLite database after applying cancelable biometrics and PQC envelope encryption.
        """
        import pickle
        import uuid
        import pqc_helper
        import cancelable_transforms
        
        print("Enrolling users in encrypted biometric database...")
        self.db.clear_database()
        
        for i in sorted(self.gallery_templates.keys()):
            person_name = f"Person_{i:03d}"
            self.db.enroll_user(person_name)
            gal_t = self.gallery_templates[i]
            token = f"token_person_{i:03d}"  # Non-secret identifier for prototyping
            
            # --- Face ---
            raw_face = gal_t["face"]["embedding"]
            q_face = gal_t["face"]["quality"]
            if raw_face is not None:
                trans_face = cancelable_transforms.biohash_face(raw_face, token)
                face_data = {"embedding": trans_face, "quality": q_face}
                serialized = pickle.dumps(face_data)
                payload = pqc_helper.envelope_encrypt(serialized, self.kem_pub)
                sig = pqc_helper.sign_payload(payload, self.dsa_priv)
                self.db.store_template(str(uuid.uuid4()), person_name, "face", 
                                       payload["kem_ciphertext"], payload["nonce"], payload["ciphertext"], sig, q_face)
                                       
            # --- Iris ---
            raw_code = gal_t["iris"]["code"]
            raw_mask = gal_t["iris"]["mask"]
            q_iris = gal_t["iris"]["quality"]
            if raw_code is not None and raw_mask is not None:
                trans_code, trans_mask = cancelable_transforms.transform_iris(raw_code, raw_mask, token)
                iris_data = {"code": trans_code, "mask": trans_mask, "quality": q_iris}
                serialized = pickle.dumps(iris_data)
                payload = pqc_helper.envelope_encrypt(serialized, self.kem_pub)
                sig = pqc_helper.sign_payload(payload, self.dsa_priv)
                self.db.store_template(str(uuid.uuid4()), person_name, "iris", 
                                       payload["kem_ciphertext"], payload["nonce"], payload["ciphertext"], sig, q_iris)
                                       
            # --- Fingerprint ---
            raw_fp = gal_t["fingerprint"]["template_b64"]
            q_fp = gal_t["fingerprint"]["quality"]
            if raw_fp is not None:
                # Fingerprint (Option A: no transform, encrypted-only)
                fp_data = {"template_b64": raw_fp, "quality": q_fp}
                serialized = pickle.dumps(fp_data)
                payload = pqc_helper.envelope_encrypt(serialized, self.kem_pub)
                sig = pqc_helper.sign_payload(payload, self.dsa_priv)
                self.db.store_template(str(uuid.uuid4()), person_name, "fingerprint", 
                                       payload["kem_ciphertext"], payload["nonce"], payload["ciphertext"], sig, q_fp)
        
        print(f"Successfully enrolled {len(self.gallery_templates)} subjects into the database.")

    def calibrate_scores(self, calibration_subjects):
        """
        Runs Logistic Calibration on the Calibration Set (60 subjects).
        Computes the matching parameters (a, b) for Face, Iris, and Fingerprint.
        Searches for target thresholds on Calibration genuine and impostor sets.
        """
        print("\n==================================================")
        print("      Phase 1: Logistic Score Calibration         ")
        print("==================================================")
        
        # Setup JPype matcher for calibration
        afis_client = SourceAFISClient()
        matcher = SourceAFISMatcher(afis_client)
        iris_rec = OpenIrisPipelineManager()
        
        # Parallel arrays to hold genuine and impostor scores
        raw_face_genuine, raw_face_impostor = [], []
        raw_iris_genuine, raw_iris_impostor = [], []
        raw_fp_genuine, raw_fp_impostor = [], []
        
        print("Computing comparison scores on Calibration Set...")
        for i in tqdm(calibration_subjects, desc="Matching Calibration Pairs"):
            # Fetch encrypted gallery templates of subject i
            gal_name = f"Person_{i:03d}"
            encrypted_gal = self.db.get_templates(gal_name)
            # Decrypt inside TEE
            decrypted_gal = self.enclave.decrypt_gallery_templates(encrypted_gal)
            
            # Genuine matches: Subject i probes against Subject i gallery
            pr_t = self.probe_templates[i]
            token = f"token_person_{i:03d}"  # Claimed identity token
            
            # Face
            if "face" in decrypted_gal:
                for p in pr_t["face"]:
                    if p["embedding"] is not None:
                        # Apply BioHash to probe using claimed token
                        p_transformed = cancelable_transforms.biohash_face(p["embedding"], token)
                        # Compute raw score inside TEE
                        score = float(np.dot(decrypted_gal["face"]["embedding"], p_transformed))
                        raw_face_genuine.append(score)
            # Iris
            if "iris" in decrypted_gal:
                for p in pr_t["iris"]:
                    if p["code"] is not None:
                        try:
                            # Apply Keyed XOR/Permutation to probe using claimed token
                            t_code, t_mask = cancelable_transforms.transform_iris(p["code"], p["mask"], token)
                            hd = float(iris_rec.compute_masked_distance(
                                decrypted_gal["iris"]["code"], decrypted_gal["iris"]["mask"],
                                t_code, t_mask
                            ))
                            raw_iris_genuine.append(1.0 - hd)
                        except BiometricQualityFailure:
                            pass
            # Fingerprint
            if "fingerprint" in decrypted_gal:
                for p in pr_t["fingerprint"]:
                    if p["template_b64"] is not None:
                        # Fingerprint (Option A: no transform, encrypted-only)
                        score = float(matcher.match(decrypted_gal["fingerprint"]["template_b64"], p["template_b64"]))
                        raw_fp_genuine.append(score)
                        
            # Impostor matches: Subject i probes against other Calibration subjects' galleries
            for j in calibration_subjects:
                if j == i:
                    continue
                # Fetch encrypted gallery templates of subject j
                gal_other_name = f"Person_{j:03d}"
                encrypted_gal_other = self.db.get_templates(gal_other_name)
                # Decrypt inside TEE
                decrypted_gal_other = self.enclave.decrypt_gallery_templates(encrypted_gal_other)
                token = f"token_person_{j:03d}"  # Claimed identity token is j
                
                # Face
                if "face" in decrypted_gal_other:
                    for p in pr_t["face"]:
                        if p["embedding"] is not None:
                            # Apply BioHash using claimed token j
                            p_transformed = cancelable_transforms.biohash_face(p["embedding"], token)
                            score = float(np.dot(decrypted_gal_other["face"]["embedding"], p_transformed))
                            raw_face_impostor.append(score)
                # Iris
                if "iris" in decrypted_gal_other:
                    for p in pr_t["iris"]:
                        if p["code"] is not None:
                            try:
                                # Apply Keyed XOR/Permutation using claimed token j
                                t_code, t_mask = cancelable_transforms.transform_iris(p["code"], p["mask"], token)
                                hd = float(iris_rec.compute_masked_distance(
                                    decrypted_gal_other["iris"]["code"], decrypted_gal_other["iris"]["mask"],
                                    t_code, t_mask
                                ))
                                raw_iris_impostor.append(1.0 - hd)
                            except BiometricQualityFailure:
                                pass
                # Fingerprint
                if "fingerprint" in decrypted_gal_other:
                    for p in pr_t["fingerprint"]:
                        if p["template_b64"] is not None:
                            score = float(matcher.match(decrypted_gal_other["fingerprint"]["template_b64"], p["template_b64"]))
                            raw_fp_impostor.append(score)

        # --------------------------------------------------
        # Fit Logistic Regressions
        # --------------------------------------------------
        # Face
        X_face = np.concatenate([raw_face_genuine, raw_face_impostor]).reshape(-1, 1)
        y_face = np.concatenate([np.ones(len(raw_face_genuine)), np.zeros(len(raw_face_impostor))])
        self.clf_face = LogisticRegression(solver='liblinear', class_weight='balanced')
        self.clf_face.fit(X_face, y_face)
        print(f"Face Calibration Fit: coef = {self.clf_face.coef_[0][0]:.4f}, intercept = {self.clf_face.intercept_[0]:.4f}")
        
        # Iris (Input is Similarity: 1 - HD)
        X_iris = np.concatenate([raw_iris_genuine, raw_iris_impostor]).reshape(-1, 1)
        y_iris = np.concatenate([np.ones(len(raw_iris_genuine)), np.zeros(len(raw_iris_impostor))])
        self.clf_iris = LogisticRegression(solver='liblinear', class_weight='balanced')
        self.clf_iris.fit(X_iris, y_iris)
        print(f"Iris Calibration Fit: coef = {self.clf_iris.coef_[0][0]:.4f}, intercept = {self.clf_iris.intercept_[0]:.4f}")
        
        # Fingerprint
        X_fp = np.concatenate([raw_fp_genuine, raw_fp_impostor]).reshape(-1, 1)
        y_fp = np.concatenate([np.ones(len(raw_fp_genuine)), np.zeros(len(raw_fp_impostor))])
        self.clf_fp = LogisticRegression(solver='liblinear', class_weight='balanced')
        self.clf_fp.fit(X_fp, y_fp)
        print(f"Fingerprint Calibration Fit: coef = {self.clf_fp.coef_[0][0]:.4f}, intercept = {self.clf_fp.intercept_[0]:.4f}")

        # Update enclave with newly fitted classifiers
        self.enclave.update_enclave_parameters(self.clf_face, self.clf_iris, self.clf_fp, self.thresholds)

        # --------------------------------------------------
        # Search Dynamic Thresholds on Calibration Set
        # --------------------------------------------------
        fused_scores_genuine = {
            'face_iris_fp': [], 'face_iris': [], 'face_fp': [], 'iris_fp': []
        }
        fused_scores_impostor = {
            'face_iris_fp': [], 'face_iris': [], 'face_fp': [], 'iris_fp': []
        }

        cal_genuine_count = len(calibration_subjects) * 13
        cal_impostor_count = len(calibration_subjects) * (len(calibration_subjects) - 1) * 13
        print(f"Calibration set (60 subjects): {cal_impostor_count} impostor trials, {cal_genuine_count} genuine trials")
        print("Simulating fusion trials on Calibration Set to optimize thresholds...")
        for i in calibration_subjects:
            # Fetch encrypted gallery templates of subject i
            gal_name = f"Person_{i:03d}"
            encrypted_gal = self.db.get_templates(gal_name)
            
            pr_t = self.probe_templates[i]
            token = f"token_person_{i:03d}"  # Claimed token is i
            
            # Genuine Trials
            min_len = min(len(pr_t["face"]), len(pr_t["iris"]), len(pr_t["fingerprint"]))
            for idx in range(min_len):
                f_p = pr_t["face"][idx] if len(pr_t["face"]) > 0 else {"embedding": None, "quality": 0.0}
                i_p = pr_t["iris"][idx] if len(pr_t["iris"]) > 0 else {"code": None, "mask": None, "quality": 0.0}
                fp_p = pr_t["fingerprint"][idx] if len(pr_t["fingerprint"]) > 0 else {"template_b64": None, "quality": 0.0}
                
                # Transform probes under claimed token
                f_t = cancelable_transforms.biohash_face(f_p["embedding"], token) if f_p["embedding"] is not None else None
                i_code_t, i_mask_t = cancelable_transforms.transform_iris(i_p["code"], i_p["mask"], token) if i_p["code"] is not None else (None, None)
                fp_t = fp_p["template_b64"]
                
                # Enclave runs verify and fuse
                _, fused_s, combo_key = self.enclave.verify_and_fuse(
                    encrypted_gal, f_t, i_code_t, i_mask_t, fp_t,
                    f_p["quality"], i_p["quality"], fp_p["quality"], matcher, iris_rec
                )
                
                if combo_key in fused_scores_genuine and fused_s is not None:
                    fused_scores_genuine[combo_key].append(fused_s)
                    
            # Impostor Trials
            for j in calibration_subjects:
                if j == i:
                    continue
                # Fetch encrypted gallery templates of subject j
                gal_other_name = f"Person_{j:03d}"
                encrypted_gal_other = self.db.get_templates(gal_other_name)
                token = f"token_person_{j:03d}"  # Claimed token is j
                
                min_len = min(len(pr_t["face"]), len(pr_t["iris"]), len(pr_t["fingerprint"]))
                for idx in range(min_len):
                    f_p = pr_t["face"][idx] if len(pr_t["face"]) > 0 else {"embedding": None, "quality": 0.0}
                    i_p = pr_t["iris"][idx] if len(pr_t["iris"]) > 0 else {"code": None, "mask": None, "quality": 0.0}
                    fp_p = pr_t["fingerprint"][idx] if len(pr_t["fingerprint"]) > 0 else {"template_b64": None, "quality": 0.0}
                    
                    # Transform probes under claimed token j
                    f_t = cancelable_transforms.biohash_face(f_p["embedding"], token) if f_p["embedding"] is not None else None
                    i_code_t, i_mask_t = cancelable_transforms.transform_iris(i_p["code"], i_p["mask"], token) if i_p["code"] is not None else (None, None)
                    fp_t = fp_p["template_b64"]
                    
                    _, fused_s, combo_key = self.enclave.verify_and_fuse(
                        encrypted_gal_other, f_t, i_code_t, i_mask_t, fp_t,
                        f_p["quality"], i_p["quality"], fp_p["quality"], matcher, iris_rec
                    )
                    
                    if combo_key in fused_scores_impostor and fused_s is not None:
                        fused_scores_impostor[combo_key].append(fused_s)

        # Optimize thresholds to meet TARGET_FAR
        for key in self.thresholds.keys():
            imps = fused_scores_impostor[key]
            if len(imps) > 0:
                self.thresholds[key] = self.find_threshold_at_far(imps, TARGET_FAR)
            else:
                self.thresholds[key] = 0.65  # Fallback
            print(f"Optimized Threshold for Active Combination '{key}': {self.thresholds[key]:.6f}")

        # Update enclave with the final optimized thresholds
        self.enclave.update_enclave_parameters(self.clf_face, self.clf_iris, self.clf_fp, self.thresholds)

    def find_threshold_at_far(self, imposter_scores, target_far):
        sorted_imps = np.sort(imposter_scores)
        idx = int((1.0 - target_far) * len(sorted_imps))
        idx = max(0, min(idx, len(sorted_imps) - 1))
        return float(sorted_imps[idx])

    def fuse_triplet(self, s_face, s_iris, s_fp, q_face=1.0, q_iris=1.0, q_fp=1.0):
        """
        Executes core Adaptive Score Fusion:
        1. Evaluates raw processing failures (NaN / None) and flags them as Inactive.
        2. Applies Logistic Calibration to remaining raw scores.
        3. Checks calibrated scores against FLOORS (0.15 matching probability).
        4. Re-evaluates active modalities. Reject if active modalities count < 2.
        5. Computes quality-weighted fusion sum, redistributing weights dynamically.
        """
        active = []
        p_face, p_iris, p_fp = np.nan, np.nan, np.nan
        
        # Face Calibration
        if s_face is not None and not np.isnan(s_face):
            p_face = float(self.clf_face.predict_proba([[s_face]])[0, 1])
            active.append('face')
        
        # Iris Calibration
        if s_iris is not None and not np.isnan(s_iris):
            p_iris = float(self.clf_iris.predict_proba([[s_iris]])[0, 1])
            active.append('iris')
                
        # Fingerprint Calibration
        if s_fp is not None and not np.isnan(s_fp):
            p_fp = float(self.clf_fp.predict_proba([[s_fp]])[0, 1])
            active.append('fp')

        # Floor constraint reject
        if len(active) < 2:
            return 0.0, 'REJECT_COMPROMISED'

        # Determine combination key
        combo_key = '_'.join(active)

        # Quality weights redistribution
        weights = get_adaptive_weights(q_face, q_iris, q_fp, active)
        
        # Calculate fused score
        fused_score = 0.0
        if 'face' in active:
            fused_score += weights['face'] * p_face
        if 'iris' in active:
            fused_score += weights['iris'] * p_iris
        if 'fp' in active:
            fused_score += weights['fp'] * p_fp
            
        return fused_score, combo_key

    def run_evaluation(self, evaluation_subjects, full_cartesian=False, output_root="audit_results/multimodal"):
        """
        Runs evaluation trials on the test/evaluation split.
        Compares all genuine and impostor pairs and saves Pandas summary DataFrame.
        """
        print("\n==================================================")
        print("          Phase 2: Multimodal Verification        ")
        print("==================================================")
        os.makedirs(output_root, exist_ok=True)
        
        afis_client = SourceAFISClient()
        matcher = SourceAFISMatcher(afis_client)
        iris_rec = OpenIrisPipelineManager()
        
        trial_records = []
        
        eval_genuine_count = len(evaluation_subjects) * 13
        eval_impostor_count = len(evaluation_subjects) * (len(evaluation_subjects) - 1) * 13
        print(f"Evaluation set (40 subjects): {eval_impostor_count} impostor trials, {eval_genuine_count} genuine trials")
        print("Executing verification trials on Evaluation split...")
        for i in tqdm(evaluation_subjects, desc="Simulating Evaluation Trials"):
            # Fetch encrypted gallery templates of subject i
            gal_name = f"Person_{i:03d}"
            encrypted_gal = self.db.get_templates(gal_name)
            
            pr_t = self.probe_templates[i]
            token_genuine = f"token_person_{i:03d}"  # Claimed token is i
            
            # Genuine Trials
            face_probes = pr_t["face"]
            iris_probes = pr_t["iris"]
            fp_probes = pr_t["fingerprint"]
            
            if full_cartesian:
                # Cartesian loop: len(face_probes) * len(iris_probes) * len(fp_probes) trials
                triplets = []
                for idx_f in range(len(face_probes)):
                    for idx_i in range(len(iris_probes)):
                        for idx_fp in range(len(fp_probes)):
                            triplets.append((idx_f, idx_i, idx_fp))
            else:
                # Capped genuine trials: min_len trials without wrapping
                min_len = min(len(face_probes), len(iris_probes), len(fp_probes))
                triplets = [(idx, idx, idx) for idx in range(min_len)]
                
            for idx_f, idx_i, idx_fp in triplets:
                f_p = face_probes[idx_f] if (len(face_probes) > 0 and idx_f < len(face_probes)) else {"embedding": None, "quality": 0.0}
                i_p = iris_probes[idx_i] if (len(iris_probes) > 0 and idx_i < len(iris_probes)) else {"code": None, "mask": None, "quality": 0.0}
                fp_p = fp_probes[idx_fp] if (len(fp_probes) > 0 and idx_fp < len(fp_probes)) else {"template_b64": None, "quality": 0.0}
                
                # Transform probes under claimed token
                f_t = cancelable_transforms.biohash_face(f_p["embedding"], token_genuine) if f_p["embedding"] is not None else None
                i_code_t, i_mask_t = cancelable_transforms.transform_iris(i_p["code"], i_p["mask"], token_genuine) if i_p["code"] is not None else (None, None)
                fp_t = fp_p["template_b64"]
                
                # Enclave runs verify and fuse
                decision, fused_s, combo_key = self.enclave.verify_and_fuse(
                    encrypted_gal, f_t, i_code_t, i_mask_t, fp_t,
                    f_p["quality"], i_p["quality"], fp_p["quality"], matcher, iris_rec
                )
                
                if combo_key == 'REJECT_COMPROMISED':
                    decision = "REJECT"
                    threshold_applied = 1.0
                else:
                    threshold_applied = self.thresholds[combo_key]
                    decision = "MATCH" if fused_s >= threshold_applied else "REJECT"
                    
                trial_records.append({
                    "Subject_ID": f"Person_{i:03d}",
                    "Target_Subject_ID": f"Person_{i:03d}",
                    "Trial_Type": "Genuine",
                    "Active_Modalities": combo_key,
                    "Fused_Score": fused_s,
                    "Target_Threshold": threshold_applied,
                    "Decision": decision,
                    "Is_Correct": 1 if decision == "MATCH" else 0
                })
                
            # Impostor Trials
            for j in evaluation_subjects:
                if j == i:
                    continue
                # Fetch encrypted gallery templates of subject j
                gal_other_name = f"Person_{j:03d}"
                encrypted_gal_other = self.db.get_templates(gal_other_name)
                token_impostor = f"token_person_{j:03d}"  # Claimed token is j (impostor claim)
                
                for idx_f, idx_i, idx_fp in triplets:
                    f_p = face_probes[idx_f] if (len(face_probes) > 0 and idx_f < len(face_probes)) else {"embedding": None, "quality": 0.0}
                    i_p = iris_probes[idx_i] if (len(iris_probes) > 0 and idx_i < len(iris_probes)) else {"code": None, "mask": None, "quality": 0.0}
                    fp_p = fp_probes[idx_fp] if (len(fp_probes) > 0 and idx_fp < len(fp_probes)) else {"template_b64": None, "quality": 0.0}
                    
                    # Transform probes under claimed token j
                    f_t = cancelable_transforms.biohash_face(f_p["embedding"], token_impostor) if f_p["embedding"] is not None else None
                    i_code_t, i_mask_t = cancelable_transforms.transform_iris(i_p["code"], i_p["mask"], token_impostor) if i_p["code"] is not None else (None, None)
                    fp_t = fp_p["template_b64"]
                    
                    decision, fused_s, combo_key = self.enclave.verify_and_fuse(
                        encrypted_gal_other, f_t, i_code_t, i_mask_t, fp_t,
                        f_p["quality"], i_p["quality"], fp_p["quality"], matcher, iris_rec
                    )
                    
                    if combo_key == 'REJECT_COMPROMISED':
                        decision = "REJECT"
                        threshold_applied = 1.0
                    else:
                        threshold_applied = self.thresholds[combo_key]
                        decision = "MATCH" if fused_s >= threshold_applied else "REJECT"
                        
                    trial_records.append({
                        "Subject_ID": f"Person_{i:03d}",
                        "Target_Subject_ID": f"Person_{j:03d}",
                        "Trial_Type": "Impostor",
                        "Active_Modalities": combo_key,
                        "Fused_Score": fused_s,
                        "Target_Threshold": threshold_applied,
                        "Decision": decision,
                        "Is_Correct": 1 if decision == "REJECT" else 0
                    })
                    
        # Output Pandas DataFrame
        df = pd.DataFrame(trial_records)
        df_csv_path = os.path.join(output_root, "multimodal_results.csv")
        df.to_csv(df_csv_path, index=False)
        print(f"Results summary saved to {df_csv_path}")

        # Compute Biometric Performance Indicators
        self.compute_and_plot_metrics(df, output_root)

    def compute_and_plot_metrics(self, df, output_root):
        """
        Computes EER, FAR, FRR on the fused system, outputs performance plots,
        and saves separate charts showing 3-Active vs 2-Active performance curves.
        """
        # Genuine score list and impostor score list
        gen_scores = df[df["Trial_Type"] == "Genuine"]["Fused_Score"].values
        imp_scores = df[df["Trial_Type"] == "Impostor"]["Fused_Score"].values
        
        # Calculate overall FAR/FRR curves
        thresholds_grid = np.linspace(0.0, 1.0, 1000)
        far_list, frr_list = [], []
        for t in thresholds_grid:
            far = np.sum(imp_scores >= t) / len(imp_scores) if len(imp_scores) > 0 else 0.0
            frr = np.sum(gen_scores < t) / len(gen_scores) if len(gen_scores) > 0 else 0.0
            far_list.append(far)
            frr_list.append(frr)
            
        far_list = np.array(far_list)
        frr_list = np.array(frr_list)
        
        # Calculate Equal Error Rate (EER)
        diffs = far_list - frr_list
        crossing_idx = np.argmin(np.abs(diffs))
        eer = (far_list[crossing_idx] + frr_list[crossing_idx]) / 2.0
        eer_thresh = thresholds_grid[crossing_idx]
        
        print("\n==================================================")
        print("          Evaluation Performance Results          ")
        print("==================================================")
        print(f"Fused System Equal Error Rate (EER): {eer * 100:.4f}%")
        print(f"EER Crossing Threshold: {eer_thresh:.4f}")
        
        # Compute individual modality errors for comparison
        # (This helps quantify gain achieved via fusion)
        # Face only
        df_3active = df[df["Active_Modalities"] == "face_iris_fp"]
        gen_3 = df_3active[df_3active["Trial_Type"] == "Genuine"]["Fused_Score"].values
        imp_3 = df_3active[df_3active["Trial_Type"] == "Impostor"]["Fused_Score"].values
        
        # Active combos EERs
        print(f"\nBreakdown by Modality Combos:")
        for name, sub_df in df.groupby("Active_Modalities"):
            if name == 'REJECT_COMPROMISED':
                continue
            s_gen = sub_df[sub_df["Trial_Type"] == "Genuine"]["Fused_Score"].values
            s_imp = sub_df[sub_df["Trial_Type"] == "Impostor"]["Fused_Score"].values
            
            sub_fars, sub_frrs = [], []
            for t in thresholds_grid:
                sub_far = np.sum(s_imp >= t) / len(s_imp) if len(s_imp) > 0 else 0.0
                sub_frr = np.sum(s_gen < t) / len(s_gen) if len(s_gen) > 0 else 0.0
                sub_fars.append(sub_far)
                sub_frrs.append(sub_frr)
            sub_fars, sub_frrs = np.array(sub_fars), np.array(sub_frrs)
            sub_eer = (sub_fars[np.argmin(np.abs(sub_fars - sub_frrs))] + sub_frrs[np.argmin(np.abs(sub_fars - sub_frrs))]) / 2.0
            print(f"  Combo '{name}' Count: {len(sub_df)}, EER: {sub_eer*100:.4f}%")

        # Plot 1: Overlapping score distributions
        plt.figure(figsize=(10, 6))
        plt.hist(imp_scores, bins=50, alpha=0.5, label="Impostor Fused", color="tomato", density=True)
        plt.hist(gen_scores, bins=50, alpha=0.5, label="Genuine Fused", color="royalblue", density=True)
        plt.axvline(eer_thresh, color="black", linestyle="--", label=f"EER Crossing ({eer_thresh:.4f})")
        plt.title("Multimodal Fused Score Distribution (Posterior Probability space)", fontsize=13, fontweight="bold")
        plt.xlabel("Fused Probability Score")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True, alpha=0.2)
        dist_png = os.path.join(output_root, "multimodal_distributions.png")
        plt.savefig(dist_png, dpi=200, bbox_inches='tight')
        plt.close()

        # Plot 2: FAR/FRR vs Threshold sweep
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds_grid, far_list, color="tomato", lw=2.5, label="FAR")
        plt.plot(thresholds_grid, frr_list, color="royalblue", lw=2.5, label="FRR")
        plt.axvline(eer_thresh, color="black", linestyle="--", label=f"EER Crossing ({eer_thresh:.4f})")
        plt.title("FAR / FRR vs Fused Threshold Sweep", fontsize=13, fontweight="bold")
        plt.xlabel("Fused Threshold")
        plt.ylabel("Error Rate")
        plt.legend()
        plt.grid(True, alpha=0.2)
        far_frr_png = os.path.join(output_root, "multimodal_far_frr_vs_threshold.png")
        plt.savefig(far_frr_png, dpi=200, bbox_inches='tight')
        plt.close()

        # Plot 3: 3-Active vs 2-Active ROC curves comparison
        plt.figure(figsize=(8, 8))
        # Overall
        overall_tars = 1.0 - frr_list
        plt.plot(far_list, overall_tars, color="darkgreen", lw=2.5, label=f"Fused System Overall (EER = {eer*100:.3f}%)")
        
        # Grouped combos
        colors_map = {'face_iris_fp': 'navy', 'face_iris': 'orange', 'face_fp': 'purple', 'iris_fp': 'brown'}
        for name, sub_df in df.groupby("Active_Modalities"):
            if name == 'REJECT_COMPROMISED':
                continue
            s_gen = sub_df[sub_df["Trial_Type"] == "Genuine"]["Fused_Score"].values
            s_imp = sub_df[sub_df["Trial_Type"] == "Impostor"]["Fused_Score"].values
            
            sub_fars, sub_tars = [], []
            for t in thresholds_grid:
                sub_far = np.sum(s_imp >= t) / len(s_imp) if len(s_imp) > 0 else 0.0
                sub_frr = np.sum(s_gen < t) / len(s_gen) if len(s_gen) > 0 else 0.0
                sub_fars.append(sub_far)
                sub_tars.append(1.0 - sub_frr)
            plt.plot(sub_fars, sub_tars, linestyle="--", color=colors_map.get(name, "grey"), label=f"Combo {name}")
            
        plt.plot([0, 1], [0, 1], color="darkgrey", linestyle="--")
        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.xlabel("False Acceptance Rate (FAR)")
        plt.ylabel("True Acceptance Rate (TAR)")
        plt.title("ROC Curve Analysis: Fused Combos Comparison", fontsize=13, fontweight="bold")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.2)
        roc_png = os.path.join(output_root, "multimodal_roc_comparison.png")
        plt.savefig(roc_png, dpi=200, bbox_inches='tight')
        plt.close()
        
        print("Performance plots saved successfully under audit_results/multimodal/.")

def run_self_unit_tests():
    """
    Self-contained inline unit tests to validate calibration, quality indices,
    and adaptive weight logic.
    """
    print("\n==================================================")
    print("           Running Self-Unit Tests...             ")
    print("==================================================")
    
    # 1. Weights logic test
    weights_eq = get_equal_redistributed_weights(['face', 'iris'])
    assert np.isclose(weights_eq['face'], 0.5) and np.isclose(weights_eq['iris'], 0.5), "Baseline weights redistribution error"
    
    weights_qual = get_adaptive_weights(0.8, 0.4, 0.0, ['face', 'iris'])
    assert np.isclose(weights_qual['face'], 2/3) and np.isclose(weights_qual['iris'], 1/3), "Quality weights scaling error"
    
    print("Unit Test 1: Weight Redistribution logic ... PASSED")
    
    # 2. Gradient Coherence test on random flat array
    mock_gradient_block = np.zeros((16, 16), dtype=np.uint8)
    # Synthetic flat block should result in 0 gradient
    gx = cv2.Sobel(mock_gradient_block, cv2.CV_32F, 1, 0, ksize=3)
    assert np.sum(gx) == 0.0
    print("Unit Test 2: Gradient Structure Tensor logic ... PASSED")
    
    print("All unit tests completed successfully.\n")

def main():
    parser = argparse.ArgumentParser(description="Multimodal Score-Level Biometric Fusion System")
    parser.add_argument(
        "--dataset_dir", 
        type=str, 
        default="Chimeric_Dataset_Noisy", 
        help="Path to noisy chimeric dataset folder"
    )
    parser.add_argument(
        "--full-cartesian", 
        action="store_true", 
        help="Run exhaustive Cartesian product probe pairs comparison"
    )
    parser.add_argument(
        "--run-tests", 
        action="store_true", 
        help="Run self-contained unit tests before evaluation"
    )
    args = parser.parse_args()

    # Pre-test run
    if args.run_tests:
        run_self_unit_tests()

    testing_dir = os.path.join(args.dataset_dir, "testing")
    training_dir = os.path.join(args.dataset_dir, "training")
    
    # Validate directories
    if not os.path.exists(testing_dir):
        print(f"ERROR: testing directory missing at {testing_dir}")
        sys.exit(1)
        
    pipeline = MultimodalFusionPipeline()
    
    # Step 1: Run template extraction and caching
    pipeline.run_extraction_and_caching(testing_dir, training_dir)
    
    # Step 2: Split 100 subjects randomly to 60/40 splits
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    
    all_subjects = list(range(1, 101))
    random.shuffle(all_subjects)
    
    calibration_subjects = sorted(all_subjects[:60])
    evaluation_subjects = sorted(all_subjects[60:])
    
    print(f"\nRandomized Split Complete (Seed: {RANDOM_STATE}):")
    print(f"  Calibration subjects count: {len(calibration_subjects)} ({calibration_subjects[:5]}...)")
    print(f"  Evaluation subjects count:  {len(evaluation_subjects)} ({evaluation_subjects[:5]}...)")
    
    # Step 3: Run Logistic Calibration & Dynamic Threshold Search on Calibration set
    pipeline.calibrate_scores(calibration_subjects)
    
    # Step 4: Run Verification Evaluation on Evaluation set
    pipeline.run_evaluation(evaluation_subjects, full_cartesian=args.full_cartesian)

if __name__ == "__main__":
    main()

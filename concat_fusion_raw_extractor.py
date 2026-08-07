"""
Standalone Multimodal Raw Feature Extractor Module for Concatenation Fusion.

Loads deep learning feature extraction models for Face, Iris, and Fingerprint:
1. Face Extractor: ArcFace / ResNet18 (512-D vector)
2. Iris Extractor: ArcIris iresnet100 + Open-IRIS normalization (512-D vector)
3. Fingerprint Extractor: DeepPrint TexMinu (512-D vector)

Combines vectors into a concatenated 1536-D L2-normalized feature vector.
"""

import os
import sys
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

# Ensure concate fusion local directories and project root are in sys.path
CONCATE_FUSION_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CONCATE_FUSION_DIR, ".."))

local_paths = [
    CONCATE_FUSION_DIR,
    os.path.join(CONCATE_FUSION_DIR, "src"),
    os.path.join(CONCATE_FUSION_DIR, "src", "pipelines"),
    os.path.join(CONCATE_FUSION_DIR, "src", "data_processing"),
    os.path.join(CONCATE_FUSION_DIR, "src", "extractors"),
    os.path.join(CONCATE_FUSION_DIR, "src", "open-iris", "src"),
    os.path.join(CONCATE_FUSION_DIR, "flx")
]
for lp in local_paths:
    if os.path.exists(lp) and lp not in sys.path:
        sys.path.insert(0, lp)

ARCIRIS_CANDIDATE_DIRS = [
    os.path.join(CONCATE_FUSION_DIR, "OpenSourceIrisRecognition", "methods", "ArcIris", "Python"),
    os.path.join(PROJECT_ROOT, "data", "raw", "OpenSourceIrisRecognition", "methods", "ArcIris", "Python"),
    os.path.join(PROJECT_ROOT, "OpenSourceIrisRecognition", "methods", "ArcIris", "Python")
]
for d in ARCIRIS_CANDIDATE_DIRS:
    if os.path.exists(d) and d not in sys.path:
        sys.path.insert(0, d)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


class MultimodalRawFeatureExtractor:
    def __init__(self, use_gpu=False):
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        print(f"[MultimodalRawFeatureExtractor] Initializing on device: {self.device}")

        self._init_face_extractor()
        self._init_iris_extractor()
        self._init_fingerprint_extractor()

    def _init_face_extractor(self):
        """Initialize Face Extractor (ArcFace ONNX w600k_r50.onnx with ResNet-50 backbone)."""
        print("  - Loading Face Extractor (ArcFace w600k_r50.onnx, ResNet-50 backbone)...")
        self.face_session = None
        self.face_extractor_fallback = None

        LOCAL_MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
        face_candidates = [
            os.path.join(LOCAL_MODELS_DIR, "w600k_r50.onnx"),
            os.path.join(PROJECT_ROOT, "models", "w600k_r50.onnx"),
            os.path.expanduser("~/.insightface/models/buffalo_l/w600k_r50.onnx")
        ]

        model_path = None
        for cand in face_candidates:
            if os.path.exists(cand):
                model_path = cand
                break

        if model_path is not None:
            try:
                import onnxruntime as ort
                available_providers = ort.get_available_providers()
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if (self.device.type == "cuda" and 'CUDAExecutionProvider' in available_providers) else ['CPUExecutionProvider']
                self.face_session = ort.InferenceSession(model_path, providers=providers)
                print(f"    [Success] ArcFace model loaded from: {model_path} (ResNet-50 backbone)")
            except Exception as e:
                print(f"    [Warning] Could not initialize ONNX session for ArcFace: {e}")

        if self.face_session is None:
            print("    [Fallback] Using ResNet-18 ImageNet feature extractor fallback.")
            from src.extractors.resnet_extractor import ResNet18FeatureExtractor
            self.face_extractor_fallback = ResNet18FeatureExtractor(use_gpu=(self.device.type == "cuda"))

    def _init_iris_extractor(self):
        """Initialize Iris Extractor (ArcIris iresnet100 + OpenIris Pipeline Manager)."""
        print("  - Loading Iris Extractor (ArcIris iresnet100)...")
        self.iris_model = None
        self.iris_rec = None

        # OpenIris Pipeline
        try:
            from open_iris_pipeline import OpenIrisPipelineManager
            self.iris_rec = OpenIrisPipelineManager()
        except Exception as e:
            print(f"    [Warning] OpenIrisPipelineManager initialization fallback: {e}")

        # ArcIris Model Weights Candidates (Prioritizes local concate fusion/models/)
        LOCAL_MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
        weights_candidates = [
            os.path.join(LOCAL_MODELS_DIR, "ResNet100_154000.pt"),
            os.path.join(PROJECT_ROOT, "data", "raw", "OpenSourceIrisRecognition", "methods", "ArcIris", "Python", "models", "ResNet100_154000.pt"),
            os.path.join(PROJECT_ROOT, "OpenSourceIrisRecognition", "methods", "ArcIris", "Python", "models", "ResNet100_154000.pt"),
            os.path.join(PROJECT_ROOT, "models", "ResNet100_154000.pt")
        ]
        
        weights_path = None
        for cand in weights_candidates:
            if os.path.exists(cand):
                weights_path = cand
                break

        if weights_path is not None:
            try:
                from modules.network import iresnet100
                iris_net = iresnet100(pretrained=False, progress=False)
                state_dict = torch.load(weights_path, map_location=self.device)
                clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                iris_net.load_state_dict(clean_state_dict, strict=True)
                iris_net.to(self.device)
                iris_net.eval()
                for p in iris_net.parameters():
                    p.requires_grad = False
                self.iris_model = iris_net
                print(f"    [Success] ArcIris iresnet100 model loaded from: {weights_path}")
            except Exception as e:
                print(f"    [Warning] Could not load ArcIris iresnet100: {e}")
        else:
            print(f"    [Warning] ArcIris weights not found in candidates.")

        self.iris_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])

    def _init_fingerprint_extractor(self):
        """Initialize Fingerprint Extractor (DeepPrint TexMinu)."""
        print("  - Loading Fingerprint Extractor (DeepPrint TexMinu)...")
        self.dp_model = None
        self.binarizer = None

        LOCAL_MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
        dp_candidates = [
            os.path.join(LOCAL_MODELS_DIR, "DeepPrint_Tex_512", "best_model.pyt"),
            os.path.join(PROJECT_ROOT, "models", "DeepPrint_Tex_512", "best_model.pyt")
        ]

        dp_weights = None
        for cand in dp_candidates:
            if os.path.exists(cand):
                dp_weights = cand
                break

        if dp_weights is not None:
            try:
                import flx.models.deep_print_arch as dpa
                from flx.image_processing.binarization import LazilyAllocatedBinarizer

                dp_checkpoint = torch.load(dp_weights, map_location=self.device)
                net = dpa.DeepPrint_TexMinu(8000, 256, 256)
                net.load_state_dict(dp_checkpoint["model_state_dict"])
                net.to(self.device)
                net.eval()
                for p in net.parameters():
                    p.requires_grad = False
                self.dp_model = net
                self.binarizer = LazilyAllocatedBinarizer(1.8)
                print(f"    [Success] DeepPrint model loaded from: {dp_weights}")
            except Exception as e:
                print(f"    [Warning] Could not load DeepPrint model: {e}")
        else:
            print(f"    [Warning] DeepPrint weights not found.")

    @property
    def degraded_components(self) -> list:
        degraded = []
        if self.face_session is None:
            degraded.append("face (ArcFace ONNX)")
        if self.iris_model is None or self.iris_rec is None:
            degraded.append("iris (ArcIris iresnet100)")
        if self.dp_model is None:
            degraded.append("fingerprint (DeepPrint TexMinu)")
        return degraded

    @property
    def is_degraded(self) -> bool:
        return len(self.degraded_components) > 0

    def get_status(self) -> dict:
        return {
            "degraded_mode": self.is_degraded,
            "degraded_components": self.degraded_components,
            "models_loaded": {
                "face_arcface_onnx": self.face_session is not None,
                "iris_arciris_iresnet100": (self.iris_model is not None and self.iris_rec is not None),
                "fingerprint_deepprint": self.dp_model is not None
            }
        }

    def _get_fallback_extractor(self):
        if self.face_extractor_fallback is None:
            from src.extractors.resnet_extractor import ResNet18FeatureExtractor
            self.face_extractor_fallback = ResNet18FeatureExtractor(use_gpu=(self.device.type == "cuda"))
        return self.face_extractor_fallback

    def extract_face_embedding(self, face_image_path: str) -> np.ndarray:
        """Extract 512-D L2-normalized Face embedding using ArcFace ONNX (ResNet-50 backbone)."""
        if not os.path.exists(face_image_path):
            raise FileNotFoundError(f"Face image not found: {face_image_path}")

        if self.face_session is not None:
            import cv2
            from extract_chimeric_face_onnx_embeddings import align_face, preprocess_tensor, get_arcface_embedding
            img_f = cv2.imread(face_image_path)
            if img_f is not None:
                aligned_patch = align_face(img_f, landmarks=None)
                tensor = preprocess_tensor(aligned_patch)
                emb = get_arcface_embedding(tensor, self.face_session)
                return emb.astype(np.float32)

        # Fallback to ResNet-18 ImageNet extractor
        print("  [Fallback] Extracting face feature via ResNet-18 extractor fallback.")
        return self._get_fallback_extractor().extract_features(face_image_path, l2_normalize=True).astype(np.float32)

    def extract_iris_embedding(self, iris_image_path: str) -> np.ndarray:
        """Extract 512-D L2-normalized Iris embedding from raw image file."""
        if not os.path.exists(iris_image_path):
            raise FileNotFoundError(f"Iris image not found: {iris_image_path}")

        if self.iris_model is not None and self.iris_rec is not None:
            _ = self.iris_rec.generate_biometric_template(iris_image_path, eye_side="right")
            norm_img = self.iris_rec.last_normalized_image
            if norm_img is not None:
                import cv2
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                norm_img = clahe.apply(norm_img)
                im_polar = Image.fromarray(norm_img, "L").resize((512, 64), Image.Resampling.BILINEAR)
                im_tensor = self.iris_transform(im_polar).unsqueeze(0).repeat(1, 3, 1, 1).to(self.device)
                with torch.no_grad():
                    emb_tensor = self.iris_model(im_tensor)
                    emb_tensor = torch.nn.functional.normalize(emb_tensor, dim=1)
                    return emb_tensor[0].cpu().numpy().astype(np.float32)

        # Fallback to ResNet feature extractor if ArcIris pipeline unavailable
        print("  [Fallback] Extracting iris feature via ResNet extractor fallback.")
        return self._get_fallback_extractor().extract_features(iris_image_path, l2_normalize=True).astype(np.float32)

    def extract_fingerprint_embedding(self, fp_image_path: str) -> np.ndarray:
        """Extract 512-D L2-normalized Fingerprint embedding from raw image file."""
        if not os.path.exists(fp_image_path):
            raise FileNotFoundError(f"Fingerprint image not found: {fp_image_path}")

        if self.dp_model is not None:
            from flx.data.image_helpers import pad_and_resize_to_deepprint_input_size
            img_fp = Image.open(fp_image_path).convert("L")
            img_np = np.array(img_fp)
            preprocessed = pad_and_resize_to_deepprint_input_size(img_np, fill=1.0)
            if self.binarizer is not None:
                preprocessed = self.binarizer(preprocessed)
            tensor = torch.stack([preprocessed, preprocessed], dim=0).to(self.device)
            with torch.no_grad():
                out = self.dp_model(tensor)
                emb_tensor = torch.cat([out.texture_embeddings, out.minutia_embeddings], dim=1)
                emb_fp = emb_tensor[0].cpu().numpy()
                norm = np.linalg.norm(emb_fp)
                if norm > 1e-8:
                    emb_fp = emb_fp / norm
                return emb_fp.astype(np.float32)

        # Fallback to ResNet feature extractor if DeepPrint model unavailable
        print("  [Fallback] Extracting fingerprint feature via ResNet extractor fallback.")
        return self._get_fallback_extractor().extract_features(fp_image_path, l2_normalize=True).astype(np.float32)

    def extract_and_fuse_from_files(self, face_path: str, iris_path: str, fp_path: str) -> np.ndarray:
        """
        Extracts 512-D embeddings for Face, Iris, and Fingerprint from raw files,
        concatenates them into a 1536-D vector, and L2-normalizes the result.
        """
        f_emb = self.extract_face_embedding(face_path)
        i_emb = self.extract_iris_embedding(iris_path)
        fp_emb = self.extract_fingerprint_embedding(fp_path)

        concatenated = np.concatenate([f_emb, i_emb, fp_emb], axis=0).astype(np.float32)
        norm = np.linalg.norm(concatenated)
        if norm > 1e-6:
            concatenated = concatenated / norm
        return concatenated


def main():
    print("================================================================================")
    print("TESTING MULTIMODAL RAW FEATURE EXTRACTOR IN CONCATE FUSION")
    print("================================================================================")

    extractor = MultimodalRawFeatureExtractor()

    sample_candidates = [
        os.path.join(PROJECT_ROOT, "concate fusion", "data", "Chimeric_Dataset_Noisy", "training", "Person_001"),
        os.path.join(PROJECT_ROOT, "data", "chimeric", "Chimeric_Dataset_Noisy", "training", "Person_001")
    ]

    sample_dir = None
    for cand in sample_candidates:
        if os.path.exists(cand):
            sample_dir = cand
            break

    if sample_dir is not None:
        face_path = os.path.join(sample_dir, "face.jpg")
        iris_path = os.path.join(sample_dir, "iris_right.jpg")
        fp_path = os.path.join(sample_dir, "fingerprint_right_thumb.jpg")

        print(f"\nFound sample raw images in: {sample_dir}")
        print(f"  - Face: {os.path.basename(face_path)} ({os.path.getsize(face_path)} bytes)")
        print(f"  - Iris: {os.path.basename(iris_path)} ({os.path.getsize(iris_path)} bytes)")
        print(f"  - Fingerprint: {os.path.basename(fp_path)} ({os.path.getsize(fp_path)} bytes)")

        fused_vec = extractor.extract_and_fuse_from_files(face_path, iris_path, fp_path)
        print(f"\nSuccessfully Extracted & Fused 1536-D Vector:")
        print(f"  - Vector Shape:     {fused_vec.shape}")
        print(f"  - Data Type:        {fused_vec.dtype}")
        print(f"  - L2 Norm:          {np.linalg.norm(fused_vec):.6f}")
        print(f"  - First 5 Elements: {fused_vec[:5]}")
    else:
        print("\nNo sample directory found.")


if __name__ == "__main__":
    main()

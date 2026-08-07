"""
Concatenation Fusion Biometric System — Full Enrollment & Authentication Pipelines

Implements Sections 10.1 (Enrollment) and 10.2 (Authentication) of the system design:

  ENROLLMENT:
    Biometric Capture → Quality Assessment → Feature Extraction (ArcFace + ArcIris + DeepPrint)
    → Concatenation Fusion (1536-D) → BioHash Template Protection → PQC Encryption (ML-KEM-768
    + AES-256-GCM) → ML-DSA-65 Signing → SQLite Storage

  AUTHENTICATION:
    Biometric Capture → Quality Assessment → Feature Extraction → Concatenation Fusion
    → BioHash Query BioCode → Retrieve & Decrypt Enrolled BioCode (signature-verified)
    → Fractional Hamming Distance Matching → Accept/Reject Decision → ML-DSA Signed Audit Log

ISOLATION RULES:
  1. All logic resides in this standalone file.
  2. Imports existing modules read-only — no modifications to any existing file.
  3. Uses a separate database file (concat_fusion_system.db) to preserve prior experiment data.
  4. Audit log appended to data/databases_and_cache/auth_audit_log.jsonl (new file).
"""

import os
import sys
import time
import json
import pickle
import hashlib
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# sys.path setup — mirrors existing scripts to find all project submodules
# ──────────────────────────────────────────────────────────────────────────────
CONCATE_FUSION_DIR = os.path.abspath(os.path.dirname(__file__))
_local_paths = [
    CONCATE_FUSION_DIR,
    os.path.join(CONCATE_FUSION_DIR, "src"),
    os.path.join(CONCATE_FUSION_DIR, "src", "pipelines"),
    os.path.join(CONCATE_FUSION_DIR, "src", "security"),
    os.path.join(CONCATE_FUSION_DIR, "src", "data_processing"),
    os.path.join(CONCATE_FUSION_DIR, "src", "extractors"),
    os.path.join(CONCATE_FUSION_DIR, "src", "open-iris", "src"),
    os.path.join(CONCATE_FUSION_DIR, "flx"),
]
for _lp in _local_paths:
    if os.path.exists(_lp) and _lp not in sys.path:
        sys.path.insert(0, _lp)

# Existing module imports (read-only usage)
from concat_fusion_raw_extractor import MultimodalRawFeatureExtractor
from concat_fusion_biohash_experiment import fuse_concatenate
from concat_fusion_pqc_crypto import PQCCryptoEngine, ConcatFusionDatabaseManager
from src.security import cancelable_transforms

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_SYSTEM_DB = os.path.join(CONCATE_FUSION_DIR, "data", "databases_and_cache", "concat_fusion_system.db")
DEFAULT_AUDIT_LOG = os.path.join(CONCATE_FUSION_DIR, "data", "databases_and_cache", "auth_audit_log.jsonl")
DEFAULT_CACHE_PATH = os.path.join(CONCATE_FUSION_DIR, "data", "databases_and_cache", "transformer_templates_cache.pkl")
TRAINING_DATA_DIR = os.path.join(CONCATE_FUSION_DIR, "data", "Chimeric_Dataset_Noisy", "training")
TESTING_DATA_DIR = os.path.join(CONCATE_FUSION_DIR, "data", "Chimeric_Dataset_Noisy", "testing")

# Quality gate thresholds
MIN_EMBEDDING_NORM = 0.1           # Minimum L2 norm to accept an embedding as valid
EXPECTED_EMBEDDING_DIM = 512       # Each modality must produce exactly 512-D
FUSED_DIM = 1536                   # 3 × 512-D concatenated
DEFAULT_HAMMING_THRESHOLD = 0.30   # Decision threshold for fractional Hamming distance (EER floor: ~0.31)


# ══════════════════════════════════════════════════════════════════════════════
# Data Classes
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class QualityScores:
    """Per-modality quality assessment results."""
    face_norm: float = 0.0
    iris_norm: float = 0.0
    fingerprint_norm: float = 0.0
    face_dim: int = 0
    iris_dim: int = 0
    fingerprint_dim: int = 0
    face_valid: bool = False
    iris_valid: bool = False
    fingerprint_valid: bool = False
    overall_valid: bool = False
    rejection_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EnrollmentResult:
    """Result of the enrollment pipeline."""
    user_id: str
    template_id: str = ""
    quality_scores: Optional[QualityScores] = None
    fused_vector_dim: int = 0
    biohash_vector_dim: int = 0
    enrollment_latency_ms: float = 0.0
    extraction_latency_ms: float = 0.0
    fusion_latency_ms: float = 0.0
    biohash_latency_ms: float = 0.0
    encryption_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    degraded_mode: bool = False
    degraded_components: List[str] = field(default_factory=list)
    success: bool = False
    error: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


@dataclass
class AuthenticationResult:
    """Result of the authentication pipeline."""
    user_id: str
    decision: str = "ERROR"           # "ACCEPT", "REJECT", "ERROR"
    hamming_distance: float = 1.0
    cosine_similarity: float = -1.0
    threshold: float = DEFAULT_HAMMING_THRESHOLD
    query_quality_scores: Optional[QualityScores] = None
    total_latency_ms: float = 0.0
    extraction_latency_ms: float = 0.0
    fusion_latency_ms: float = 0.0
    biohash_latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    match_latency_ms: float = 0.0
    degraded_mode: bool = False
    degraded_components: List[str] = field(default_factory=list)
    audit_signature_hex: str = ""
    reason: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ══════════════════════════════════════════════════════════════════════════════
# Main System Class
# ══════════════════════════════════════════════════════════════════════════════

class ConcatFusionBiometricSystem:
    """
    End-to-end multimodal biometric system implementing:
      • Section 10.1 — Enrollment Process Pipeline
      • Section 10.2 — Authentication Process Pipeline

    Uses concatenation-based feature-level fusion of Face (ArcFace 512-D),
    Iris (ArcIris 512-D), and Fingerprint (DeepPrint 512-D) modalities into
    a 1536-D fused representation, protected by BioHash cancelable transforms
    and post-quantum cryptographic envelope encryption.
    """

    def __init__(
        self,
        db_path: str = DEFAULT_SYSTEM_DB,
        audit_log_path: str = DEFAULT_AUDIT_LOG,
        use_gpu: bool = False,
        load_models: bool = True,
    ):
        """
        Initialize the biometric system.

        Args:
            db_path: Path for the SQLite enrollment database.
            audit_log_path: Path for the JSONL audit log file.
            use_gpu: Whether to use GPU for deep learning inference.
            load_models: If True, loads all DL models at startup (heavy, ~1.4GB).
                         If False, skips model loading (cache-only mode).
        """
        print("=" * 80)
        print("INITIALIZING CONCATENATION FUSION BIOMETRIC SYSTEM")
        print("=" * 80)

        self.db_path = db_path
        self.audit_log_path = audit_log_path

        # ── 1. Initialize PQC Crypto Engine ──
        print("\n[1/4] Initializing Post-Quantum Cryptographic Engine...")
        self.crypto_engine = PQCCryptoEngine()
        print(f"  ✓ ML-KEM-768 key pair loaded")
        print(f"  ✓ ML-DSA-65 signing key pair loaded")

        # ── 2. Initialize Database Manager ──
        print("\n[2/4] Initializing Encrypted Template Database...")
        self.db_manager = ConcatFusionDatabaseManager(
            db_path=self.db_path,
            crypto_engine=self.crypto_engine
        )
        enrolled_count = self.db_manager.get_enrolled_count()
        print(f"  ✓ Database ready at: {self.db_path}")
        print(f"  ✓ Currently enrolled templates: {enrolled_count}")

        # ── 3. Initialize Feature Extraction Models ──
        self.extractor = None
        if load_models:
            print("\n[3/4] Loading Deep Learning Feature Extraction Models...")
            self.extractor = MultimodalRawFeatureExtractor(use_gpu=use_gpu)
            print(f"  ✓ Feature extractors initialized")
        else:
            print("\n[3/4] Skipping model loading (cache-only mode)")

        # ── 4. Ensure Audit Log Directory ──
        print("\n[4/4] Initializing Audit Log...")
        os.makedirs(os.path.dirname(self.audit_log_path), exist_ok=True)
        print(f"  ✓ Audit log path: {self.audit_log_path}")

        # ── Clear BioHash projection cache for clean start ──
        cancelable_transforms.clear_projection_cache()

        print("\n" + "=" * 80)
        print("SYSTEM INITIALIZATION COMPLETE")
        print("=" * 80)

    # ══════════════════════════════════════════════════════════════════════════
    # Quality Assessment
    # ══════════════════════════════════════════════════════════════════════════

    def _assess_quality(
        self,
        face_emb: np.ndarray,
        iris_emb: np.ndarray,
        fp_emb: np.ndarray,
    ) -> QualityScores:
        """
        Performs per-modality quality assessment on extracted embeddings.

        Quality gates:
          1. Embedding must not be None
          2. Embedding must be exactly 512-D
          3. Embedding must not contain NaN or Inf values
          4. L2 norm must be ≥ MIN_EMBEDDING_NORM (0.1)
        """
        qs = QualityScores()
        reasons = []

        # ── Face Quality ──
        if face_emb is None:
            reasons.append("Face: extraction returned None")
        else:
            face_flat = face_emb.flatten()
            qs.face_dim = len(face_flat)
            qs.face_norm = float(np.linalg.norm(face_flat))

            if qs.face_dim != EXPECTED_EMBEDDING_DIM:
                reasons.append(f"Face: unexpected dimension {qs.face_dim} (expected {EXPECTED_EMBEDDING_DIM})")
            elif np.any(np.isnan(face_flat)) or np.any(np.isinf(face_flat)):
                reasons.append("Face: embedding contains NaN or Inf values")
            elif qs.face_norm < MIN_EMBEDDING_NORM:
                reasons.append(f"Face: L2 norm {qs.face_norm:.6f} below threshold {MIN_EMBEDDING_NORM}")
            else:
                qs.face_valid = True

        # ── Iris Quality ──
        if iris_emb is None:
            reasons.append("Iris: extraction returned None")
        else:
            iris_flat = iris_emb.flatten()
            qs.iris_dim = len(iris_flat)
            qs.iris_norm = float(np.linalg.norm(iris_flat))

            if qs.iris_dim != EXPECTED_EMBEDDING_DIM:
                reasons.append(f"Iris: unexpected dimension {qs.iris_dim} (expected {EXPECTED_EMBEDDING_DIM})")
            elif np.any(np.isnan(iris_flat)) or np.any(np.isinf(iris_flat)):
                reasons.append("Iris: embedding contains NaN or Inf values")
            elif qs.iris_norm < MIN_EMBEDDING_NORM:
                reasons.append(f"Iris: L2 norm {qs.iris_norm:.6f} below threshold {MIN_EMBEDDING_NORM}")
            else:
                qs.iris_valid = True

        # ── Fingerprint Quality ──
        if fp_emb is None:
            reasons.append("Fingerprint: extraction returned None")
        else:
            fp_flat = fp_emb.flatten()
            qs.fingerprint_dim = len(fp_flat)
            qs.fingerprint_norm = float(np.linalg.norm(fp_flat))

            if qs.fingerprint_dim != EXPECTED_EMBEDDING_DIM:
                reasons.append(f"Fingerprint: unexpected dimension {qs.fingerprint_dim} (expected {EXPECTED_EMBEDDING_DIM})")
            elif np.any(np.isnan(fp_flat)) or np.any(np.isinf(fp_flat)):
                reasons.append("Fingerprint: embedding contains NaN or Inf values")
            elif qs.fingerprint_norm < MIN_EMBEDDING_NORM:
                reasons.append(f"Fingerprint: L2 norm {qs.fingerprint_norm:.6f} below threshold {MIN_EMBEDDING_NORM}")
            else:
                qs.fingerprint_valid = True

        qs.rejection_reasons = reasons
        qs.overall_valid = qs.face_valid and qs.iris_valid and qs.fingerprint_valid
        return qs

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 10.1 — Enrollment Process Pipeline
    # ══════════════════════════════════════════════════════════════════════════

    def enroll_user(
        self,
        user_id: str,
        face_path: str,
        iris_path: str,
        fp_path: str,
        user_seed: str,
        user_name: Optional[str] = None,
    ) -> EnrollmentResult:
        """
        Full enrollment pipeline from raw biometric images.

        Steps:
          1. Biometric Capture & Quality Assessment
          2. Feature Extraction (ArcFace + ArcIris + DeepPrint → 3×512-D)
          3. Feature-Level Fusion (Concatenation → 1536-D, L2-normalized)
          4. Template Protection (BioHash → 1536-D bipolar BioCode)
          5. PQC Encryption (ML-KEM-768 + AES-256-GCM) & ML-DSA-65 Signing
          6. Encrypted Storage in SQLite

        Args:
            user_id: Unique identifier for the user (e.g., "Person_001").
            face_path: Path to the face image file.
            iris_path: Path to the iris image file.
            fp_path: Path to the fingerprint image file.
            user_seed: User-specific secret token for BioHash revocability.
            user_name: Optional display name (defaults to user_id).

        Returns:
            EnrollmentResult with detailed metrics and status.
        """
        if self.extractor is None:
            return EnrollmentResult(
                user_id=user_id,
                error="Feature extraction models not loaded. Initialize with load_models=True."
            )

        result = EnrollmentResult(
            user_id=user_id,
            degraded_mode=self.extractor.is_degraded if self.extractor else False,
            degraded_components=self.extractor.degraded_components if self.extractor else []
        )
        if user_name is None:
            user_name = user_id

        t_total_start = time.perf_counter()

        # ── Step 1 & 2: Biometric Capture + Feature Extraction ──
        print(f"\n  [Enrollment] Step 1-2: Extracting features for {user_id}...")
        t_extract_start = time.perf_counter()

        try:
            face_emb = self.extractor.extract_face_embedding(face_path)
        except Exception as e:
            face_emb = None
            print(f"    [WARNING] Face extraction failed: {e}")

        try:
            iris_emb = self.extractor.extract_iris_embedding(iris_path)
        except Exception as e:
            iris_emb = None
            print(f"    [WARNING] Iris extraction failed: {e}")

        try:
            fp_emb = self.extractor.extract_fingerprint_embedding(fp_path)
        except Exception as e:
            fp_emb = None
            print(f"    [WARNING] Fingerprint extraction failed: {e}")

        t_extract_end = time.perf_counter()
        result.extraction_latency_ms = (t_extract_end - t_extract_start) * 1000.0

        # ── Quality Assessment ──
        print(f"  [Enrollment] Quality assessment...")
        qs = self._assess_quality(face_emb, iris_emb, fp_emb)
        result.quality_scores = qs

        if not qs.overall_valid:
            result.error = f"Quality gate failure: {'; '.join(qs.rejection_reasons)}"
            result.total_latency_ms = (time.perf_counter() - t_total_start) * 1000.0
            print(f"    [REJECTED] {result.error}")
            return result

        print(f"    ✓ Quality: Face L2={qs.face_norm:.4f}, Iris L2={qs.iris_norm:.4f}, FP L2={qs.fingerprint_norm:.4f}")

        # ── Step 3: Feature-Level Fusion (Concatenation) ──
        print(f"  [Enrollment] Step 3: Concatenation fusion → 1536-D...")
        t_fuse_start = time.perf_counter()
        fused_vec = fuse_concatenate(face_emb, iris_emb, fp_emb)
        t_fuse_end = time.perf_counter()
        result.fusion_latency_ms = (t_fuse_end - t_fuse_start) * 1000.0
        result.fused_vector_dim = len(fused_vec)
        print(f"    ✓ Fused vector: dim={result.fused_vector_dim}, L2 norm={np.linalg.norm(fused_vec):.6f}")

        # ── Step 4: Template Protection (BioHash) ──
        print(f"  [Enrollment] Step 4: BioHash template protection...")
        t_bh_start = time.perf_counter()
        biocode = cancelable_transforms.biohash_fused(fused_vec, user_seed)
        t_bh_end = time.perf_counter()
        result.biohash_latency_ms = (t_bh_end - t_bh_start) * 1000.0
        result.biohash_vector_dim = len(biocode)
        print(f"    ✓ BioCode: dim={result.biohash_vector_dim}, L2 norm={np.linalg.norm(biocode):.6f}")

        # ── Step 5 & 6: PQC Encryption + Signing + Storage ──
        print(f"  [Enrollment] Step 5-6: PQC encryption & storage...")
        t_enc_start = time.perf_counter()

        try:
            quality_score = float(np.mean([qs.face_norm, qs.iris_norm, qs.fingerprint_norm]))
            db_result = self.db_manager.enroll_template(
                user_id=user_id,
                user_name=user_name,
                fused_vec=biocode,
                quality_score=quality_score,
            )
            result.template_id = db_result["template_id"]
            result.encryption_latency_ms = db_result["latency_ms"]
            result.success = True
            print(f"    ✓ Enrolled template: {result.template_id}")
            print(f"    ✓ Encryption latency: {result.encryption_latency_ms:.4f} ms")
        except Exception as e:
            result.error = f"PQC encryption/storage failed: {e}"
            print(f"    [ERROR] {result.error}")

        t_enc_end = time.perf_counter()
        result.enrollment_latency_ms = (t_enc_end - t_enc_start) * 1000.0
        result.total_latency_ms = (time.perf_counter() - t_total_start) * 1000.0

        return result

    def enroll_from_embeddings(
        self,
        user_id: str,
        face_emb: np.ndarray,
        iris_emb: np.ndarray,
        fp_emb: np.ndarray,
        user_seed: str,
        user_name: Optional[str] = None,
    ) -> EnrollmentResult:
        """
        Enrollment pipeline using pre-computed embeddings (from cache).
        Skips raw image extraction; performs quality check → fusion → BioHash → encrypt → store.
        """
        result = EnrollmentResult(user_id=user_id)
        if user_name is None:
            user_name = user_id

        t_total_start = time.perf_counter()

        # ── Quality Assessment ──
        qs = self._assess_quality(face_emb, iris_emb, fp_emb)
        result.quality_scores = qs
        result.extraction_latency_ms = 0.0  # Embeddings pre-computed

        if not qs.overall_valid:
            result.error = f"Quality gate failure: {'; '.join(qs.rejection_reasons)}"
            result.total_latency_ms = (time.perf_counter() - t_total_start) * 1000.0
            return result

        # ── Fusion ──
        t_fuse_start = time.perf_counter()
        fused_vec = fuse_concatenate(face_emb, iris_emb, fp_emb)
        t_fuse_end = time.perf_counter()
        result.fusion_latency_ms = (t_fuse_end - t_fuse_start) * 1000.0
        result.fused_vector_dim = len(fused_vec)

        # ── BioHash ──
        t_bh_start = time.perf_counter()
        biocode = cancelable_transforms.biohash_fused(fused_vec, user_seed)
        t_bh_end = time.perf_counter()
        result.biohash_latency_ms = (t_bh_end - t_bh_start) * 1000.0
        result.biohash_vector_dim = len(biocode)

        # ── PQC Encrypt + Store ──
        t_enc_start = time.perf_counter()
        try:
            quality_score = float(np.mean([qs.face_norm, qs.iris_norm, qs.fingerprint_norm]))
            db_result = self.db_manager.enroll_template(
                user_id=user_id,
                user_name=user_name,
                fused_vec=biocode,
                quality_score=quality_score,
            )
            result.template_id = db_result["template_id"]
            result.encryption_latency_ms = db_result["latency_ms"]
            result.success = True
        except Exception as e:
            result.error = f"PQC encryption/storage failed: {e}"

        result.enrollment_latency_ms = (time.perf_counter() - t_enc_start) * 1000.0
        result.total_latency_ms = (time.perf_counter() - t_total_start) * 1000.0
        return result

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 10.2 — Authentication Process Pipeline
    # ══════════════════════════════════════════════════════════════════════════

    def _compute_fractional_hamming_distance(
        self, query_biocode: np.ndarray, enrolled_biocode: np.ndarray
    ) -> float:
        """
        Computes the fractional Hamming distance between two BioHash BioCodes.

        Both vectors are bipolar {+1/√N, −1/√N} after L2 normalization.
        The fractional Hamming distance is the fraction of dimensions where
        the sign bits disagree.

        Returns:
            float in [0.0, 1.0]: 0.0 = identical signs, 1.0 = all signs differ.
        """
        signs_query = np.sign(query_biocode)
        signs_enrolled = np.sign(enrolled_biocode)
        disagreements = np.sum(signs_query != signs_enrolled)
        total_bits = len(query_biocode)
        return float(disagreements) / float(total_bits)

    def _write_audit_log(
        self,
        auth_result: AuthenticationResult,
    ) -> str:
        """
        Creates and signs an ML-DSA-65 audit log entry for non-repudiation.

        Returns:
            Hex-encoded signature string.
        """
        import datetime

        log_entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "user_id": auth_result.user_id,
            "decision": auth_result.decision,
            "hamming_distance": round(auth_result.hamming_distance, 8),
            "cosine_similarity": round(auth_result.cosine_similarity, 8),
            "threshold": auth_result.threshold,
            "total_latency_ms": round(auth_result.total_latency_ms, 4),
            "degraded_mode": auth_result.degraded_mode,
            "degraded_components": auth_result.degraded_components,
            "reason": auth_result.reason,
        }

        # Serialize the log entry deterministically
        log_bytes = json.dumps(log_entry, sort_keys=True, separators=(",", ":")).encode("utf-8")

        # Sign with the system's ML-DSA-65 private key
        signature = self.crypto_engine.dsa_priv.sign(log_bytes)
        sig_hex = signature.hex()

        # Append to audit log file
        log_entry["audit_signature"] = sig_hex
        try:
            with open(self.audit_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, separators=(",", ":")) + "\n")
        except Exception as e:
            print(f"    [WARNING] Failed to write audit log: {e}")

        return sig_hex

    def verify_audit_log_entry(self, log_line: str) -> bool:
        """
        Verifies the ML-DSA-65 signature on an audit log entry.

        Args:
            log_line: A single JSON line from the audit log file.

        Returns:
            True if signature is valid, False otherwise.
        """
        entry = json.loads(log_line)
        sig_hex = entry.pop("audit_signature")
        signature = bytes.fromhex(sig_hex)

        # Reconstruct the signed message
        msg_bytes = json.dumps(entry, sort_keys=True, separators=(",", ":")).encode("utf-8")

        try:
            self.crypto_engine.dsa_pub.verify(signature, msg_bytes)
            return True
        except Exception:
            return False

    def authenticate_user(
        self,
        user_id: str,
        face_path: str,
        iris_path: str,
        fp_path: str,
        user_seed: str,
        threshold: float = DEFAULT_HAMMING_THRESHOLD,
    ) -> AuthenticationResult:
        """
        Full authentication pipeline from raw biometric images.

        Steps:
          1. Biometric Capture & Real-time Quality Assessment
          2. Preprocessing & Feature Extraction (identical models as enrollment)
          3. Feature-Level Fusion (Concatenation → 1536-D L2-normalized)
          4. BioHash Query BioCode generation
          5. Retrieve & Decrypt enrolled template (ML-DSA verified, ML-KEM decapsulated)
          6. Fractional Hamming Distance matching → Accept/Reject
          7. ML-DSA signed audit log

        Args:
            user_id: The claimed identity to verify against.
            face_path: Path to the fresh face image.
            iris_path: Path to the fresh iris image.
            fp_path: Path to the fresh fingerprint image.
            user_seed: User-specific secret token (must match enrollment seed).
            threshold: Hamming distance decision threshold (default 0.35).

        Returns:
            AuthenticationResult with decision, distances, and signed audit trail.
        """
        if self.extractor is None:
            return AuthenticationResult(
                user_id=user_id,
                decision="ERROR",
                reason="Feature extraction models not loaded. Initialize with load_models=True.",
            )

        result = AuthenticationResult(
            user_id=user_id,
            threshold=threshold,
            degraded_mode=self.extractor.is_degraded if self.extractor else False,
            degraded_components=self.extractor.degraded_components if self.extractor else []
        )
        t_total_start = time.perf_counter()

        # ── Step 1-2: Biometric Capture + Feature Extraction ──
        print(f"\n  [Auth] Step 1-2: Extracting features for claimed identity {user_id}...")
        t_extract_start = time.perf_counter()

        try:
            face_emb = self.extractor.extract_face_embedding(face_path)
        except Exception as e:
            face_emb = None
            print(f"    [WARNING] Face extraction failed: {e}")

        try:
            iris_emb = self.extractor.extract_iris_embedding(iris_path)
        except Exception as e:
            iris_emb = None
            print(f"    [WARNING] Iris extraction failed: {e}")

        try:
            fp_emb = self.extractor.extract_fingerprint_embedding(fp_path)
        except Exception as e:
            fp_emb = None
            print(f"    [WARNING] Fingerprint extraction failed: {e}")

        t_extract_end = time.perf_counter()
        result.extraction_latency_ms = (t_extract_end - t_extract_start) * 1000.0

        # ── Quality Assessment ──
        qs = self._assess_quality(face_emb, iris_emb, fp_emb)
        result.query_quality_scores = qs

        if not qs.overall_valid:
            result.decision = "REJECT"
            result.reason = f"Quality gate failure: {'; '.join(qs.rejection_reasons)}"
            result.total_latency_ms = (time.perf_counter() - t_total_start) * 1000.0
            print(f"    [REJECTED] Poor quality sample: {result.reason}")
            result.audit_signature_hex = self._write_audit_log(result)
            return result

        # ── Step 3: Concatenation Fusion ──
        print(f"  [Auth] Step 3: Concatenation fusion → 1536-D...")
        t_fuse_start = time.perf_counter()
        query_fused = fuse_concatenate(face_emb, iris_emb, fp_emb)
        t_fuse_end = time.perf_counter()
        result.fusion_latency_ms = (t_fuse_end - t_fuse_start) * 1000.0

        # ── Step 4: BioHash Query BioCode ──
        print(f"  [Auth] Step 4: BioHash projection → Query BioCode...")
        t_bh_start = time.perf_counter()
        query_biocode = cancelable_transforms.biohash_fused(query_fused, user_seed)
        t_bh_end = time.perf_counter()
        result.biohash_latency_ms = (t_bh_end - t_bh_start) * 1000.0

        # ── Step 5: Retrieve & Decrypt Enrolled Template ──
        print(f"  [Auth] Step 5: Retrieving encrypted enrolled template...")
        t_ret_start = time.perf_counter()
        try:
            enrolled_biocode = self.db_manager.retrieve_and_decrypt_template(user_id)
        except KeyError:
            result.decision = "REJECT"
            result.reason = f"No enrolled template found for user: {user_id}"
            result.total_latency_ms = (time.perf_counter() - t_total_start) * 1000.0
            print(f"    [REJECTED] {result.reason}")
            result.audit_signature_hex = self._write_audit_log(result)
            return result
        except ValueError as e:
            result.decision = "ERROR"
            result.reason = f"Template integrity check failed: {e}"
            result.total_latency_ms = (time.perf_counter() - t_total_start) * 1000.0
            print(f"    [ERROR] {result.reason}")
            result.audit_signature_hex = self._write_audit_log(result)
            return result
        t_ret_end = time.perf_counter()
        result.retrieval_latency_ms = (t_ret_end - t_ret_start) * 1000.0
        print(f"    ✓ Template decrypted & signature verified ({result.retrieval_latency_ms:.4f} ms)")

        # ── Step 6: Matching & Verification ──
        print(f"  [Auth] Step 6: Fractional Hamming distance matching...")
        t_match_start = time.perf_counter()

        hamming_dist = self._compute_fractional_hamming_distance(query_biocode, enrolled_biocode)
        cos_sim = float(np.dot(query_biocode, enrolled_biocode))

        result.hamming_distance = hamming_dist
        result.cosine_similarity = cos_sim

        if hamming_dist < threshold:
            result.decision = "ACCEPT"
            result.reason = f"Hamming distance {hamming_dist:.6f} < threshold {threshold}"
        else:
            result.decision = "REJECT"
            result.reason = f"Hamming distance {hamming_dist:.6f} >= threshold {threshold}"

        t_match_end = time.perf_counter()
        result.match_latency_ms = (t_match_end - t_match_start) * 1000.0

        print(f"    Hamming Distance:    {hamming_dist:.6f}")
        print(f"    Cosine Similarity:   {cos_sim:.6f}")
        print(f"    Decision Threshold:  {threshold}")
        print(f"    ──► Decision:        {result.decision}")

        # ── Step 7: ML-DSA Signed Audit Log ──
        result.total_latency_ms = (time.perf_counter() - t_total_start) * 1000.0
        print(f"  [Auth] Step 7: Writing ML-DSA signed audit log...")
        result.audit_signature_hex = self._write_audit_log(result)
        print(f"    ✓ Audit signature: {result.audit_signature_hex[:64]}...")

        return result

    def authenticate_from_embeddings(
        self,
        user_id: str,
        face_emb: np.ndarray,
        iris_emb: np.ndarray,
        fp_emb: np.ndarray,
        user_seed: str,
        threshold: float = DEFAULT_HAMMING_THRESHOLD,
    ) -> AuthenticationResult:
        """
        Authentication pipeline using pre-computed probe embeddings.
        Skips raw image extraction; performs quality → fusion → BioHash → match → audit.
        """
        result = AuthenticationResult(user_id=user_id, threshold=threshold)
        t_total_start = time.perf_counter()
        result.extraction_latency_ms = 0.0  # Embeddings pre-computed

        # ── Quality ──
        qs = self._assess_quality(face_emb, iris_emb, fp_emb)
        result.query_quality_scores = qs

        if not qs.overall_valid:
            result.decision = "REJECT"
            result.reason = f"Quality gate failure: {'; '.join(qs.rejection_reasons)}"
            result.total_latency_ms = (time.perf_counter() - t_total_start) * 1000.0
            result.audit_signature_hex = self._write_audit_log(result)
            return result

        # ── Fusion ──
        t_fuse_start = time.perf_counter()
        query_fused = fuse_concatenate(face_emb, iris_emb, fp_emb)
        t_fuse_end = time.perf_counter()
        result.fusion_latency_ms = (t_fuse_end - t_fuse_start) * 1000.0

        # ── BioHash ──
        t_bh_start = time.perf_counter()
        query_biocode = cancelable_transforms.biohash_fused(query_fused, user_seed)
        t_bh_end = time.perf_counter()
        result.biohash_latency_ms = (t_bh_end - t_bh_start) * 1000.0

        # ── Retrieve Enrolled Template ──
        t_ret_start = time.perf_counter()
        try:
            enrolled_biocode = self.db_manager.retrieve_and_decrypt_template(user_id)
        except KeyError:
            result.decision = "REJECT"
            result.reason = f"No enrolled template found for user: {user_id}"
            result.total_latency_ms = (time.perf_counter() - t_total_start) * 1000.0
            result.audit_signature_hex = self._write_audit_log(result)
            return result
        except ValueError as e:
            result.decision = "ERROR"
            result.reason = f"Template integrity check failed: {e}"
            result.total_latency_ms = (time.perf_counter() - t_total_start) * 1000.0
            result.audit_signature_hex = self._write_audit_log(result)
            return result
        t_ret_end = time.perf_counter()
        result.retrieval_latency_ms = (t_ret_end - t_ret_start) * 1000.0

        # ── Match ──
        t_match_start = time.perf_counter()
        hamming_dist = self._compute_fractional_hamming_distance(query_biocode, enrolled_biocode)
        cos_sim = float(np.dot(query_biocode, enrolled_biocode))
        result.hamming_distance = hamming_dist
        result.cosine_similarity = cos_sim

        if hamming_dist < threshold:
            result.decision = "ACCEPT"
            result.reason = f"Hamming distance {hamming_dist:.6f} < threshold {threshold}"
        else:
            result.decision = "REJECT"
            result.reason = f"Hamming distance {hamming_dist:.6f} >= threshold {threshold}"

        t_match_end = time.perf_counter()
        result.match_latency_ms = (t_match_end - t_match_start) * 1000.0
        result.total_latency_ms = (time.perf_counter() - t_total_start) * 1000.0

        # ── Audit ──
        result.audit_signature_hex = self._write_audit_log(result)
        return result

    # ══════════════════════════════════════════════════════════════════════════
    # Bulk Operations (Cache-Based)
    # ══════════════════════════════════════════════════════════════════════════

    def enroll_from_dataset(
        self,
        dataset_dir: str = TRAINING_DATA_DIR,
        user_seed_prefix: str = "seed_person_",
    ) -> Dict[str, EnrollmentResult]:
        """
        Bulk-enrolls subjects in real-time from raw biometric images in a dataset directory.
        No pre-cached embeddings are required; dynamic feature extraction runs on raw images.

        Args:
            dataset_dir: Path to the raw dataset folder (e.g. data/Chimeric_Dataset_Noisy/training).
            user_seed_prefix: Prefix for per-user seed generation.

        Returns:
            Dict mapping user_id -> EnrollmentResult.
        """
        print("\n" + "=" * 80)
        print("REAL-TIME BULK ENROLLMENT FROM RAW DATASET IMAGES")
        print("=" * 80)

        if not os.path.exists(dataset_dir):
            print(f"  [ERROR] Dataset directory not found: {dataset_dir}")
            return {}

        person_dirs = sorted([
            d for d in os.listdir(dataset_dir)
            if os.path.isdir(os.path.join(dataset_dir, d)) and d.startswith("Person_")
        ])

        if not person_dirs:
            print(f"  [WARNING] No 'Person_XXX' directories found in {dataset_dir}")
            return {}

        results = {}
        success_count = 0
        fail_count = 0

        for p_dir in person_dirs:
            full_path = os.path.join(dataset_dir, p_dir)
            files = os.listdir(full_path)

            face_files = [f for f in files if ("face" in f.lower()) and f.endswith((".jpg", ".png", ".jpeg"))]
            iris_files = [f for f in files if ("iris" in f.lower()) and f.endswith((".jpg", ".png", ".jpeg"))]
            fp_files = [f for f in files if ("fingerprint" in f.lower() or "fp" in f.lower()) and f.endswith((".jpg", ".png", ".tif", ".bmp"))]

            if not (face_files and iris_files and fp_files):
                print(f"  [SKIP] {p_dir}: missing required modality files")
                continue

            user_id = p_dir
            try:
                num_id = int(p_dir.split("_")[-1])
                user_seed = f"{user_seed_prefix}{num_id:03d}"
            except ValueError:
                user_seed = f"{user_seed_prefix}{user_id}"

            res = self.enroll_user(
                user_id=user_id,
                face_path=os.path.join(full_path, face_files[0]),
                iris_path=os.path.join(full_path, iris_files[0]),
                fp_path=os.path.join(full_path, fp_files[0]),
                user_seed=user_seed,
                user_name=user_id,
            )

            results[user_id] = res
            if res.success:
                success_count += 1
            else:
                fail_count += 1
                print(f"  [FAIL] {user_id}: {res.error}")

        print(f"\nReal-Time Enrollment Complete: {success_count} succeeded, {fail_count} failed out of {len(person_dirs)} total.")
        print(f"Database now contains {self.db_manager.get_enrolled_count()} enrolled templates.")
        return results

    def enroll_from_cache(
        self,
        cache_path: str = DEFAULT_CACHE_PATH,
        user_seed_prefix: str = "seed_person_",
    ) -> Dict[str, EnrollmentResult]:
        """
        Bulk-enrolls all subjects from the transformer_templates_cache.pkl.

        Args:
            cache_path: Path to the pre-computed embeddings cache.
            user_seed_prefix: Prefix for generating per-user seeds (seed = prefix + subject_id).

        Returns:
            Dict mapping user_id → EnrollmentResult.
        """
        print("\n" + "=" * 80)
        print("BULK ENROLLMENT FROM CACHED EMBEDDINGS")
        print("=" * 80)

        with open(cache_path, "rb") as f:
            data = pickle.load(f)

        gallery = data["gallery"]
        subjects = sorted(gallery.keys())
        results = {}
        success_count = 0
        fail_count = 0

        for s in subjects:
            gal = gallery[s]
            face_emb = gal["face"]["embedding"]
            iris_emb = gal["iris"]["embedding"]
            fp_emb = gal["fingerprint"]["embedding"]

            user_id = f"Person_{s:03d}"
            user_seed = f"{user_seed_prefix}{s:03d}"

            res = self.enroll_from_embeddings(
                user_id=user_id,
                face_emb=face_emb,
                iris_emb=iris_emb,
                fp_emb=fp_emb,
                user_seed=user_seed,
                user_name=f"Subject_{s:03d}",
            )

            results[user_id] = res
            if res.success:
                success_count += 1
            else:
                fail_count += 1
                print(f"  [FAIL] {user_id}: {res.error}")

        print(f"\nBulk Enrollment Complete: {success_count} succeeded, {fail_count} failed out of {len(subjects)} total.")
        print(f"Database now contains {self.db_manager.get_enrolled_count()} enrolled templates.")
        return results

    def batch_authenticate_from_cache(
        self,
        cache_path: str = DEFAULT_CACHE_PATH,
        user_seed_prefix: str = "seed_person_",
        num_genuine: int = 5,
        num_impostor: int = 5,
        threshold: float = DEFAULT_HAMMING_THRESHOLD,
    ) -> Tuple[List[AuthenticationResult], List[AuthenticationResult]]:
        """
        Runs a batch of genuine and impostor authentications from cached probe embeddings.

        Args:
            cache_path: Path to the pre-computed embeddings cache.
            user_seed_prefix: Prefix for per-user seed generation.
            num_genuine: Number of genuine authentication attempts to test.
            num_impostor: Number of impostor authentication attempts to test.
            threshold: Hamming distance decision threshold.

        Returns:
            Tuple of (genuine_results, impostor_results).
        """
        print("\n" + "=" * 80)
        print("BATCH AUTHENTICATION FROM CACHED PROBE EMBEDDINGS")
        print("=" * 80)

        with open(cache_path, "rb") as f:
            data = pickle.load(f)

        probes = data["probes"]
        subjects = sorted(probes.keys())

        genuine_results = []
        impostor_results = []

        # ── Genuine Tests ──
        print(f"\n--- Genuine Authentication Tests (Probe from same person) ---")
        for i in range(min(num_genuine, len(subjects))):
            s = subjects[i]
            prb = probes[s]
            user_id = f"Person_{s:03d}"
            user_seed = f"{user_seed_prefix}{s:03d}"

            # Get first probe for each modality
            face_emb = prb["face"][0]["embedding"]
            iris_emb = prb["iris"][0]["embedding"]
            fp_emb = prb["fingerprint"][0]["embedding"]

            print(f"\n  Genuine Test {i+1}: {user_id} probe vs {user_id} enrollment")
            res = self.authenticate_from_embeddings(
                user_id=user_id,
                face_emb=face_emb,
                iris_emb=iris_emb,
                fp_emb=fp_emb,
                user_seed=user_seed,
                threshold=threshold,
            )
            genuine_results.append(res)
            print(f"    Decision: {res.decision} | Hamming: {res.hamming_distance:.6f} | Cosine: {res.cosine_similarity:.6f}")

        # ── Impostor Tests ──
        print(f"\n--- Impostor Authentication Tests (Probe from different person) ---")
        for i in range(min(num_impostor, len(subjects) - 1)):
            # Impostor: use subject i+1's biometrics to claim subject i's identity
            s_claim = subjects[i]
            s_impostor = subjects[i + 1]
            user_id = f"Person_{s_claim:03d}"
            user_seed = f"{user_seed_prefix}{s_claim:03d}"

            prb = probes[s_impostor]
            face_emb = prb["face"][0]["embedding"]
            iris_emb = prb["iris"][0]["embedding"]
            fp_emb = prb["fingerprint"][0]["embedding"]

            print(f"\n  Impostor Test {i+1}: Person_{s_impostor:03d} probe vs {user_id} enrollment")
            res = self.authenticate_from_embeddings(
                user_id=user_id,
                face_emb=face_emb,
                iris_emb=iris_emb,
                fp_emb=fp_emb,
                user_seed=user_seed,
                threshold=threshold,
            )
            impostor_results.append(res)
            print(f"    Decision: {res.decision} | Hamming: {res.hamming_distance:.6f} | Cosine: {res.cosine_similarity:.6f}")

        # ── Summary ──
        gen_correct = sum(1 for r in genuine_results if r.decision == "ACCEPT")
        imp_correct = sum(1 for r in impostor_results if r.decision == "REJECT")
        print(f"\n{'=' * 60}")
        print(f"Batch Authentication Summary:")
        print(f"  Genuine:  {gen_correct}/{len(genuine_results)} correctly accepted")
        print(f"  Impostor: {imp_correct}/{len(impostor_results)} correctly rejected")
        print(f"{'=' * 60}")

        return genuine_results, impostor_results

    # ══════════════════════════════════════════════════════════════════════════
    # 1:N Gallery Search
    # ══════════════════════════════════════════════════════════════════════════

    def retrieve_all_enrolled_templates(self) -> Dict[str, np.ndarray]:
        """
        Retrieves and decrypts ALL enrolled templates from the database.
        Verifies ML-DSA-65 signature for each row.

        Returns:
            Dict mapping user_id -> decrypted 1536-D BioCode vector.
        """
        import sqlite3

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT user_id, kem_ciphertext, nonce, ciphertext, signature
            FROM concat_fused_templates
            WHERE modality = 'concatenated_fused'
        """)
        rows = cursor.fetchall()
        conn.close()

        gallery = {}
        for user_id, kem_ct, nonce, ct, sig in rows:
            payload = {"kem_ciphertext": kem_ct, "nonce": nonce, "ciphertext": ct}
            if not self.crypto_engine.verify_payload(payload, sig):
                print(f"  [WARNING] Signature verification failed for {user_id}, skipping.")
                continue
            decrypted_bytes = self.crypto_engine.envelope_decrypt(payload)
            gallery[user_id] = pickle.loads(decrypted_bytes)

        return gallery

    def identify_against_gallery(
        self,
        face_emb: np.ndarray,
        iris_emb: np.ndarray,
        fp_emb: np.ndarray,
        user_seed_prefix: str = "seed_person_",
        threshold: float = DEFAULT_HAMMING_THRESHOLD,
        top_k: int = 5,
        gallery_cache: Optional[Dict[str, np.ndarray]] = None,
        progress_callback=None,
    ) -> List[dict]:
        """
        1:N Identification: compare query biometrics against ALL enrolled templates.

        For each enrolled user, the query is re-projected through that user's
        BioHash seed (required because BioHash is user-specific / cancelable).

        Args:
            face_emb: 512-D face embedding.
            iris_emb: 512-D iris embedding.
            fp_emb: 512-D fingerprint embedding.
            user_seed_prefix: Prefix for per-user seed derivation.
            threshold: Hamming distance decision threshold.
            top_k: Number of top matches to return.
            gallery_cache: Optional pre-loaded gallery dict (user_id -> BioCode).
            progress_callback: Optional callable(current, total) for progress reporting.

        Returns:
            List of top_k dicts sorted by hamming_distance (ascending), each containing:
              user_id, hamming_distance, cosine_similarity, decision, rank
        """
        # Quality check
        qs = self._assess_quality(face_emb, iris_emb, fp_emb)
        if not qs.overall_valid:
            return [{"user_id": "N/A", "hamming_distance": 1.0, "cosine_similarity": -1.0,
                     "decision": "REJECT", "rank": 1,
                     "reason": f"Quality failure: {'; '.join(qs.rejection_reasons)}"}]

        # Fuse query
        query_fused = fuse_concatenate(face_emb, iris_emb, fp_emb)

        # Load gallery
        if gallery_cache is not None:
            gallery = gallery_cache
        else:
            gallery = self.retrieve_all_enrolled_templates()

        if len(gallery) == 0:
            return [{"user_id": "N/A", "hamming_distance": 1.0, "cosine_similarity": -1.0,
                     "decision": "ERROR", "rank": 1, "reason": "No enrolled templates in database"}]

        # Compare against each enrolled identity
        results = []
        total = len(gallery)
        for idx, (enrolled_uid, enrolled_biocode) in enumerate(sorted(gallery.items())):
            # Extract the numeric ID to derive the per-user seed
            try:
                num_id = int(enrolled_uid.split("_")[-1])
                user_seed = f"{user_seed_prefix}{num_id:03d}"
            except ValueError:
                user_seed = f"{user_seed_prefix}{enrolled_uid}"

            # Project query through this user's BioHash seed
            query_biocode = cancelable_transforms.biohash_fused(query_fused, user_seed)

            # Compute distances
            hamming_dist = self._compute_fractional_hamming_distance(query_biocode, enrolled_biocode)
            cos_sim = float(np.dot(query_biocode, enrolled_biocode))

            decision = "ACCEPT" if hamming_dist < threshold else "REJECT"

            results.append({
                "user_id": enrolled_uid,
                "hamming_distance": hamming_dist,
                "cosine_similarity": cos_sim,
                "decision": decision,
            })

            if progress_callback:
                progress_callback(idx + 1, total)

        # Sort by hamming distance (ascending = best match first)
        results.sort(key=lambda x: x["hamming_distance"])

        # Assign ranks and trim to top_k
        for i, r in enumerate(results):
            r["rank"] = i + 1

        return results[:top_k]

    # ══════════════════════════════════════════════════════════════════════════
    # System Utilities
    # ══════════════════════════════════════════════════════════════════════════

    def get_system_status(self) -> dict:
        """Returns a summary of the system's current state."""
        return {
            "models_loaded": self.extractor is not None,
            "face_model": "ArcFace w600k_r50 ONNX" if (self.extractor and self.extractor.face_session) else ("ResNet-18 fallback" if self.extractor else "Not loaded"),
            "iris_model": "ArcIris iresnet100" if (self.extractor and self.extractor.iris_model) else ("Fallback" if self.extractor else "Not loaded"),
            "fingerprint_model": "DeepPrint TexMinu" if (self.extractor and self.extractor.dp_model) else ("Fallback" if self.extractor else "Not loaded"),
            "db_path": self.db_path,
            "enrolled_templates": self.db_manager.get_enrolled_count(),
            "audit_log_path": self.audit_log_path,
            "kem_algorithm": "ML-KEM-768",
            "dsa_algorithm": "ML-DSA-65",
            "symmetric_cipher": "AES-256-GCM",
        }

    def print_system_status(self):
        """Prints a formatted system status report."""
        status = self.get_system_status()
        print("\n" + "=" * 80)
        print("SYSTEM STATUS REPORT")
        print("=" * 80)
        for key, val in status.items():
            label = key.replace("_", " ").title()
            print(f"  {label:.<40} {val}")
        print("=" * 80)


# ══════════════════════════════════════════════════════════════════════════════
# Full Demonstration
# ══════════════════════════════════════════════════════════════════════════════

def run_full_demo():
    """
    End-to-end demonstration of the Concatenation Fusion Biometric System.

    Test Plan:
      1. System initialization (cache-only mode for speed)
      2. Bulk enrollment of 100 subjects from cached embeddings
      3. Genuine authentication tests (5 subjects)
      4. Impostor authentication tests (5 subjects)
      5. Quality rejection test (None embeddings)
      6. Anti-tampering detection test
      7. Audit log signature verification
      8. Final system status report
    """
    print("\n" + "█" * 80)
    print("█  CONCATENATION FUSION BIOMETRIC SYSTEM — FULL DEMONSTRATION")
    print("█" * 80)

    # ── Remove previous demo artifacts for clean run ──
    system_db = DEFAULT_SYSTEM_DB
    if os.path.exists(system_db):
        os.remove(system_db)
    if os.path.exists(DEFAULT_AUDIT_LOG):
        os.remove(DEFAULT_AUDIT_LOG)

    # ── 1. Initialize System (cache-only mode — no heavy model loading) ──
    system = ConcatFusionBiometricSystem(
        db_path=system_db,
        audit_log_path=DEFAULT_AUDIT_LOG,
        load_models=False,
    )

    # ══════════════════════════════════════════════════════════════════════
    # TEST 1: Bulk Enrollment (100 subjects from cache)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "▓" * 80)
    print("▓  TEST 1: BULK ENROLLMENT FROM CACHED EMBEDDINGS")
    print("▓" * 80)

    enrollment_results = system.enroll_from_cache()

    success_count = sum(1 for r in enrollment_results.values() if r.success)
    fail_count = sum(1 for r in enrollment_results.values() if not r.success)

    print(f"\n  Enrollment Results: {success_count} succeeded, {fail_count} failed")
    assert success_count == 100, f"Expected 100 enrollments, got {success_count}"

    # Print summary statistics for first 3 enrollments
    print("\n  Sample Enrollment Details (first 3):")
    for i, (uid, res) in enumerate(sorted(enrollment_results.items())[:3]):
        print(f"    {uid}: template={res.template_id}, "
              f"fusion={res.fusion_latency_ms:.4f}ms, "
              f"biohash={res.biohash_latency_ms:.4f}ms, "
              f"encrypt={res.encryption_latency_ms:.4f}ms, "
              f"total={res.total_latency_ms:.4f}ms")

    # ══════════════════════════════════════════════════════════════════════
    # TEST 2: Genuine Authentication (5 subjects)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "▓" * 80)
    print("▓  TEST 2: GENUINE AUTHENTICATION (5 SUBJECTS)")
    print("▓" * 80)

    genuine_results, impostor_results = system.batch_authenticate_from_cache(
        num_genuine=5,
        num_impostor=0,
    )

    gen_correct = sum(1 for r in genuine_results if r.decision == "ACCEPT")
    print(f"\n  Genuine Auth Accuracy: {gen_correct}/{len(genuine_results)} = {gen_correct/len(genuine_results)*100:.1f}%")
    assert gen_correct == len(genuine_results), f"Expected all genuine to ACCEPT, got {gen_correct}/{len(genuine_results)}"

    # ══════════════════════════════════════════════════════════════════════
    # TEST 3: Impostor Authentication (5 subjects)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "▓" * 80)
    print("▓  TEST 3: IMPOSTOR AUTHENTICATION (5 SUBJECTS)")
    print("▓" * 80)

    _, impostor_results = system.batch_authenticate_from_cache(
        num_genuine=0,
        num_impostor=5,
    )

    imp_correct = sum(1 for r in impostor_results if r.decision == "REJECT")
    print(f"\n  Impostor Rejection Accuracy: {imp_correct}/{len(impostor_results)} = {imp_correct/len(impostor_results)*100:.1f}%")
    assert imp_correct == len(impostor_results), f"Expected all impostor to REJECT, got {imp_correct}/{len(impostor_results)}"

    # ══════════════════════════════════════════════════════════════════════
    # TEST 4: Quality Rejection (Null Embeddings)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "▓" * 80)
    print("▓  TEST 4: QUALITY REJECTION TEST")
    print("▓" * 80)

    print("\n  Testing with None face embedding...")
    bad_res = system.authenticate_from_embeddings(
        user_id="Person_001",
        face_emb=None,
        iris_emb=np.random.randn(512).astype(np.float32),
        fp_emb=np.random.randn(512).astype(np.float32),
        user_seed="seed_person_001",
    )
    print(f"  Decision: {bad_res.decision} | Reason: {bad_res.reason}")
    assert bad_res.decision == "REJECT", f"Expected REJECT for null face, got {bad_res.decision}"

    print("\n  Testing with NaN iris embedding...")
    nan_emb = np.full(512, np.nan, dtype=np.float32)
    bad_res2 = system.authenticate_from_embeddings(
        user_id="Person_001",
        face_emb=np.random.randn(512).astype(np.float32),
        iris_emb=nan_emb,
        fp_emb=np.random.randn(512).astype(np.float32),
        user_seed="seed_person_001",
    )
    print(f"  Decision: {bad_res2.decision} | Reason: {bad_res2.reason}")
    assert bad_res2.decision == "REJECT", f"Expected REJECT for NaN iris, got {bad_res2.decision}"

    print("\n  Testing with wrong-dimension fingerprint embedding...")
    bad_res3 = system.authenticate_from_embeddings(
        user_id="Person_001",
        face_emb=np.random.randn(512).astype(np.float32),
        iris_emb=np.random.randn(512).astype(np.float32),
        fp_emb=np.random.randn(256).astype(np.float32),
        user_seed="seed_person_001",
    )
    print(f"  Decision: {bad_res3.decision} | Reason: {bad_res3.reason}")
    assert bad_res3.decision == "REJECT", f"Expected REJECT for wrong dim, got {bad_res3.decision}"

    print("\n  ✓ All quality rejection tests passed!")

    # ══════════════════════════════════════════════════════════════════════
    # TEST 5: Anti-Tampering Detection
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "▓" * 80)
    print("▓  TEST 5: ANTI-TAMPERING DETECTION (ML-DSA SIGNATURE)")
    print("▓" * 80)

    # Enroll a test user
    test_vec = np.random.randn(1536).astype(np.float32)
    test_vec /= np.linalg.norm(test_vec)

    # Encrypt and sign
    plaintext_bytes = pickle.dumps(test_vec)
    payload = system.crypto_engine.envelope_encrypt(plaintext_bytes)
    signature = system.crypto_engine.sign_payload(payload)

    # Verify original
    original_valid = system.crypto_engine.verify_payload(payload, signature)
    print(f"  Original payload signature valid: {original_valid}")
    assert original_valid, "Original signature verification should succeed"

    # Tamper with ciphertext
    tampered_payload = {
        "kem_ciphertext": payload["kem_ciphertext"],
        "nonce": payload["nonce"],
        "ciphertext": bytearray(payload["ciphertext"]),
    }
    tampered_payload["ciphertext"][0] ^= 0xFF
    tampered_payload["ciphertext"] = bytes(tampered_payload["ciphertext"])

    tampered_valid = system.crypto_engine.verify_payload(tampered_payload, signature)
    print(f"  Tampered payload signature valid: {tampered_valid}")
    assert not tampered_valid, "Tampered signature verification should fail"

    print("  ✓ Anti-tampering detection working correctly!")

    # ══════════════════════════════════════════════════════════════════════
    # TEST 6: Audit Log Signature Verification
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "▓" * 80)
    print("▓  TEST 6: AUDIT LOG SIGNATURE VERIFICATION")
    print("▓" * 80)

    if os.path.exists(DEFAULT_AUDIT_LOG):
        with open(DEFAULT_AUDIT_LOG, "r", encoding="utf-8") as f:
            lines = f.readlines()

        print(f"  Total audit log entries: {len(lines)}")
        verified_count = 0
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            is_valid = system.verify_audit_log_entry(line)
            if is_valid:
                verified_count += 1
            else:
                print(f"    [FAIL] Entry {i} failed signature verification!")

        print(f"  Verified: {verified_count}/{len(lines)} entries have valid ML-DSA signatures")
        assert verified_count == len(lines), "All audit log entries must have valid signatures"
        print("  ✓ All audit log signatures verified!")
    else:
        print("  [SKIP] No audit log file found")

    # ══════════════════════════════════════════════════════════════════════
    # TEST 7: Unenrolled User Rejection
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "▓" * 80)
    print("▓  TEST 7: UNENROLLED USER REJECTION")
    print("▓" * 80)

    unknown_res = system.authenticate_from_embeddings(
        user_id="Person_999",
        face_emb=np.random.randn(512).astype(np.float32),
        iris_emb=np.random.randn(512).astype(np.float32),
        fp_emb=np.random.randn(512).astype(np.float32),
        user_seed="seed_person_999",
    )
    print(f"  Decision: {unknown_res.decision} | Reason: {unknown_res.reason}")
    assert unknown_res.decision == "REJECT", f"Expected REJECT for unknown user, got {unknown_res.decision}"
    print("  ✓ Unenrolled user correctly rejected!")

    # ══════════════════════════════════════════════════════════════════════
    # Final System Status
    # ══════════════════════════════════════════════════════════════════════
    system.print_system_status()

    # ══════════════════════════════════════════════════════════════════════
    # Final Summary
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "█" * 80)
    print("█  ALL TESTS PASSED — SYSTEM FULLY OPERATIONAL")
    print("█" * 80)
    print(f"""
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  CONCATENATION FUSION BIOMETRIC SYSTEM — DEMO SUMMARY                 │
  ├─────────────────────────────────────────────────────────────────────────┤
  │  Test 1: Bulk Enrollment (100 subjects)              ✓ PASS           │
  │  Test 2: Genuine Authentication (5 subjects)         ✓ PASS           │
  │  Test 3: Impostor Authentication (5 subjects)        ✓ PASS           │
  │  Test 4: Quality Rejection (null/NaN/dim)            ✓ PASS           │
  │  Test 5: Anti-Tampering Detection (ML-DSA)           ✓ PASS           │
  │  Test 6: Audit Log Signature Verification            ✓ PASS           │
  │  Test 7: Unenrolled User Rejection                   ✓ PASS           │
  ├─────────────────────────────────────────────────────────────────────────┤
  │  Enrollment Pipeline:  Capture → Quality → Extract → Fuse (1536-D)    │
  │                        → BioHash → ML-KEM/AES-256 Encrypt → ML-DSA   │
  │                        Sign → SQLite Store                            │
  │                                                                       │
  │  Authentication Pipeline:  Capture → Quality → Extract → Fuse         │
  │                            → BioHash Query BioCode → Decrypt Enrolled │
  │                            Template (ML-DSA Verified) → Fractional    │
  │                            Hamming Distance → Accept/Reject → Signed  │
  │                            Audit Log                                  │
  └─────────────────────────────────────────────────────────────────────────┘
""")


def main():
    """Entry point: runs the full system demonstration."""
    run_full_demo()


if __name__ == "__main__":
    main()

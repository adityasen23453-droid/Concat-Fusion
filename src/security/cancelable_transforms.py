import numpy as np
import hashlib

def _derive_seed_from_token(token: str) -> int:
    """
    Derives a stable 32-bit integer seed from a user-specific token using SHA-256.
    """
    sha = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return int(sha[:8], 16) % (2**32)

# SECURITY NOTE:
# This cache holds per-user cancelable-transform projection matrices (Q) in process memory
# for the duration of the process lifetime to optimize batch evaluations.
# The cache keys must be derived via secure hashing (SHA-256) of the user token, never the
# raw token in plaintext, to prevent leakage via memory dumps, logs, or crash reports.
# This transform material must never be logged, serialized, or shared across tenant/process boundaries.
_PROJECTION_CACHE = {}

def clear_projection_cache() -> None:
    """
    Clears all cached projection matrices from memory to evict sensitive transform materials.
    """
    global _PROJECTION_CACHE
    _PROJECTION_CACHE.clear()

def biohash_face(embedding: np.ndarray, token: str) -> np.ndarray:
    """
    Applies BioHashing to a 512-D ArcFace embedding using a user-specific token.
    
    1. Generates an orthonormal random projection matrix R based on the user seed.
    2. Projects the embedding: p = embedding @ R.
    3. Binarizes the projection to bipolar values: b = +1 if p > 0 else -1.
    4. Normalizes to unit length to preserve cosine similarity range [-1.0, 1.0].
    """
    if embedding is None:
        return None
        
    global _PROJECTION_CACHE
    # Hash the token to derive a secure cache key with modality tag
    cache_key = "face_" + hashlib.sha256(token.encode("utf-8")).hexdigest()
    
    if cache_key in _PROJECTION_CACHE:
        Q = _PROJECTION_CACHE[cache_key]
    else:
        seed = _derive_seed_from_token(token + "_face")
        rng = np.random.default_rng(seed)
        
        # Generate random matrix and make columns orthonormal via QR decomposition
        R = rng.normal(0.0, 1.0, (512, 512))
        Q, _ = np.linalg.qr(R)
        _PROJECTION_CACHE[cache_key] = Q
    
    # Project embedding (should be 512-D)
    flat_emb = embedding.flatten()
    projected = flat_emb @ Q
    
    # Binarize to bipolar values
    bipolar = np.where(projected > 0, 1.0, -1.0).astype(np.float32)
    
    # Normalize to unit length for direct Cosine Similarity (dot product) matching
    norm = np.linalg.norm(bipolar)
    if norm > 1e-6:
        bipolar = bipolar / norm
        
    return bipolar

def biohash_iris(embedding: np.ndarray, token: str) -> np.ndarray:
    """
    Applies BioHashing to a 512-D ArcIris embedding using a user-specific token.
    Uses modality-tagged domain-isolated cache keys.
    """
    if embedding is None:
        return None
        
    global _PROJECTION_CACHE
    cache_key = "iris_" + hashlib.sha256(token.encode("utf-8")).hexdigest()
    
    if cache_key in _PROJECTION_CACHE:
        Q = _PROJECTION_CACHE[cache_key]
    else:
        seed = _derive_seed_from_token(token + "_iris")
        rng = np.random.default_rng(seed)
        
        R = rng.normal(0.0, 1.0, (512, 512))
        Q, _ = np.linalg.qr(R)
        _PROJECTION_CACHE[cache_key] = Q
    
    flat_emb = embedding.flatten()
    projected = flat_emb @ Q
    bipolar = np.where(projected > 0, 1.0, -1.0).astype(np.float32)
    norm = np.linalg.norm(bipolar)
    if norm > 1e-6:
        bipolar = bipolar / norm
        
    return bipolar

def biohash_fingerprint(embedding: np.ndarray, token: str) -> np.ndarray:
    """
    Applies BioHashing to a 512-D DeepPrint fingerprint embedding using a user-specific token.
    Uses modality-tagged domain-isolated cache keys.
    """
    if embedding is None:
        return None
        
    global _PROJECTION_CACHE
    cache_key = "fp_" + hashlib.sha256(token.encode("utf-8")).hexdigest()
    
    if cache_key in _PROJECTION_CACHE:
        Q = _PROJECTION_CACHE[cache_key]
    else:
        seed = _derive_seed_from_token(token + "_fp")
        rng = np.random.default_rng(seed)
        
        R = rng.normal(0.0, 1.0, (512, 512))
        Q, _ = np.linalg.qr(R)
        _PROJECTION_CACHE[cache_key] = Q
    
    flat_emb = embedding.flatten()
    projected = flat_emb @ Q
    bipolar = np.where(projected > 0, 1.0, -1.0).astype(np.float32)
    norm = np.linalg.norm(bipolar)
    if norm > 1e-6:
        bipolar = bipolar / norm
        
    return bipolar

def transform_iris(iris_code_list: list, iris_mask_list: list, token: str) -> tuple:
    """
    Applies a cancelable transform to binary Iris codes and masks.
    To preserve horizontal shifting (axial head tilt correction) and masked Hamming distance:
    1. Generates a column-invariant XOR key (shape: 16, 1, 2) that broadcasts across columns.
    2. Generates row permutations (axis 0) and channel permutations (axis 2).
    3. Performs XOR on the code, and permutes both code and mask.
    """
    if iris_code_list is None or iris_mask_list is None:
        return None, None
        
    seed = _derive_seed_from_token(token)
    rng = np.random.default_rng(seed)
    
    transformed_codes = []
    transformed_masks = []
    
    for code, mask in zip(iris_code_list, iris_mask_list):
        # Code and Mask should be numpy arrays of shape (16, 256, 2)
        h, w, d = code.shape
        
        # 1. Column-invariant XOR key (must be constant along axis 1 to preserve np.roll along columns)
        xor_key = rng.integers(0, 2, size=(h, 1, d)).astype(bool)
        
        # 2. Permutation lists for rows (axis 0) and channels (axis 2)
        row_perm = rng.permutation(h)
        chan_perm = rng.permutation(d)
        
        # Apply XOR and permute code
        xored_code = code ^ xor_key
        t_code = xored_code[row_perm][:, :, chan_perm]
        
        # Apply permutation to mask (no XOR since mask represents noise availability)
        t_mask = mask[row_perm][:, :, chan_perm]
        
        transformed_codes.append(t_code)
        transformed_masks.append(t_mask)
        
    return transformed_codes, transformed_masks

def biohash_fused(embedding: np.ndarray, token: str) -> np.ndarray:
    """
    Applies BioHashing to a 1536-D concatenated feature vector using a user-specific token.
    Uses modality-tagged domain-isolated cache keys ("fused_...").
    """
    if embedding is None:
        return None
        
    global _PROJECTION_CACHE
    cache_key = "fused_" + hashlib.sha256(token.encode("utf-8")).hexdigest()
    
    if cache_key in _PROJECTION_CACHE:
        Q = _PROJECTION_CACHE[cache_key]
    else:
        seed = _derive_seed_from_token(token + "_fused")
        rng = np.random.default_rng(seed)
        
        R = rng.normal(0.0, 1.0, (1536, 1536))
        Q, _ = np.linalg.qr(R)
        _PROJECTION_CACHE[cache_key] = Q
    
    flat_emb = embedding.flatten()
    projected = flat_emb @ Q
    bipolar = np.where(projected > 0, 1.0, -1.0).astype(np.float32)
    norm = np.linalg.norm(bipolar)
    if norm > 1e-6:
        bipolar = bipolar / norm
        
    return bipolar


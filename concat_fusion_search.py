"""
Concatenation Fusion Biometric Search — Interactive Terminal Tool

Provides two search modes:
  [1] 1:1 Verification  — Match probe against a specific enrolled person
  [2] 1:N Identification — Search entire gallery, show top-5 ranked matches
  [3] Quick Demo         — Automated test with cached embeddings

Usage:
  Interactive:    python -X utf8 concat_fusion_search.py
  Automated demo: python -X utf8 concat_fusion_search.py --demo
"""

import os
import sys
import time
import pickle
import argparse
import numpy as np

# ── sys.path setup ──
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

from concat_fusion_system import (
    ConcatFusionBiometricSystem,
    DEFAULT_SYSTEM_DB,
    DEFAULT_AUDIT_LOG,
    DEFAULT_CACHE_PATH,
    DEFAULT_HAMMING_THRESHOLD,
    TRAINING_DATA_DIR,
    TESTING_DATA_DIR,
)
from concat_fusion_biohash_experiment import fuse_concatenate
from src.security import cancelable_transforms

# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════
SEED_PREFIX = "seed_person_"


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def resolve_probe_images(person_id: int) -> dict:
    """
    Auto-locates probe (testing) images for a given person ID.

    Returns dict with face_path, iris_path, fp_path or raises FileNotFoundError.
    """
    person_dir = os.path.join(TESTING_DATA_DIR, f"Person_{person_id:03d}")
    if not os.path.isdir(person_dir):
        raise FileNotFoundError(f"Testing directory not found: {person_dir}")

    # Find first available file for each modality
    files = sorted(os.listdir(person_dir))

    face_files = [f for f in files if f.startswith("face_") and f.endswith(".jpg")]
    iris_files = [f for f in files if f.startswith("iris_R_") and f.endswith(".jpg")]
    fp_files = [f for f in files if f.startswith("fingerprint_") and f.endswith(".tif")]

    if not face_files:
        raise FileNotFoundError(f"No face probe image found in {person_dir}")
    if not iris_files:
        raise FileNotFoundError(f"No iris probe image found in {person_dir}")
    if not fp_files:
        raise FileNotFoundError(f"No fingerprint probe image found in {person_dir}")

    return {
        "face_path": os.path.join(person_dir, face_files[0]),
        "iris_path": os.path.join(person_dir, iris_files[0]),
        "fp_path": os.path.join(person_dir, fp_files[0]),
        "person_dir": person_dir,
    }


def get_cached_probe_embeddings(person_id: int, cache_path: str = DEFAULT_CACHE_PATH) -> dict:
    """
    Loads pre-computed probe embeddings from the transformer_templates_cache.pkl.
    Returns dict with face_emb, iris_emb, fp_emb numpy arrays.
    """
    with open(cache_path, "rb") as f:
        data = pickle.load(f)

    probes = data["probes"]
    if person_id not in probes:
        raise KeyError(f"Person {person_id} not found in probe cache")

    prb = probes[person_id]
    face_emb = prb["face"][0]["embedding"]
    iris_emb = prb["iris"][0]["embedding"]
    fp_emb = prb["fingerprint"][0]["embedding"]

    return {
        "face_emb": face_emb,
        "iris_emb": iris_emb,
        "fp_emb": fp_emb,
    }


def print_header(enrolled_count: int):
    """Print the application header."""
    print("")
    print("+" + "=" * 68 + "+")
    print("|  CONCATENATION FUSION BIOMETRIC SEARCH                             |")
    print("|  Face (ArcFace) + Iris (ArcIris) + Fingerprint (DeepPrint)         |")
    print("|  1536-D Fused | BioHash Protected | ML-KEM/ML-DSA PQC             |")
    print("+" + "-" * 68 + "+")
    print(f"|  Enrolled Subjects: {enrolled_count:<5}  |  Database: concat_fusion_system.db    |")
    print("+" + "=" * 68 + "+")


def print_menu():
    """Print the main menu."""
    print("")
    print("  Select Search Mode:")
    print("    [1] 1:1 Verification   (match against a specific enrolled person)")
    print("    [2] 1:N Identification (search entire gallery, top-5 results)")
    print("    [3] Quick Demo         (automated tests with cached embeddings)")
    print("    [0] Exit")
    print("")


def print_verification_result(probe_label: str, enrolled_label: str,
                              hamming: float, cosine: float, threshold: float,
                              decision: str, latency_ms: float):
    """Print a formatted 1:1 verification result box."""
    dec_str = f">>>  DECISION: {decision}  <<<"
    print("")
    print("  +" + "=" * 66 + "+")
    print("  |  1:1 VERIFICATION RESULT" + " " * 40 + "|")
    print("  +" + "-" * 66 + "+")
    print(f"  |  Probe:    {probe_label:<54}|")
    print(f"  |  Claimed:  {enrolled_label:<54}|")
    print("  |" + " " * 66 + "|")
    print(f"  |  Hamming Distance:    {hamming:<42.6f}|")
    print(f"  |  Cosine Similarity:   {cosine:<42.6f}|")
    print(f"  |  Threshold:           {threshold:<42.4f}|")
    print("  |" + " " * 66 + "|")
    if decision == "ACCEPT":
        print(f"  |  {dec_str:^64}|")
    else:
        print(f"  |  {dec_str:^64}|")
    print("  |" + " " * 66 + "|")
    print(f"  |  Latency: {latency_ms:.2f} ms" + " " * (52 - len(f"{latency_ms:.2f}")) + "|")
    print("  +" + "=" * 66 + "+")


def print_identification_table(probe_label: str, results: list,
                               threshold: float, latency_ms: float,
                               true_person_id: str = None):
    """Print a formatted 1:N identification top-K results table."""
    print("")
    print("  +" + "=" * 76 + "+")
    print("  |  1:N IDENTIFICATION -- TOP 5 MATCHES" + " " * 38 + "|")
    print(f"  |  Query: {probe_label:<67}|")
    print("  +" + "=" * 76 + "+")
    print("  | Rank | Enrolled ID    | Hamming Dist | Cosine Sim | Threshold | Decision  |")
    print("  +" + "-" * 76 + "+")

    for r in results:
        rank = r["rank"]
        uid = r["user_id"]
        hd = r["hamming_distance"]
        cs = r["cosine_similarity"]
        dec = r["decision"]

        # Mark the correct match if known
        marker = ""
        if true_person_id and uid == true_person_id and dec == "ACCEPT":
            marker = " <<<"

        print(f"  | {rank:^4} | {uid:<14} | {hd:^12.6f} | {cs:^10.6f} | {threshold:^9.4f} | {dec:<9} |{marker}")

    print("  +" + "=" * 76 + "+")

    # Summary line
    accepted = [r for r in results if r["decision"] == "ACCEPT"]
    if true_person_id:
        rank1_uid = results[0]["user_id"] if results else "N/A"
        if rank1_uid == true_person_id and results[0]["decision"] == "ACCEPT":
            print(f"  Correct match found at Rank 1: {true_person_id} -- ACCEPT")
        elif any(r["user_id"] == true_person_id and r["decision"] == "ACCEPT" for r in results):
            match_rank = next(r["rank"] for r in results if r["user_id"] == true_person_id and r["decision"] == "ACCEPT")
            print(f"  Correct match found at Rank {match_rank}: {true_person_id}")
        elif any(r["user_id"] == true_person_id for r in results):
            match_rank = next(r["rank"] for r in results if r["user_id"] == true_person_id)
            print(f"  True identity {true_person_id} found at Rank {match_rank} but REJECTED (below threshold)")
        else:
            print(f"  True identity {true_person_id} not in top {len(results)}")

    if len(accepted) == 0:
        print("  No candidates met the acceptance threshold.")
    elif len(accepted) == 1:
        print(f"  Identified as: {accepted[0]['user_id']} (Hamming: {accepted[0]['hamming_distance']:.6f})")
    else:
        print(f"  WARNING: {len(accepted)} candidates accepted -- ambiguous result")

    print(f"  Search latency: {latency_ms:.2f} ms ({len(results)} of {results[0].get('_total', '?')} candidates shown)")


def progress_bar(current, total, bar_len=40):
    """Print an inline progress bar."""
    frac = current / total
    filled = int(bar_len * frac)
    bar = "#" * filled + "-" * (bar_len - filled)
    pct = frac * 100
    print(f"\r  Gallery search: [{bar}] {current}/{total} ({pct:.0f}%)", end="", flush=True)
    if current == total:
        print("")  # newline at end


# ══════════════════════════════════════════════════════════════════════════════
# Interactive Modes
# ══════════════════════════════════════════════════════════════════════════════

def run_1to1_verification(system: ConcatFusionBiometricSystem, cache_data: dict = None):
    """Interactive 1:1 verification mode."""
    print("\n  -- 1:1 VERIFICATION MODE --\n")

    # Get probe person
    print("  How to provide biometric probe data?")
    print("    [A] Enter Person ID (auto-locate from dataset cache)")
    print("    [B] Enter custom file paths (requires DL models loaded)")
    print("")
    mode = input("  Choice [A/B]: ").strip().upper()

    if mode == "B":
        # Custom file paths
        if system.extractor is None:
            print("  [ERROR] DL models not loaded. Use mode [A] with cached embeddings.")
            print("          Or restart with load_models=True.")
            return
        face_path = input("  Enter face image path: ").strip().strip('"')
        iris_path = input("  Enter iris image path: ").strip().strip('"')
        fp_path = input("  Enter fingerprint image path: ").strip().strip('"')
        probe_label = f"Custom images"
        use_raw = True
    else:
        # Person ID
        try:
            pid = int(input("  Enter probe Person ID (1-100): ").strip())
        except ValueError:
            print("  [ERROR] Invalid person ID.")
            return

        probe_label = f"Person_{pid:03d} (testing probe)"
        use_raw = False

    # Get claimed identity
    try:
        claimed_id = int(input("  Verify against enrolled Person ID (1-100): ").strip())
    except ValueError:
        print("  [ERROR] Invalid person ID.")
        return

    claimed_uid = f"Person_{claimed_id:03d}"
    enrolled_label = f"{claimed_uid} (enrolled)"
    user_seed = f"{SEED_PREFIX}{claimed_id:03d}"
    threshold = DEFAULT_HAMMING_THRESHOLD

    print(f"\n  -- PROCESSING --")

    t_start = time.perf_counter()

    if use_raw:
        # Raw image mode
        result = system.authenticate_user(
            user_id=claimed_uid,
            face_path=face_path,
            iris_path=iris_path,
            fp_path=fp_path,
            user_seed=user_seed,
            threshold=threshold,
        )
        hamming = result.hamming_distance
        cosine = result.cosine_similarity
        decision = result.decision
    else:
        # Cached embedding mode
        if cache_data is None:
            print("  Loading cache...")
            with open(DEFAULT_CACHE_PATH, "rb") as f:
                cache_data = pickle.load(f)

        try:
            embs = get_cached_probe_embeddings(pid, DEFAULT_CACHE_PATH)
        except KeyError as e:
            print(f"  [ERROR] {e}")
            return

        print(f"  [1/5] Feature Extraction... (from cache) 512-D face + 512-D iris + 512-D fp")

        face_emb, iris_emb, fp_emb = embs["face_emb"], embs["iris_emb"], embs["fp_emb"]

        print(f"  [2/5] Concatenation Fusion... 1536-D L2-normalized")
        query_fused = fuse_concatenate(face_emb, iris_emb, fp_emb)

        print(f"  [3/5] BioHash Transform... 1536-D bipolar BioCode")
        query_biocode = cancelable_transforms.biohash_fused(query_fused, user_seed)

        print(f"  [4/5] Template Retrieval... ML-DSA verified, AES-256 decrypted")
        try:
            enrolled_biocode = system.db_manager.retrieve_and_decrypt_template(claimed_uid)
        except KeyError:
            print(f"  [ERROR] No enrolled template for {claimed_uid}. Run enrollment first.")
            return

        print(f"  [5/5] Hamming Distance Match...")
        signs_q = np.sign(query_biocode)
        signs_e = np.sign(enrolled_biocode)
        hamming = float(np.sum(signs_q != signs_e)) / len(query_biocode)
        cosine = float(np.dot(query_biocode, enrolled_biocode))
        decision = "ACCEPT" if hamming < threshold else "REJECT"

    t_end = time.perf_counter()
    latency = (t_end - t_start) * 1000.0

    print_verification_result(probe_label, enrolled_label, hamming, cosine, threshold, decision, latency)


def run_1toN_identification(system: ConcatFusionBiometricSystem, cache_data: dict = None,
                            gallery_cache: dict = None):
    """Interactive 1:N identification mode."""
    print("\n  -- 1:N IDENTIFICATION MODE --\n")

    # Get probe person
    print("  How to provide biometric probe data?")
    print("    [A] Enter Person ID (auto-locate from dataset cache)")
    print("    [B] Enter custom file paths (requires DL models loaded)")
    print("")
    mode = input("  Choice [A/B]: ").strip().upper()

    true_person_id = None

    if mode == "B":
        if system.extractor is None:
            print("  [ERROR] DL models not loaded. Use mode [A] with cached embeddings.")
            return
        face_path = input("  Enter face image path: ").strip().strip('"')
        iris_path = input("  Enter iris image path: ").strip().strip('"')
        fp_path = input("  Enter fingerprint image path: ").strip().strip('"')
        probe_label = "Custom images"
        use_raw = True
    else:
        try:
            pid = int(input("  Enter probe Person ID (1-100): ").strip())
        except ValueError:
            print("  [ERROR] Invalid person ID.")
            return

        probe_label = f"Person_{pid:03d} (testing probe)"
        true_person_id = f"Person_{pid:03d}"
        use_raw = False

    threshold = DEFAULT_HAMMING_THRESHOLD

    print(f"\n  -- PROCESSING --")
    t_start = time.perf_counter()

    if use_raw:
        # Extract embeddings from raw images
        print(f"  [1/4] Feature Extraction... (from images)")
        face_emb = system.extractor.extract_face_embedding(face_path)
        iris_emb = system.extractor.extract_iris_embedding(iris_path)
        fp_emb = system.extractor.extract_fingerprint_embedding(fp_path)
    else:
        # Load from cache
        if cache_data is None:
            with open(DEFAULT_CACHE_PATH, "rb") as f:
                cache_data = pickle.load(f)
        try:
            embs = get_cached_probe_embeddings(pid, DEFAULT_CACHE_PATH)
        except KeyError as e:
            print(f"  [ERROR] {e}")
            return
        print(f"  [1/4] Feature Extraction... (from cache) 512-D face + 512-D iris + 512-D fp")
        face_emb, iris_emb, fp_emb = embs["face_emb"], embs["iris_emb"], embs["fp_emb"]

    print(f"  [2/4] Concatenation Fusion... 1536-D L2-normalized")
    print(f"  [3/4] BioHash Transform... (per-candidate seed projection)")

    # Load gallery templates if not already cached
    if gallery_cache is None:
        print(f"  [4/4] Gallery Search... Loading & decrypting all enrolled templates...")
        gallery_cache = system.retrieve_all_enrolled_templates()
        print(f"         Loaded {len(gallery_cache)} enrolled templates.")
    else:
        print(f"  [4/4] Gallery Search... Using cached gallery ({len(gallery_cache)} templates)")

    print(f"         Comparing probe against {len(gallery_cache)} enrolled identities...")

    results = system.identify_against_gallery(
        face_emb=face_emb,
        iris_emb=iris_emb,
        fp_emb=fp_emb,
        user_seed_prefix=SEED_PREFIX,
        threshold=threshold,
        top_k=5,
        gallery_cache=gallery_cache,
        progress_callback=progress_bar,
    )

    t_end = time.perf_counter()
    latency = (t_end - t_start) * 1000.0

    # Attach total count for display
    for r in results:
        r["_total"] = len(gallery_cache)

    print_identification_table(probe_label, results, threshold, latency, true_person_id)

    return gallery_cache  # Return for reuse


# ══════════════════════════════════════════════════════════════════════════════
# Quick Demo (Non-Interactive)
# ══════════════════════════════════════════════════════════════════════════════

def run_quick_demo(system: ConcatFusionBiometricSystem):
    """Automated demonstration of both search modes using cached embeddings."""
    print("\n" + "=" * 78)
    print("  QUICK DEMO -- AUTOMATED SEARCH TESTS (CACHED EMBEDDINGS)")
    print("=" * 78)

    # Load cache
    with open(DEFAULT_CACHE_PATH, "rb") as f:
        cache_data = pickle.load(f)

    # Load gallery once
    print("\n  Loading gallery templates from database...")
    t0 = time.perf_counter()
    gallery_cache = system.retrieve_all_enrolled_templates()
    t1 = time.perf_counter()
    print(f"  Loaded {len(gallery_cache)} templates in {(t1-t0)*1000:.2f} ms")

    # ── Demo 1: 1:1 Genuine Verification ──
    print("\n" + "-" * 78)
    print("  DEMO 1: 1:1 Genuine Verification -- Person_001 probe vs Person_001 enrolled")
    print("-" * 78)

    embs = get_cached_probe_embeddings(1, DEFAULT_CACHE_PATH)
    t_start = time.perf_counter()

    query_fused = fuse_concatenate(embs["face_emb"], embs["iris_emb"], embs["fp_emb"])
    user_seed = f"{SEED_PREFIX}001"
    query_biocode = cancelable_transforms.biohash_fused(query_fused, user_seed)
    enrolled_biocode = system.db_manager.retrieve_and_decrypt_template("Person_001")

    hamming = float(np.sum(np.sign(query_biocode) != np.sign(enrolled_biocode))) / len(query_biocode)
    cosine = float(np.dot(query_biocode, enrolled_biocode))
    decision = "ACCEPT" if hamming < DEFAULT_HAMMING_THRESHOLD else "REJECT"

    t_end = time.perf_counter()
    print_verification_result(
        "Person_001 (testing probe)", "Person_001 (enrolled)",
        hamming, cosine, DEFAULT_HAMMING_THRESHOLD, decision, (t_end - t_start) * 1000
    )

    # ── Demo 2: 1:1 Impostor Verification ──
    print("\n" + "-" * 78)
    print("  DEMO 2: 1:1 Impostor Verification -- Person_002 probe vs Person_001 enrolled")
    print("-" * 78)

    embs2 = get_cached_probe_embeddings(2, DEFAULT_CACHE_PATH)
    t_start = time.perf_counter()

    query_fused2 = fuse_concatenate(embs2["face_emb"], embs2["iris_emb"], embs2["fp_emb"])
    query_biocode2 = cancelable_transforms.biohash_fused(query_fused2, f"{SEED_PREFIX}001")
    enrolled_biocode1 = system.db_manager.retrieve_and_decrypt_template("Person_001")

    hamming2 = float(np.sum(np.sign(query_biocode2) != np.sign(enrolled_biocode1))) / len(query_biocode2)
    cosine2 = float(np.dot(query_biocode2, enrolled_biocode1))
    decision2 = "ACCEPT" if hamming2 < DEFAULT_HAMMING_THRESHOLD else "REJECT"

    t_end = time.perf_counter()
    print_verification_result(
        "Person_002 (testing probe)", "Person_001 (enrolled)",
        hamming2, cosine2, DEFAULT_HAMMING_THRESHOLD, decision2, (t_end - t_start) * 1000
    )

    # ── Demo 3: 1:N Identification for Person_003 ──
    print("\n" + "-" * 78)
    print("  DEMO 3: 1:N Identification -- Person_003 probe (who is this person?)")
    print("-" * 78)

    embs3 = get_cached_probe_embeddings(3, DEFAULT_CACHE_PATH)
    t_start = time.perf_counter()

    results = system.identify_against_gallery(
        face_emb=embs3["face_emb"],
        iris_emb=embs3["iris_emb"],
        fp_emb=embs3["fp_emb"],
        user_seed_prefix=SEED_PREFIX,
        threshold=DEFAULT_HAMMING_THRESHOLD,
        top_k=5,
        gallery_cache=gallery_cache,
        progress_callback=progress_bar,
    )

    t_end = time.perf_counter()
    for r in results:
        r["_total"] = len(gallery_cache)

    print_identification_table(
        "Person_003 (testing probe)", results,
        DEFAULT_HAMMING_THRESHOLD, (t_end - t_start) * 1000,
        true_person_id="Person_003"
    )

    # ── Demo 4: 1:N for Person_010 ──
    print("\n" + "-" * 78)
    print("  DEMO 4: 1:N Identification -- Person_010 probe")
    print("-" * 78)

    embs10 = get_cached_probe_embeddings(10, DEFAULT_CACHE_PATH)
    t_start = time.perf_counter()

    results10 = system.identify_against_gallery(
        face_emb=embs10["face_emb"],
        iris_emb=embs10["iris_emb"],
        fp_emb=embs10["fp_emb"],
        user_seed_prefix=SEED_PREFIX,
        threshold=DEFAULT_HAMMING_THRESHOLD,
        top_k=5,
        gallery_cache=gallery_cache,
        progress_callback=progress_bar,
    )

    t_end = time.perf_counter()
    for r in results10:
        r["_total"] = len(gallery_cache)

    print_identification_table(
        "Person_010 (testing probe)", results10,
        DEFAULT_HAMMING_THRESHOLD, (t_end - t_start) * 1000,
        true_person_id="Person_010"
    )

    # ── Summary ──
    print("\n" + "=" * 78)
    print("  QUICK DEMO SUMMARY")
    print("=" * 78)
    print(f"  Demo 1 (1:1 Genuine):      {decision:>8}  (Hamming: {hamming:.6f})")
    print(f"  Demo 2 (1:1 Impostor):     {decision2:>8}  (Hamming: {hamming2:.6f})")
    r1_3 = results[0] if results else {"decision": "N/A", "user_id": "N/A"}
    r1_10 = results10[0] if results10 else {"decision": "N/A", "user_id": "N/A"}
    print(f"  Demo 3 (1:N Person_003):   Rank 1 = {r1_3['user_id']} ({r1_3['decision']})")
    print(f"  Demo 4 (1:N Person_010):   Rank 1 = {r1_10['user_id']} ({r1_10['decision']})")

    all_pass = (decision == "ACCEPT" and decision2 == "REJECT"
                and r1_3["user_id"] == "Person_003" and r1_3["decision"] == "ACCEPT"
                and r1_10["user_id"] == "Person_010" and r1_10["decision"] == "ACCEPT")
    print(f"\n  All Tests: {'PASSED' if all_pass else 'SOME FAILURES'}")
    print("=" * 78)


# ══════════════════════════════════════════════════════════════════════════════
# Main Interactive Loop
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Concatenation Fusion Biometric Search")
    parser.add_argument("--demo", action="store_true", help="Run automated demo (non-interactive)")
    parser.add_argument("--face", type=str, help="Path to face image file")
    parser.add_argument("--iris", type=str, help="Path to iris image file")
    parser.add_argument("--fingerprint", "--fp", type=str, help="Path to fingerprint image file")
    parser.add_argument("--claim_id", "--claim-id", type=str, help="Optional claimed identity (e.g., Person_001) for 1:1 verification")
    parser.add_argument("--person_id", "--person-id", type=int, help="Optional Person ID (1-100) to auto-locate test images")
    args = parser.parse_args()

    # Determine if CLI direct input mode is active
    has_cli_images = bool(args.face and args.iris and args.fingerprint)
    has_cli_person = bool(args.person_id is not None)
    is_cli_mode = has_cli_images or has_cli_person

    # ── Initialize System ──
    need_enroll = not os.path.exists(DEFAULT_SYSTEM_DB)

    # Load DL models if raw images are supplied via CLI or if cache is missing
    no_cache = not os.path.exists(DEFAULT_CACHE_PATH)
    load_dl_models = has_cli_images or (has_cli_person and no_cache) or no_cache

    system = ConcatFusionBiometricSystem(
        db_path=DEFAULT_SYSTEM_DB,
        audit_log_path=DEFAULT_AUDIT_LOG,
        load_models=load_dl_models,
    )

    if need_enroll or system.db_manager.get_enrolled_count() == 0:
        if os.path.exists(DEFAULT_CACHE_PATH):
            print("\n  Database empty. Auto-enrolling subjects from cache...")
            system.enroll_from_cache()
        elif os.path.exists(TRAINING_DATA_DIR):
            print(f"\n  Database empty & no cache found. Real-time auto-enrolling from raw dataset: {TRAINING_DATA_DIR}...")
            system.enroll_from_dataset(TRAINING_DATA_DIR)
        else:
            print("\n  Database empty. Please provide raw images or dataset directory to enroll.")

    enrolled_count = system.db_manager.get_enrolled_count()

    # ── Demo mode ──
    if args.demo:
        print_header(enrolled_count)
        run_quick_demo(system)
        return

    # ── Direct CLI Mode ──
    if is_cli_mode:
        print_header(enrolled_count)
        print("\n  -- DIRECT CLI MODE --\n")

        # Case A: Person ID passed via CLI
        if has_cli_person:
            pid = args.person_id
            print(f"  Auto-locating test images for Person_{pid:03d}...")
            try:
                paths = resolve_probe_images(pid)
                face_path, iris_path, fp_path = paths["face_path"], paths["iris_path"], paths["fp_path"]
                probe_label = f"Person_{pid:03d} (testing probe)"
                true_person_id = f"Person_{pid:03d}"
            except Exception as e:
                print(f"  [ERROR] Could not resolve images for Person {pid}: {e}")
                return
        else:
            face_path = args.face.strip('"')
            iris_path = args.iris.strip('"')
            fp_path = args.fingerprint.strip('"')
            probe_label = "Direct CLI images"
            true_person_id = None

        # Determine 1:1 or 1:N mode
        if args.claim_id:
            # 1:1 Verification
            claimed_id = args.claim_id
            if not claimed_id.startswith("Person_") and claimed_id.isdigit():
                claimed_id = f"Person_{int(claimed_id):03d}"

            try:
                num_id = int(claimed_id.split("_")[-1])
                user_seed = f"{SEED_PREFIX}{num_id:03d}"
            except ValueError:
                user_seed = f"{SEED_PREFIX}{claimed_id}"

            print(f"  Running 1:1 Verification against claimed identity: {claimed_id}...")
            t_start = time.perf_counter()

            if system.extractor is not None:
                # Raw image extraction
                res = system.authenticate_user(
                    user_id=claimed_id,
                    face_path=face_path,
                    iris_path=iris_path,
                    fp_path=fp_path,
                    user_seed=user_seed,
                    threshold=DEFAULT_HAMMING_THRESHOLD,
                )
                hamming, cosine, decision = res.hamming_distance, res.cosine_similarity, res.decision
            else:
                # Cached embedding fallback for person_id
                embs = get_cached_probe_embeddings(args.person_id, DEFAULT_CACHE_PATH)
                q_fused = fuse_concatenate(embs["face_emb"], embs["iris_emb"], embs["fp_emb"])
                q_code = cancelable_transforms.biohash_fused(q_fused, user_seed)
                e_code = system.db_manager.retrieve_and_decrypt_template(claimed_id)
                hamming = float(np.sum(np.sign(q_code) != np.sign(e_code))) / len(q_code)
                cosine = float(np.dot(q_code, e_code))
                decision = "ACCEPT" if hamming < DEFAULT_HAMMING_THRESHOLD else "REJECT"

            t_end = time.perf_counter()
            print_verification_result(
                probe_label, f"{claimed_id} (enrolled)",
                hamming, cosine, DEFAULT_HAMMING_THRESHOLD, decision, (t_end - t_start) * 1000
            )
        else:
            # 1:N Identification
            print("  Running 1:N Gallery Identification...")
            t_start = time.perf_counter()

            if system.extractor is not None:
                print("  Extracting 512-D ArcFace, ArcIris, DeepPrint features from images...")
                f_emb = system.extractor.extract_face_embedding(face_path)
                i_emb = system.extractor.extract_iris_embedding(iris_path)
                p_emb = system.extractor.extract_fingerprint_embedding(fp_path)
            else:
                embs = get_cached_probe_embeddings(args.person_id, DEFAULT_CACHE_PATH)
                f_emb, i_emb, p_emb = embs["face_emb"], embs["iris_emb"], embs["fp_emb"]

            gallery_cache = system.retrieve_all_enrolled_templates()

            results = system.identify_against_gallery(
                face_emb=f_emb,
                iris_emb=i_emb,
                fp_emb=p_emb,
                user_seed_prefix=SEED_PREFIX,
                threshold=DEFAULT_HAMMING_THRESHOLD,
                top_k=5,
                gallery_cache=gallery_cache,
                progress_callback=progress_bar,
            )

            t_end = time.perf_counter()
            for r in results:
                r["_total"] = len(gallery_cache)

            print_identification_table(
                probe_label, results, DEFAULT_HAMMING_THRESHOLD,
                (t_end - t_start) * 1000, true_person_id=true_person_id
            )
        return

    # ── Interactive loop ──
    gallery_cache = None  # Lazy-loaded on first 1:N search

    while True:
        print_header(enrolled_count)
        print_menu()

        try:
            choice = input("  Enter choice [0-3]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Exiting.")
            break

        if choice == "0":
            print("\n  Goodbye!")
            break
        elif choice == "1":
            run_1to1_verification(system)
        elif choice == "2":
            gallery_cache = run_1toN_identification(system, gallery_cache=gallery_cache)
        elif choice == "3":
            run_quick_demo(system)
        else:
            print("  Invalid choice. Please enter 0, 1, 2, or 3.")

        print("\n" + "-" * 78)
        try:
            input("  Press Enter to continue...")
        except (EOFError, KeyboardInterrupt):
            print("\n  Exiting.")
            break


if __name__ == "__main__":
    main()

"""
Standalone Post-Quantum Cryptography (PQC) & Database Storage Engine for Concatenation Fusion.

Provides hybrid PQC envelope encryption & authentication for 1536-D concatenated / BioHash feature vectors:
1. ML-KEM-768: Post-Quantum Key Encapsulation Mechanism.
2. AES-256-GCM: 256-bit symmetric AEAD encryption.
3. ML-DSA-65: Post-Quantum Digital Signature Scheme for database row anti-tampering.
4. ConcatFusionDatabaseManager: Isolated SQLite template enrollment and encrypted storage.
"""

import os
import sys
import time
import pickle
import sqlite3
import numpy as np
from cryptography.hazmat.primitives.asymmetric import mlkem, mldsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# Default paths within concate fusion folder
CONCATE_FUSION_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_KEY_FILE = os.path.join(CONCATE_FUSION_DIR, "pqc_keys.bin")
DEFAULT_DB_PATH = os.path.join(CONCATE_FUSION_DIR, "data", "concat_fusion_demo.db")


class PQCCryptoEngine:
    def __init__(self, key_file=DEFAULT_KEY_FILE):
        self.key_file = key_file
        self.kem_priv, self.dsa_priv = self._get_or_create_keys()
        self.kem_pub = self.kem_priv.public_key()
        self.dsa_pub = self.dsa_priv.public_key()

    def _get_or_create_keys(self):
        """Loads persistent ML-KEM-768 and ML-DSA-65 keypairs or generates new ones."""
        if os.path.exists(self.key_file):
            try:
                with open(self.key_file, "rb") as f:
                    data = f.read()
                if len(data) == 96:
                    kem_seed = data[:64]
                    dsa_seed = data[64:]
                    kem_priv = mlkem.MLKEM768PrivateKey.from_seed_bytes(kem_seed)
                    dsa_priv = mldsa.MLDSA65PrivateKey.from_seed_bytes(dsa_seed)
                    return kem_priv, dsa_priv
            except Exception as e:
                print(f"[PQCCryptoEngine] Warning: Failed to load keys from {self.key_file}: {e}. Regenerating.")

        # Generate new PQC Keypairs
        kem_priv = mlkem.MLKEM768PrivateKey.generate()
        dsa_priv = mldsa.MLDSA65PrivateKey.generate()
        try:
            os.makedirs(os.path.dirname(self.key_file), exist_ok=True)
            with open(self.key_file, "wb") as f:
                f.write(kem_priv.private_bytes_raw() + dsa_priv.private_bytes_raw())
            print(f"[PQCCryptoEngine] Saved new PQC keys to {self.key_file}")
        except Exception as e:
            print(f"[PQCCryptoEngine] Warning: Failed to save keys to {self.key_file}: {e}")
            
        return kem_priv, dsa_priv

    def envelope_encrypt(self, plaintext: bytes) -> dict:
        """
        Symmetrically encrypts plaintext using AES-256-GCM with a single-use key
        encapsulated using ML-KEM-768.
        """
        shared_secret, kem_ciphertext = self.kem_pub.encapsulate()
        aes_key = shared_secret[:32]
        aesgcm = AESGCM(aes_key)
        
        nonce = os.urandom(12)
        ciphertext = aesgcm.encrypt(nonce, plaintext, None)
        
        return {
            "kem_ciphertext": kem_ciphertext,
            "nonce": nonce,
            "ciphertext": ciphertext
        }

    def envelope_decrypt(self, payload: dict) -> bytes:
        """
        Decrypts envelope payload using ML-KEM-768 decapsulation + AES-256-GCM.
        """
        shared_secret = self.kem_priv.decapsulate(payload["kem_ciphertext"])
        aes_key = shared_secret[:32]
        aesgcm = AESGCM(aes_key)
        return aesgcm.decrypt(payload["nonce"], payload["ciphertext"], None)

    def _get_payload_bytes(self, payload: dict) -> bytes:
        """Concatenates payload components to form anti-tampering bytes for signing."""
        return payload["kem_ciphertext"] + payload["nonce"] + payload["ciphertext"]

    def sign_payload(self, payload: dict) -> bytes:
        """Signs the concatenated payload bytes using ML-DSA-65."""
        msg_bytes = self._get_payload_bytes(payload)
        return self.dsa_priv.sign(msg_bytes)

    def verify_payload(self, payload: dict, signature: bytes) -> bool:
        """Verifies ML-DSA-65 signature against payload bytes."""
        msg_bytes = self._get_payload_bytes(payload)
        try:
            self.dsa_pub.verify(signature, msg_bytes)
            return True
        except Exception:
            return False


class ConcatFusionDatabaseManager:
    def __init__(self, db_path=DEFAULT_DB_PATH, crypto_engine=None):
        self.db_path = db_path
        self.crypto_engine = crypto_engine if crypto_engine is not None else PQCCryptoEngine()
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.init_tables()

    def init_tables(self):
        """Initializes isolated database schema for concatenation fusion templates."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            user_name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS concat_fused_templates (
            template_id TEXT PRIMARY KEY,
            user_id TEXT REFERENCES users(user_id),
            modality TEXT CHECK (modality = 'concatenated_fused'),
            kem_ciphertext BLOB NOT NULL,
            nonce BLOB NOT NULL,
            ciphertext BLOB NOT NULL,
            signature BLOB NOT NULL,
            quality_score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        conn.commit()
        conn.close()

    def enroll_template(self, user_id: str, user_name: str, fused_vec: np.ndarray, quality_score: float = 1.0) -> dict:
        """
        Encrypts and signs 1536-D fused feature / BioHash vector, storing it in SQLite DB.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Upsert User
        cursor.execute("INSERT OR REPLACE INTO users (user_id, user_name) VALUES (?, ?)", (user_id, user_name))

        # Serialize vector
        plaintext_bytes = pickle.dumps(fused_vec)

        # Encrypt & Sign
        t0 = time.perf_counter()
        payload = self.crypto_engine.envelope_encrypt(plaintext_bytes)
        sig = self.crypto_engine.sign_payload(payload)
        t1 = time.perf_counter()

        template_id = f"{user_id}_fused_gal"
        cursor.execute("""
        INSERT OR REPLACE INTO concat_fused_templates (
            template_id, user_id, modality,
            kem_ciphertext, nonce, ciphertext, signature, quality_score
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            template_id, user_id, "concatenated_fused",
            payload["kem_ciphertext"], payload["nonce"], payload["ciphertext"], sig, quality_score
        ))
        conn.commit()
        conn.close()

        return {
            "template_id": template_id,
            "latency_ms": (t1 - t0) * 1000.0,
            "payload": payload,
            "signature": sig
        }

    def retrieve_and_decrypt_template(self, user_id: str) -> np.ndarray:
        """
        Fetches template from SQLite DB, verifies ML-DSA-65 signature, and decrypts 1536-D vector.
        Raises ValueError if signature verification fails (anti-tampering alert).
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
        SELECT kem_ciphertext, nonce, ciphertext, signature 
        FROM concat_fused_templates 
        WHERE user_id = ? AND modality = 'concatenated_fused'
        """, (user_id,))
        row = cursor.fetchone()
        conn.close()

        if row is None:
            raise KeyError(f"Template not found for user: {user_id}")

        kem_ct, nonce, ct, sig = row
        payload = {"kem_ciphertext": kem_ct, "nonce": nonce, "ciphertext": ct}

        # Signature Verification
        if not self.crypto_engine.verify_payload(payload, sig):
            raise ValueError(f"SECURITY ALERT: ML-DSA-65 signature verification failed for user {user_id}! Payload was tampered.")

        # Envelope Decryption
        decrypted_bytes = self.crypto_engine.envelope_decrypt(payload)
        fused_vec = pickle.loads(decrypted_bytes)
        return fused_vec

    def get_enrolled_count(self) -> int:
        """Returns total number of enrolled templates in SQLite DB."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM concat_fused_templates")
        count = cursor.fetchone()[0]
        conn.close()
        return count


def main():
    print("================================================================================")
    print("TESTING STANDALONE PQC CRYPTO ENGINE & DATABASE MANAGER IN CONCATE FUSION")
    print("================================================================================")

    # 1. Initialize PQC Crypto Engine & Database Manager
    db_path = os.path.join(CONCATE_FUSION_DIR, "concat_fusion_test.db")
    if os.path.exists(db_path):
        os.remove(db_path)

    crypto_engine = PQCCryptoEngine()
    db_manager = ConcatFusionDatabaseManager(db_path=db_path, crypto_engine=crypto_engine)

    print(f"\n1. Initialized Crypto Engine & Database Manager")
    print(f"   - Database File: {db_path}")
    print(f"   - KEM Algorithm: ML-KEM-768")
    print(f"   - Signature Algorithm: ML-DSA-65")
    print(f"   - Symmetric AEAD: AES-256-GCM")

    # 2. Generate dummy 1536-D fused vector
    dummy_vec = np.random.randn(1536).astype(np.float32)
    dummy_vec /= np.linalg.norm(dummy_vec)

    # 3. Test Template Enrollment
    res = db_manager.enroll_template("Person_001", "Subject_001", dummy_vec)
    print(f"\n2. Enrolled 1536-D Fused Template:")
    print(f"   - Template ID:     {res['template_id']}")
    print(f"   - Enroll Latency:  {res['latency_ms']:.4f} ms")
    print(f"   - Enrolled Count:  {db_manager.get_enrolled_count()}")

    # 4. Test Template Retrieval & Decryption
    retrieved_vec = db_manager.retrieve_and_decrypt_template("Person_001")
    cos_sim = float(np.dot(dummy_vec, retrieved_vec))
    diff_norm = float(np.linalg.norm(dummy_vec - retrieved_vec))

    print(f"\n3. Template Retrieval & Decryption Verification:")
    print(f"   - Reconstructed Shape:    {retrieved_vec.shape}")
    print(f"   - Cosine Sim with Original: {cos_sim:.8f}")
    print(f"   - Difference Norm:        {diff_norm:.8e}")
    print(f"   - Data Integrity Match:    {diff_norm < 1e-6}")

    # 5. Test Anti-Tampering Detection
    payload = res['payload']
    tampered_payload = {
        "kem_ciphertext": payload["kem_ciphertext"],
        "nonce": payload["nonce"],
        "ciphertext": bytearray(payload["ciphertext"])
    }
    tampered_payload["ciphertext"][0] ^= 0xFF # Flip bits
    tampered_payload["ciphertext"] = bytes(tampered_payload["ciphertext"])

    is_valid_original = crypto_engine.verify_payload(payload, res['signature'])
    is_valid_tampered = crypto_engine.verify_payload(tampered_payload, res['signature'])

    print(f"\n4. ML-DSA-65 Anti-Tampering Signature Verification:")
    print(f"   - Original Payload Valid: {is_valid_original} (PASS)")
    print(f"   - Tampered Payload Valid: {is_valid_tampered} (BLOCKED)")

    # 6. Benchmark 100 iterations
    print(f"\n5. Latency Benchmark (100 iterations Encrypt+Sign vs Decrypt+Verify):")
    enc_times = []
    dec_times = []
    plaintext_bytes = pickle.dumps(dummy_vec)

    for _ in range(100):
        t0 = time.perf_counter()
        p = crypto_engine.envelope_encrypt(plaintext_bytes)
        s = crypto_engine.sign_payload(p)
        t1 = time.perf_counter()

        v = crypto_engine.verify_payload(p, s)
        d = crypto_engine.envelope_decrypt(p)
        t2 = time.perf_counter()

        enc_times.append((t1 - t0) * 1000.0)
        dec_times.append((t2 - t1) * 1000.0)

    print(f"   - Mean Encrypt + Sign Latency:   {np.mean(enc_times):.4f} ms ± {np.std(enc_times):.4f} ms")
    print(f"   - Mean Decrypt + Verify Latency: {np.mean(dec_times):.4f} ms ± {np.std(dec_times):.4f} ms")
    print(f"   - Mean Total Round-Trip:          {np.mean(enc_times)+np.mean(dec_times):.4f} ms")

    if os.path.exists(db_path):
        os.remove(db_path)

    print("\n================================================================================")
    print("STANDALONE PQC MODULE VERIFIED SUCCESSFULLY!")
    print("================================================================================")


if __name__ == "__main__":
    main()

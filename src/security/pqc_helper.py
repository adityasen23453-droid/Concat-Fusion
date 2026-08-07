import os
from cryptography.hazmat.primitives.asymmetric import mlkem, mldsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

KEY_FILE = "pqc_keys.bin"

def _get_or_create_keys():
    if os.path.exists(KEY_FILE):
        try:
            with open(KEY_FILE, "rb") as f:
                data = f.read()
            if len(data) == 96:
                kem_seed = data[:64]
                dsa_seed = data[64:]
                kem_priv = mlkem.MLKEM768PrivateKey.from_seed_bytes(kem_seed)
                dsa_priv = mldsa.MLDSA65PrivateKey.from_seed_bytes(dsa_seed)
                return kem_priv, dsa_priv
        except Exception as e:
            print(f"Warning: Failed to load persistent keys from {KEY_FILE}: {e}. Regenerating.")
            
    # Generate new keys
    kem_priv = mlkem.MLKEM768PrivateKey.generate()
    dsa_priv = mldsa.MLDSA65PrivateKey.generate()
    try:
        with open(KEY_FILE, "wb") as f:
            f.write(kem_priv.private_bytes_raw() + dsa_priv.private_bytes_raw())
    except Exception as e:
        print(f"Warning: Failed to save persistent keys to {KEY_FILE}: {e}")
    return kem_priv, dsa_priv

def generate_kem_keypair():
    """
    Generates or loads an ML-KEM-768 private and public key pair.
    """
    kem_priv, _ = _get_or_create_keys()
    return kem_priv, kem_priv.public_key()

def envelope_encrypt(plaintext: bytes, kem_public_key) -> dict:
    """
    Symmetrically encrypts plaintext using AES-256-GCM with a single-use key
    wrapped using ML-KEM-768.
    
    Returns:
        dict: containing 'kem_ciphertext', 'nonce', and 'ciphertext'.
    """
    # encapsulate returns: (shared_secret, ciphertext)
    shared_secret, kem_ciphertext = kem_public_key.encapsulate()
    
    # AES-256-GCM needs a 32-byte key. The shared secret of ML-KEM-768 is 32 bytes.
    aes_key = shared_secret[:32]
    aesgcm = AESGCM(aes_key)
    
    nonce = os.urandom(12)
    ciphertext = aesgcm.encrypt(nonce, plaintext, None)
    
    return {
        "kem_ciphertext": kem_ciphertext,
        "nonce": nonce,
        "ciphertext": ciphertext
    }

def envelope_decrypt(payload: dict, kem_private_key) -> bytes:
    """
    Decrypts the envelope-encrypted payload using the ML-KEM-768 private key.
    """
    kem_ciphertext = payload["kem_ciphertext"]
    nonce = payload["nonce"]
    ciphertext = payload["ciphertext"]
    
    # decapsulate recovers the original 32-byte shared secret
    shared_secret = kem_private_key.decapsulate(kem_ciphertext)
    aes_key = shared_secret[:32]
    
    aesgcm = AESGCM(aes_key)
    return aesgcm.decrypt(nonce, ciphertext, None)

def generate_signing_keypair():
    """
    Generates or loads an ML-DSA-65 private and public signing key pair.
    """
    _, dsa_priv = _get_or_create_keys()
    return dsa_priv, dsa_priv.public_key()

def get_payload_bytes(payload: dict) -> bytes:
    """
    Concatenates the encrypted payload components to form the message bytes.
    This ensures that signature validation protects against cipher text and nonce tampering.
    """
    return payload["kem_ciphertext"] + payload["nonce"] + payload["ciphertext"]

def sign_payload(payload: dict, signing_private_key) -> bytes:
    """
    Signs the concatenated bytes of the encrypted payload database row components
    using ML-DSA-65.
    """
    message_bytes = get_payload_bytes(payload)
    return signing_private_key.sign(message_bytes)

def verify_payload(payload: dict, signature: bytes, signing_public_key) -> bool:
    """
    Verifies the ML-DSA-65 signature against the concatenated payload bytes.
    Returns True if valid, False otherwise.
    """
    message_bytes = get_payload_bytes(payload)
    try:
        signing_public_key.verify(signature, message_bytes)
        return True
    except Exception:
        return False

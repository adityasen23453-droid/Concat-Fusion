# Concatenation Fusion Biometric System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![PQC](https://img.shields.io/badge/Post--Quantum-ML--KEM--768%20%7C%20ML--DSA--65-success.svg)](https://csrc.nist.gov/projects/post-quantum-cryptography)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An end-to-end, privacy-preserving, post-quantum secure **multimodal biometric system**. It fuses feature representations from **Face (ArcFace)**, **Iris (ArcIris)**, and **Fingerprint (DeepPrint)** into a **1536-D concatenated feature vector**, protected by **BioHash cancelable transformations** and NIST-standardized **Post-Quantum Cryptography (ML-KEM-768 & ML-DSA-65)**.

---

## 🌟 Key Features

- **Multimodal Deep Feature Extraction**:
  - **Face**: ArcFace (ResNet-50 backbone, 512-D, ONNX Runtime)
  - **Iris**: ArcIris (iresnet100 backbone + OpenIris normalization, 512-D, PyTorch)
  - **Fingerprint**: DeepPrint (TexMinu architecture, 512-D, PyTorch)
- **Feature-Level Concatenation Fusion**: Fuses modalities into a unified **1536-D L2-normalized** vector.
- **Cancelable BioHash Template Protection**: Converts fused vectors into **1536-D bipolar BioCodes** ($\pm 1/\sqrt{N}$) using user-seeded pseudo-random orthogonal projections.
- **Post-Quantum Security**:
  - **ML-KEM-768 (Kyber768)** for Key Encapsulation Mechanism.
  - **AES-256-GCM** for authenticated payload encryption.
  - **ML-DSA-65 (Dilithium3)** for digital signatures and audit trail non-repudiation.
- **Interactive Search Terminal**: 1:1 Verification & 1:N Gallery Identification CLI tool.
- **Real-Time Execution Engine**: On-the-fly feature extraction and database enrollment from raw image files without pre-cached embeddings.

---

## 📐 System Architecture Overview

```mermaid
flowchart TD
    subgraph INPUT ["1. Biometric Acquisition Layer"]
        A1["Face Image (JPEG/PNG)"]
        A2["Iris Image (JPEG/PNG)"]
        A3["Fingerprint Image (TIFF/PNG)"]
    end

    subgraph EXTRACTION ["2. Deep Feature Extraction Engine"]
        B1["ArcFace ONNX (512-D)"]
        B2["ArcIris iresnet100 (512-D)"]
        B3["DeepPrint TexMinu (512-D)"]
        QA["Quality Assessment Gate (L2 Norm >= 0.1, Non-NaN)"]
    end

    subgraph FUSION ["3. Concatenation Fusion Layer"]
        C["Concatenation Fusion: Fused Vector (1536-D, L2-Normalized)"]
    end

    subgraph PROTECTION ["4. Cancelable BioHash Layer"]
        D1["User Secret Seed"] --> D2["Pseudo-Random Orthogonal Projection (1536 x 1536)"]
        C --> D3["Sign Quantization: Bipolar BioCode (+1 / -1, 1536-D)"]
        D2 --> D3
    end

    subgraph PQC ["5. Post-Quantum Security Layer"]
        E1["ML-KEM-768 Key Encapsulation"]
        E2["AES-256-GCM Encryption"]
        E3["ML-DSA-65 Signature Generation"]
    end

    subgraph STORAGE ["6. Encrypted Storage & Audit"]
        F1[("SQLite Database (concat_fusion_system.db)")]
        F2["JSONL Audit Log (ML-DSA Signed)"]
    end

    A1 --> B1
    A2 --> B2
    A3 --> B3
    B1 & B2 & B3 --> QA
    QA --> C
    D3 --> E1 --> E2 --> E3
    E3 --> F1
    E3 --> F2
```

---

## 🚀 Quickstart Guide

### 1. Clone Repository
```bash
git clone https://github.com/adityasen23453-droid/Concat-Fusion.git
cd Concat-Fusion
```

### 2. Install Dependencies
```bash
pip install -r src/open-iris/requirements/base.txt
pip install torch torchvision onnxruntime numpy pillow opencv-python
```

### 3. Place Raw Dataset Images
Place dataset folders under `data/Chimeric_Dataset_Noisy/`:
```text
data/Chimeric_Dataset_Noisy/
├── training/
│   ├── Person_001/
│   │   ├── face.jpg
│   │   ├── iris_right.jpg
│   │   └── fingerprint_right_thumb.jpg
│   └── Person_002/
└── testing/
    ├── Person_001/
    └── ...
```

---

## 💻 Terminal Commands & Usage

### A. Launch Interactive Terminal
```bash
python -X utf8 concat_fusion_search.py
```
> **Real-time Enrollment**: If `concat_fusion_system.db` is empty, the terminal automatically extracts features from `data/Chimeric_Dataset_Noisy/training/` and enrolls all subjects in real-time.

### B. Direct CLI 1:1 Verification
```bash
python -X utf8 concat_fusion_search.py --person_id 1 --claim_id Person_001
```

### C. Direct CLI 1:N Identification
```bash
python -X utf8 concat_fusion_search.py --person_id 5
```

### D. Automated System Demo
```bash
python -X utf8 concat_fusion_search.py --demo
```

### E. Population Benchmark Evaluation
```bash
python run_concat_fusion_population_eval.py
```

---

## 📚 Project Documentation

- 📖 [`system_working.md`](system_working.md): Comprehensive step-by-step project guide, execution flow, and mathematical proofs.
- 📐 [`architecture.md`](architecture.md): In-depth architectural specifications and component breakdown.
- 📂 [`files.md`](files.md): Repository file tree documentation and handoff guidelines.

---

## 🛡️ License

Distributed under the MIT License. See `LICENSE` for more information.

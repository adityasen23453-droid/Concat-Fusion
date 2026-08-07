import os
import pickle
import glob
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
import concurrent.futures

# Global variables for workers
iris_model = None
iris_transform = None
iris_rec = None
dp_model = None
binarizer = None

def init_worker():
    global iris_model, iris_transform, iris_rec, dp_model, binarizer
    import sys
    sys.path.append("OpenSourceIrisRecognition/methods/ArcIris/Python")
    import torch
    torch.set_num_threads(1)
    from open_iris_pipeline import OpenIrisPipelineManager
    from modules.network import iresnet100
    from torchvision.transforms import Compose, ToTensor, Normalize
    import flx.models.deep_print_arch as dpa
    from flx.image_processing.binarization import LazilyAllocatedBinarizer

    # Initialize specialized ArcIris Model (ResNet-100)
    iris_model = iresnet100(pretrained=False, progress=False)
    weights_path = "OpenSourceIrisRecognition/methods/ArcIris/Python/models/ResNet100_154000.pt"
    state_dict = torch.load(weights_path, map_location="cpu")
    clean_state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
    iris_model.load_state_dict(clean_state_dict, strict=True)
    iris_model.eval()

    # Freeze all parameters
    for param in iris_model.parameters():
        param.requires_grad = False

    # Define ArcIris expected preprocessing transform
    iris_transform = Compose([
        ToTensor(),
        Normalize(mean=(0.5,), std=(0.5,))
    ])
    
    # Initialize OpenIris Pipeline Manager
    iris_rec = OpenIrisPipelineManager()
    
    # Initialize DeepPrint Model
    dp_weights_path = "models/DeepPrint_Tex_512/best_model.pyt"
    dp_checkpoint = torch.load(dp_weights_path, map_location="cpu")
    dp_model = dpa.DeepPrint_TexMinu(8000, 256, 256)
    dp_model.load_state_dict(dp_checkpoint["model_state_dict"])
    dp_model.eval()

    # Initialize Fingerprint Binarizer
    binarizer = LazilyAllocatedBinarizer(1.8)

def process_subject(i, old_gallery_i, old_probes_i, training_dir, testing_dir):
    global iris_model, iris_transform, iris_rec, dp_model, binarizer
    import os
    import glob
    import numpy as np
    import torch
    from PIL import Image
    
    # Imports inside task block to be safe on spawn environments
    from open_iris_pipeline import BiometricQualityFailure
    from flx.data.image_helpers import pad_and_resize_to_deepprint_input_size
    
    person_name = f"Person_{i:03d}"
    subj_train_dir = os.path.join(training_dir, person_name)
    subj_test_dir = os.path.join(testing_dir, person_name)
    
    gallery_i = {}
    probes_i = {"face": [], "iris": [], "fingerprint": []}
    
    # 1. Face (reused from old cache)
    gallery_i["face"] = old_gallery_i["face"]
    probes_i["face"] = old_probes_i["face"]
    
    # 2. Iris Gallery
    iris_gal_path = os.path.join(subj_train_dir, "iris_right.jpg")
    q_iris = old_gallery_i["iris"]["quality"]
    if os.path.exists(iris_gal_path):
        try:
            _ = iris_rec.generate_biometric_template(iris_gal_path, eye_side="right")
            norm_img = iris_rec.last_normalized_image
            if norm_img is not None:
                import cv2
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                norm_img = clahe.apply(norm_img)
                # Resize from 128x512 to 64x512 expected by ArcIris (width=512, height=64 in PIL)
                im_polar = Image.fromarray(norm_img, "L").resize((512, 64), Image.Resampling.BILINEAR)
                im_tensor = iris_transform(im_polar).unsqueeze(0).repeat(1, 3, 1, 1)
                with torch.no_grad():
                    emb_tensor = iris_model(im_tensor)
                    emb_tensor = torch.nn.functional.normalize(emb_tensor, dim=1)
                    emb_iris = emb_tensor[0].cpu().numpy()
                gallery_i["iris"] = {"embedding": emb_iris, "quality": q_iris}
            else:
                raise ValueError("OpenIris last_normalized_image is None")
        except Exception as e:
            gallery_i["iris"] = {"embedding": None, "quality": 0.0}
    else:
        gallery_i["iris"] = {"embedding": None, "quality": 0.0}
        
    # 3. Fingerprint Gallery
    fp_gal_path = os.path.join(subj_train_dir, "fingerprint_right_thumb.jpg")
    q_fp = old_gallery_i["fingerprint"]["quality"]
    if os.path.exists(fp_gal_path):
        try:
            img_fp = Image.open(fp_gal_path).convert("L")
            img_np = np.array(img_fp)
            preprocessed = pad_and_resize_to_deepprint_input_size(img_np, fill=1.0)
            if binarizer is not None:
                preprocessed = binarizer(preprocessed)
            tensor = torch.stack([preprocessed, preprocessed], dim=0)
            with torch.no_grad():
                out = dp_model(tensor)
                emb_tensor = torch.cat([out.texture_embeddings, out.minutia_embeddings], dim=1)
                emb_fp = emb_tensor[0].cpu().numpy()
            gallery_i["fingerprint"] = {"embedding": emb_fp, "quality": q_fp}
        except Exception as e:
            gallery_i["fingerprint"] = {"embedding": None, "quality": 0.0}
    else:
        gallery_i["fingerprint"] = {"embedding": None, "quality": 0.0}
        
    # 4. Iris Probes
    old_iris_probes = {p["filename"]: p for p in old_probes_i["iris"]}
    iris_files = sorted(glob.glob(os.path.join(subj_test_dir, "iris_R_*.jpg")))
    if len(iris_files) == 0:
        iris_files = sorted(glob.glob(os.path.join(subj_test_dir, "iris_right*.jpg")))
        
    for file_path in iris_files:
        filename = os.path.basename(file_path)
        q = old_iris_probes.get(filename, {}).get("quality", 0.0)
        try:
            _ = iris_rec.generate_biometric_template(file_path, eye_side="right")
            norm_img = iris_rec.last_normalized_image
            if norm_img is not None:
                import cv2
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                norm_img = clahe.apply(norm_img)
                
                # Check if gallery embedding is available for alignment
                gal_emb = gallery_i["iris"].get("embedding")
                if gal_emb is not None:
                    # Search for best shift to align with gallery
                    shifts = [-24, -16, -8, 0, 8, 16, 24]
                    best_sim = -1.0
                    best_emb = None
                    
                    for s in shifts:
                        rolled = np.roll(norm_img, s, axis=1)
                        im_polar = Image.fromarray(rolled, "L").resize((512, 64), Image.Resampling.BILINEAR)
                        im_tensor = iris_transform(im_polar).unsqueeze(0).repeat(1, 3, 1, 1)
                        with torch.no_grad():
                            emb_tensor = iris_model(im_tensor)
                            emb_tensor = torch.nn.functional.normalize(emb_tensor, dim=1)
                            emb = emb_tensor[0].cpu().numpy()
                        sim = float(np.dot(gal_emb, emb))
                        if sim > best_sim:
                            best_sim = sim
                            best_emb = emb
                    emb_iris = best_emb
                else:
                    # Fallback to unshifted
                    im_polar = Image.fromarray(norm_img, "L").resize((512, 64), Image.Resampling.BILINEAR)
                    im_tensor = iris_transform(im_polar).unsqueeze(0).repeat(1, 3, 1, 1)
                    with torch.no_grad():
                        emb_tensor = iris_model(im_tensor)
                        emb_tensor = torch.nn.functional.normalize(emb_tensor, dim=1)
                        emb_iris = emb_tensor[0].cpu().numpy()
                        
                probes_i["iris"].append({
                    "embedding": emb_iris, "quality": q, "filename": filename
                })
            else:
                raise ValueError("OpenIris last_normalized_image is None")
        except Exception as e:
            probes_i["iris"].append({
                "embedding": None, "quality": 0.0, "filename": filename
            })
            
    # 5. Fingerprint Probes
    old_fp_probes = {p["filename"]: p for p in old_probes_i["fingerprint"]}
    for idx in range(1, 9):
        fp_probe_path = os.path.join(subj_test_dir, f"fingerprint_{idx}.tif")
        if not os.path.exists(fp_probe_path):
            fp_probe_path = os.path.join(subj_test_dir, f"fingerprint_right_thumb{idx}.jpg")
            
        filename = os.path.basename(fp_probe_path)
        q = old_fp_probes.get(filename, {}).get("quality", 0.0)
        if os.path.exists(fp_probe_path):
            try:
                img_fp = Image.open(fp_probe_path).convert("L")
                img_np = np.array(img_fp)
                preprocessed = pad_and_resize_to_deepprint_input_size(img_np, fill=1.0)
                if binarizer is not None:
                    preprocessed = binarizer(preprocessed)
                tensor = torch.stack([preprocessed, preprocessed], dim=0)
                with torch.no_grad():
                    out = dp_model(tensor)
                    emb_tensor = torch.cat([out.texture_embeddings, out.minutia_embeddings], dim=1)
                    emb_fp = emb_tensor[0].cpu().numpy()
                probes_i["fingerprint"].append({
                    "embedding": emb_fp, "quality": q, "filename": filename
                })
            except Exception as e:
                probes_i["fingerprint"].append({
                    "embedding": None, "quality": 0.0, "filename": filename
                })
        else:
            probes_i["fingerprint"].append({
                "embedding": None, "quality": 0.0, "filename": filename
            })
            
    return i, gallery_i, probes_i

def main():
    print("==================================================")
    # 1. Load existing cache to reuse Face embeddings and quality scores
    old_cache_path = "multimodal_templates_cache.pkl"
    if not os.path.exists(old_cache_path):
        print(f"Error: {old_cache_path} not found. Please run multimodal_fusion_pipeline.py first.")
        return
        
    with open(old_cache_path, "rb") as f:
        old_cache = pickle.load(f)
        
    old_gallery = old_cache["gallery"]
    old_probes = old_cache["probes"]
    
    new_gallery = {}
    new_probes = {}
    
    dataset_dir = "Chimeric_Dataset_Noisy"
    test_dir = os.path.join(dataset_dir, "testing")
    train_dir = os.path.join(dataset_dir, "training")
    
    # Validate weights path exists before launching
    dp_weights_path = "models/DeepPrint_Tex_512/best_model.pyt"
    if not os.path.exists(dp_weights_path):
        print(f"Error: DeepPrint weights not found at {dp_weights_path}")
        return

    print("Extracting features using parallel ProcessPoolExecutor (6 workers)...")
    
    # We use 6 workers (half of our 12 CPU cores)
    num_workers = 6
    
    subjects = list(range(1, 101))
    
    # Submit tasks
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker) as executor_pool:
        # Pass required dicts to tasks
        futures = {
            executor_pool.submit(
                process_subject, i, old_gallery[i], old_probes[i], train_dir, test_dir
            ): i for i in subjects
        }
        
        # Track with progress bar
        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Subjects feature extraction"):
            i, gal_i, prb_i = fut.result()
            new_gallery[i] = gal_i
            new_probes[i] = prb_i
                
    # 5. Save cached features
    new_cache_path = "transformer_templates_cache.pkl"
    print(f"Caching features to {new_cache_path}...")
    with open(new_cache_path, "wb") as f:
        pickle.dump({"gallery": new_gallery, "probes": new_probes}, f)
    print("Feature caching complete.")

if __name__ == "__main__":
    main()

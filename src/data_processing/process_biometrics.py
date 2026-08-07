import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm
from insightface.app import FaceAnalysis
import insightface.utils.face_align as face_align

# Input paths
FACE_DIR = "Face dataset/extracted"
IRIS_DIR = "iris dataset/extracted/CASIA-Iris-Thousand"
FINGERPRINT_DIR = "FVC2002/Dbs/Db1_a"

# Output paths
OUTPUT_DIR = "Chimeric_Dataset"
TRAIN_DIR = os.path.join(OUTPUT_DIR, "training")
TEST_DIR = os.path.join(OUTPUT_DIR, "testing")

SERIAL_FACE_DIR = "Used_Faces"
SERIAL_IRIS_DIR = "Used_Irises"
SERIAL_FINGERPRINT_DIR = "Used_Fingerprints"

# Clean and ensure output directories exist
for d in [TRAIN_DIR, TEST_DIR, SERIAL_FACE_DIR, SERIAL_IRIS_DIR, SERIAL_FINGERPRINT_DIR]:
    if os.path.exists(d):
        print(f"Clearing directory: {d}")
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)

# Initialize InsightFace FaceAnalysis app
print("Initializing InsightFace FaceAnalysis app (detection module)...")
face_app = FaceAnalysis(allowed_modules=['detection'])
face_app.prepare(ctx_id=-1, det_size=(640, 640))

# Scan for the first 100 completely glasses-free subjects in CASIA-Iris-Thousand
print("Scanning CASIA-Iris-Thousand for 100 completely glasses-free subjects...")
glasses_free_subs = []
for sub in range(1000):
    folder = os.path.join(IRIS_DIR, f"{sub:03d}", "R")
    if os.path.exists(folder):
        files = os.listdir(folder)
        has_glasses = False
        for f in files:
            p = os.path.join(folder, f)
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Glasses threshold
                if np.sum(img >= 220) > 200:
                    has_glasses = True
                    break
        if not has_glasses and len(files) >= 4:
            glasses_free_subs.append(sub)
            if len(glasses_free_subs) == 100:
                break

if len(glasses_free_subs) < 100:
    print(f"WARNING: Found only {len(glasses_free_subs)} glasses-free subjects.")
else:
    print(f"Found 100 glasses-free subjects successfully: {glasses_free_subs[:10]}...")


def find_face_image_path(subject_id, img_idx):
    filename = f"{subject_id}-{img_idx}.jpg"
    for part in ["originalimages_part1", "originalimages_part2", "originalimages_part3", "originalimages_part4"]:
        path = os.path.join(FACE_DIR, part, filename)
        if os.path.exists(path):
            return path
    return None

def preprocess_face(path):
    """
    Processes face image using InsightFace deep-learning models:
    - Runs SCRFD detector to find face and 5landmarks.
    - Selects the target face maximizing det_score * bbox_area.
    - Performs 5-point Affine Similarity Transformation to output 112x112.
    - Normalizes pixels to [-1.0, 1.0] and transposes to CHW (3x112x112) format.
    - Returns (aligned_bgr, normalized_chw).
    """
    img = cv2.imread(path)
    if img is None:
        print(f"CRITICAL ERROR: Failed to read image {path}.")
        return None, None

    # Step 2: Detect faces and 5-point landmarks
    faces = face_app.get(img)
    
    # Step 3: Selection & Filtering
    if len(faces) > 0:
        # Select face that maximizes det_score * bbox_area
        best_face = None
        max_metric = -1.0
        
        for face in faces:
            bbox = face.bbox
            det_score = face.det_score
            box_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            metric = det_score * box_area
            if metric > max_metric:
                max_metric = metric
                best_face = face
                
        # Step 4: 5-Point Affine Similarity Transformation
        kps = best_face.kps
        # norm_crop returns 112x112 BGR/RGB warped crop
        aligned_bgr = face_align.norm_crop(img, landmark=kps, image_size=112)
        
    else:
        # Step 3 (Exception Block): Graceful fallback to center crop
        print(f"CRITICAL ERROR: No faces detected in {path}. Falling back to center crop.")
        h, w = img.shape[:2]
        side = min(h, w)
        cx, cy = w // 2, h // 2
        crop = img[cy - side // 2 : cy + side // 2, cx - side // 2 : cx + side // 2]
        aligned_bgr = cv2.resize(crop, (112, 112))
        
    # Step 5: Tensor Normalization & Transposition
    # Convert BGR crop to RGB
    aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
    
    # Convert to float32 and map to [-1, 1] via: Pixel_norm = (Pixel - 127.5) / 128.0
    img_norm = (aligned_rgb.astype(np.float32) - 127.5) / 128.0
    
    # Transpose layout from HWC to CHW
    img_norm_transposed = np.transpose(img_norm, (2, 0, 1))
    
    return aligned_bgr, img_norm_transposed

def select_faces_for_subject(subject_id):
    """
    Selects 1 training face: exactly the 11th image (frontal neutral)
    Selects 3 testing faces:
      - 1st: exactly the 12th image (frontal smiling)
      - 2nd & 3rd: best-quality remaining images
    """
    # Training face is always index 11
    train_path = find_face_image_path(subject_id, "11")
    if not train_path:
        return None, []
        
    # Testing face 1 is always index 12
    test_1_path = find_face_image_path(subject_id, "12")
    
    # Select other testing faces from the remaining 12 images
    test_candidates = []
    for i in range(1, 15):
        idx = f"{i:02d}"
        if idx in ["11", "12"]:
            continue
        path = find_face_image_path(subject_id, idx)
        if path:
            # We rank candidates by detecting face with Haar/InsightFace.
            # To avoid slow app.get calls on all 12 rotated images,
            # we use a fast Haar Cascade face check, fallback to score ranking.
            # Let's just use the image index or compute a fast score
            # to select which rotated images to use.
            # Actually, using face_app is fine since it takes only ~5ms per image.
            # Let's run face_app to get detection score.
            img_c = cv2.imread(path)
            faces_c = face_app.get(img_c)
            score = faces_c[0].det_score if len(faces_c) > 0 else 0.0
            test_candidates.append((path, idx, score))
            
    # Sort remaining test candidates by detection score descending
    test_candidates.sort(key=lambda x: x[2], reverse=True)
    
    test_paths = []
    if test_1_path:
        test_paths.append(test_1_path)
    
    # Add next 2 best from candidates
    for cand in test_candidates[:2]:
        test_paths.append(cand[0])
        
    return train_path, test_paths

def analyze_iris(path):
    """
    Scores iris image based on:
    - Sharpness (Laplacian variance)
    - Absence of eyeglasses/heavy glare (pixels with intensity >= 220 must be <= 200)
    - Pupil presence and circularity
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return -99999, True
        
    sharpness = cv2.Laplacian(img, cv2.CV_64F).var()
    
    # 1. Glasses/Glare Detection:
    # A sensitive threshold: count pixels with intensity >= 220.
    # Normal eyes have only small LED spots (< 150 pixels), glasses/glare covers large areas (> 200 pixels).
    tot_bright = np.sum(img >= 220)
    has_glare_or_glasses = tot_bright > 200
            
    # 2. Pupil Detection:
    # Pupil is the darkest central region
    _, thresh_pupil = cv2.threshold(img, 45, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh_pupil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    pupil_detected = False
    pupil_circularity = 0.0
    for contour in contours:
        area = cv2.contourArea(contour)
        if 800 <= area <= 15000:
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
                if circularity > 0.6:
                    pupil_detected = True
                    pupil_circularity = max(pupil_circularity, circularity)
                    
    # Calculate score
    score = sharpness
    if pupil_detected:
        score += 1000.0 + (pupil_circularity * 500.0)
    if has_glare_or_glasses:
        score -= 10000.0 # Heavy penalty to filter out images with glasses
        
    return score, has_glare_or_glasses

def select_irises_for_subject(subject_id):
    """
    Selects 1 training right eye iris (best of S5{subject}R00 to R09, avoiding glasses)
    Selects 3 testing right eye irises (next 3 best, avoiding glasses)
    """
    candidates = []
    for i in range(10):
        filename = f"S5{subject_id:03d}R{i:02d}.jpg"
        path = os.path.join(IRIS_DIR, f"{subject_id:03d}", "R", filename)
        if os.path.exists(path):
            score, has_glare = analyze_iris(path)
            candidates.append((path, score, has_glare))
            
    if not candidates:
        return None, []
        
    # Sort candidates by score descending
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    train_path = candidates[0][0]
    test_paths = [x[0] for x in candidates[1:4]]
    
    return train_path, test_paths

def analyze_fingerprint(path):
    """
    Scores fingerprint based on:
    - Orientation alignment (using contour ellipse fitting to find deviation from vertical 0/180 axis)
    - Centering (distance from image center)
    - Completeness (foreground area ratio)
    - Ridge patterns / contrast (standard deviation of foreground pixels)
    - Sharpness (Laplacian variance)
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return -999999
        
    h, w = img.shape
    
    # Segment foreground (ridges are dark, background is clean white)
    _, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return -999999
        
    largest = max(contours, key=cv2.contourArea)
    
    # 1. Orientation Angle Check:
    # Fit ellipse to find rotation angle (0 to 180 degrees)
    # A vertically aligned fingerprint (90-degree aligned, upright) has an angle close to 0 or 180.
    angle_diff = 90.0
    if len(largest) >= 5:
        ellipse = cv2.fitEllipse(largest)
        angle = ellipse[2]
        angle_diff = min(angle, 180 - angle)
    else:
        angle_diff = 90.0 # Heavy penalty
        
    # 2. Centering Check:
    # Centroid of the largest contour should be horizontally centered
    M = cv2.moments(largest)
    if M["m00"] > 0:
        cx = M["m10"] / M["m00"]
        center_diff = abs(cx - w/2)
    else:
        center_diff = w
        
    # 3. Completeness (Foreground area ratio)
    foreground_mask = img < 210
    foreground_ratio = np.sum(foreground_mask) / img.size
    
    # 4. Contrast
    foreground_pixels = img[foreground_mask]
    if len(foreground_pixels) > 500:
        std_dev = np.std(foreground_pixels)
    else:
        std_dev = 0
        
    # 5. Sharpness
    sharpness = cv2.Laplacian(img, cv2.CV_64F).var()
    
    # Base Quality
    base_score = sharpness * std_dev * foreground_ratio
    
    # Penalize rotation (if angle deviates from vertical by > 15 degrees)
    rotation_penalty = 0.0
    if angle_diff > 15.0:
        rotation_penalty += (angle_diff - 15.0) * 1500.0 # Proportional penalty
        
    # Penalize off-centering (if center offsets from middle by > 15% of width)
    centering_penalty = 0.0
    max_allowed_offset = w * 0.15
    if center_diff > max_allowed_offset:
        centering_penalty += (center_diff - max_allowed_offset) * 500.0
        
    # Combined score
    score = base_score - rotation_penalty - centering_penalty
    return score

def select_fingerprints_for_subject(subject_id):
    """
    Selects 1 training fingerprint (best of subject_1.tif to _8.tif, avoiding rotation)
    Selects 3 testing fingerprints (next 3 best, avoiding rotation)
    """
    candidates = []
    for i in range(1, 9):
        filename = f"{subject_id}_{i}.tif"
        path = os.path.join(FINGERPRINT_DIR, filename)
        if os.path.exists(path):
            score = analyze_fingerprint(path)
            candidates.append((path, score))
            
    if not candidates:
        return None, []
        
    # Sort candidates by score descending
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    train_path = candidates[0][0]
    test_paths = [x[0] for x in candidates[1:4]]
    
    return train_path, test_paths

def process_subject(chimeric_id, face_sub_id, iris_sub_id, finger_sub_id):
    """
    Processes a single chimeric individual:
    - Selects face, iris, fingerprint
    - Runs Face Preprocessing Pipeline (alignment, cropping, normalization)
    - Saves face.jpg and face.npy (as well as face X.jpg and face X.npy)
    - Saves to Used_Faces, Used_Irises, Used_Fingerprints
    """
    # 1. Select files
    face_train, face_tests = select_faces_for_subject(face_sub_id)
    iris_train, iris_tests = select_irises_for_subject(iris_sub_id)
    finger_train, finger_tests = select_fingerprints_for_subject(finger_sub_id)
    
    if not (face_train and iris_train and finger_train):
        print(f"Skipping Subject Person_{chimeric_id:03d} due to missing data.")
        return False
        
    # 2. Setup subject directories
    person_name = f"Person_{chimeric_id:03d}"
    train_person_dir = os.path.join(TRAIN_DIR, person_name)
    test_person_dir = os.path.join(TEST_DIR, person_name)
    
    os.makedirs(train_person_dir, exist_ok=True)
    os.makedirs(test_person_dir, exist_ok=True)
    
    # 3. Process and Copy Training Face (Use Original Face, No Preprocessing/Normalization)
    shutil.copy(face_train, os.path.join(train_person_dir, "face.jpg"))
    shutil.copy(face_train, os.path.join(SERIAL_FACE_DIR, f"{person_name}_train.jpg"))
        
    # Copy Iris
    shutil.copy(iris_train, os.path.join(train_person_dir, "iris_right.jpg"))
    shutil.copy(iris_train, os.path.join(SERIAL_IRIS_DIR, f"{person_name}_train.jpg"))
    
    # Copy Fingerprint (convert tif to jpg)
    img_finger = cv2.imread(finger_train)
    cv2.imwrite(os.path.join(train_person_dir, "fingerprint_right_thumb.jpg"), img_finger)
    cv2.imwrite(os.path.join(SERIAL_FINGERPRINT_DIR, f"{person_name}_train.jpg"), img_finger)
    
    # 4. Copy Testing Face (without preprocessing/normalization/npy)
    for i, path in enumerate(face_tests):
        idx = i + 1
        dest_test_path = os.path.join(test_person_dir, f"face {idx}.jpg")
        dest_serial_path = os.path.join(SERIAL_FACE_DIR, f"{person_name}_test{idx}.jpg")
        shutil.copy(path, dest_test_path)
        shutil.copy(path, dest_serial_path)
        
    # Copy Iris testing: iris_right1.jpg, iris_right2.jpg, iris_right3.jpg (no space)
    for i, path in enumerate(iris_tests):
        idx = i + 1
        shutil.copy(path, os.path.join(test_person_dir, f"iris_right{idx}.jpg"))
        shutil.copy(path, os.path.join(SERIAL_IRIS_DIR, f"{person_name}_test{idx}.jpg"))
        
    # Copy Fingerprint testing: fingerprint_right_thumb1.jpg, ... (no space, convert tif to jpg)
    for i, path in enumerate(finger_tests):
        idx = i + 1
        img_f = cv2.imread(path)
        cv2.imwrite(os.path.join(test_person_dir, f"fingerprint_right_thumb{idx}.jpg"), img_f)
        cv2.imwrite(os.path.join(SERIAL_FINGERPRINT_DIR, f"{person_name}_test{idx}.jpg"), img_f)
        
    return True

if __name__ == "__main__":
    print("Processing Revised Chimeric Biometric Dataset creation...")
    success_count = 0
    for i in tqdm(range(1, 101), desc="Generating Subjects"):
        chimeric_id = i
        face_sub_id = i
        iris_sub_id = glasses_free_subs[i - 1] if i - 1 < len(glasses_free_subs) else (i - 1)
        finger_sub_id = i
        
        success = process_subject(chimeric_id, face_sub_id, iris_sub_id, finger_sub_id)
        if success:
            success_count += 1
            
    print(f"Dataset creation complete! Generated {success_count} subjects successfully.")

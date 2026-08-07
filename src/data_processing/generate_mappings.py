import os
import cv2
import numpy as np
from tqdm import tqdm

# Input paths
FACE_DIR = "Face dataset/extracted"
IRIS_DIR = "iris dataset/extracted/CASIA-Iris-Thousand"
FINGERPRINT_DIR = "FVC2002/Dbs/Db1_a"

# 1. Scan for glasses-free subjects (same logic as process_biometrics.py)
def scan_glasses_free_subjects():
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
                    if np.sum(img >= 220) > 200:
                        has_glasses = True
                        break
            if not has_glasses and len(files) >= 4:
                glasses_free_subs.append(sub)
                if len(glasses_free_subs) == 100:
                    break
    return glasses_free_subs

# 2. Iris Analysis Logic (same as process_biometrics.py)
def analyze_iris(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return -99999, True
    sharpness = cv2.Laplacian(img, cv2.CV_64F).var()
    tot_bright = np.sum(img >= 220)
    has_glare_or_glasses = tot_bright > 200
    
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
                    
    score = sharpness
    if pupil_detected:
        score += 1000.0 + (pupil_circularity * 500.0)
    if has_glare_or_glasses:
        score -= 10000.0
    return score, has_glare_or_glasses

def select_irises_for_subject(subject_id):
    candidates = []
    for i in range(10):
        filename = f"S5{subject_id:03d}R{i:02d}.jpg"
        path = os.path.join(IRIS_DIR, f"{subject_id:03d}", "R", filename)
        if os.path.exists(path):
            score, has_glare = analyze_iris(path)
            candidates.append((path, score, has_glare))
    if not candidates:
        return None, []
    candidates.sort(key=lambda x: x[1], reverse=True)
    train_path = candidates[0][0]
    test_paths = [x[0] for x in candidates[1:4]]
    return train_path, test_paths

# 3. Fingerprint Analysis Logic (same as process_biometrics.py)
def analyze_fingerprint(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return -999999
    h, w = img.shape
    _, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return -999999
    largest = max(contours, key=cv2.contourArea)
    
    angle_diff = 90.0
    if len(largest) >= 5:
        ellipse = cv2.fitEllipse(largest)
        angle = ellipse[2]
        angle_diff = min(angle, 180 - angle)
        
    M = cv2.moments(largest)
    if M["m00"] > 0:
        cx = M["m10"] / M["m00"]
        center_diff = abs(cx - w/2)
    else:
        center_diff = w
        
    foreground_mask = img < 210
    foreground_ratio = np.sum(foreground_mask) / img.size
    
    foreground_pixels = img[foreground_mask]
    if len(foreground_pixels) > 500:
        std_dev = np.std(foreground_pixels)
    else:
        std_dev = 0
        
    sharpness = cv2.Laplacian(img, cv2.CV_64F).var()
    base_score = sharpness * std_dev * foreground_ratio
    
    rotation_penalty = 0.0
    if angle_diff > 15.0:
        rotation_penalty += (angle_diff - 15.0) * 1500.0
        
    centering_penalty = 0.0
    max_allowed_offset = w * 0.15
    if center_diff > max_allowed_offset:
        centering_penalty += (center_diff - max_allowed_offset) * 500.0
        
    score = base_score - rotation_penalty - centering_penalty
    return score

def select_fingerprints_for_subject(subject_id):
    candidates = []
    for i in range(1, 9):
        filename = f"{subject_id}_{i}.tif"
        path = os.path.join(FINGERPRINT_DIR, filename)
        if os.path.exists(path):
            score = analyze_fingerprint(path)
            candidates.append((path, score))
    if not candidates:
        return None, []
    candidates.sort(key=lambda x: x[1], reverse=True)
    train_path = candidates[0][0]
    test_paths = [x[0] for x in candidates[1:4]]
    return train_path, test_paths

# 4. Face Size Mapping Logic (finds original FEI image by exact size matching)
def find_face_original_filename(subject_id, chimeric_filename, target_size_bytes):
    # Find all original parts files for this subject
    prefix = f"{subject_id}-"
    for part in ["originalimages_part1", "originalimages_part2", "originalimages_part3", "originalimages_part4"]:
        part_dir = os.path.join(FACE_DIR, part)
        if os.path.exists(part_dir):
            for file in os.listdir(part_dir):
                if file.startswith(prefix) and file.lower().endswith(".jpg"):
                    file_path = os.path.join(part_dir, file)
                    if os.path.getsize(file_path) == target_size_bytes:
                        return file
    return None

def main():
    print("Scanning glasses-free subjects...")
    glasses_free_subs = scan_glasses_free_subjects()
    print(f"Loaded {len(glasses_free_subs)} glasses-free subjects.")
    
    face_rows = []
    iris_rows = []
    finger_rows = []
    
    for i in tqdm(range(1, 101), desc="Mapping Chimeric Dataset"):
        person_name = f"Person_{i:03d}"
        
        # --- FACE MAPPINGS ---
        # Training face: always mapped to the 11th image of FEI subject i
        face_rows.append((person_name, "training", "face.jpg", f"{i}-11.jpg"))
        
        # Testing faces: face 1.jpg, face 2.jpg, face 3.jpg (if exists)
        test_dir = f"Chimeric_Dataset/testing/{person_name}"
        for idx in range(1, 4):
            test_face_name = f"face {idx}.jpg"
            test_face_path = os.path.join(test_dir, test_face_name)
            if os.path.exists(test_face_path):
                size = os.path.getsize(test_face_path)
                orig_file = find_face_original_filename(i, test_face_name, size)
                if orig_file:
                    face_rows.append((person_name, "testing", test_face_name, orig_file))
                else:
                    face_rows.append((person_name, "testing", test_face_name, f"FEI Subject {i} (unmapped size: {size})"))
                    
        # --- IRIS MAPPINGS ---
        iris_sub_id = glasses_free_subs[i - 1]
        iris_train, iris_tests = select_irises_for_subject(iris_sub_id)
        if iris_train:
            iris_rows.append((person_name, "training", "iris_right.jpg", os.path.basename(iris_train)))
        for idx, path in enumerate(iris_tests):
            iris_rows.append((person_name, "testing", f"iris_right{idx+1}.jpg", os.path.basename(path)))
            
        # --- FINGERPRINT MAPPINGS ---
        finger_train, finger_tests = select_fingerprints_for_subject(i)
        if finger_train:
            finger_rows.append((person_name, "training", "fingerprint_right_thumb.jpg", os.path.basename(finger_train)))
        for idx, path in enumerate(finger_tests):
            finger_rows.append((person_name, "testing", f"fingerprint_right_thumb{idx+1}.jpg", os.path.basename(path)))
            
    # 5. Write Markdown Files
    with open("face_mappings.md", "w") as f:
        f.write("# Chimeric Dataset Face Mappings\n\n")
        f.write("This file maps each chimeric subject's face images back to the original FEI Face Database files.\n\n")
        f.write("| Chimeric Subject | Subset | Chimeric Filename | Original Filename |\n")
        f.write("| --- | --- | --- | --- |\n")
        for row in face_rows:
            f.write(f"| {row[0]} | {row[1]} | {row[2]} | `{row[3]}` |\n")
            
    with open("iris_mappings.md", "w") as f:
        f.write("# Chimeric Dataset Iris Mappings\n\n")
        f.write("This file maps each chimeric subject's iris images back to the original CASIA-Iris-Thousand files (using 100 glasses-free subjects).\n\n")
        f.write("| Chimeric Subject | Subset | Chimeric Filename | Original Filename |\n")
        f.write("| --- | --- | --- | --- |\n")
        for row in iris_rows:
            f.write(f"| {row[0]} | {row[1]} | {row[2]} | `{row[3]}` |\n")
            
    with open("fingerprint_mappings.md", "w") as f:
        f.write("# Chimeric Dataset Fingerprint Mappings\n\n")
        f.write("This file maps each chimeric subject's fingerprint images back to the original FVC2002 Database files.\n\n")
        f.write("| Chimeric Subject | Subset | Chimeric Filename | Original Filename |\n")
        f.write("| --- | --- | --- | --- |\n")
        for row in finger_rows:
            f.write(f"| {row[0]} | {row[1]} | {row[2]} | `{row[3]}` |\n")
            
    print("Done! Mapping files generated successfully.")

if __name__ == "__main__":
    main()

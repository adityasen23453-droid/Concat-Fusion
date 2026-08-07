import os
import numpy as np
import cv2
from tqdm import tqdm
from insightface.app import FaceAnalysis

def main():
    print("Initializing InsightFace FaceAnalysis app (detection + recognition modules)...")
    # Allowed modules must include both detection and recognition
    app = FaceAnalysis(allowed_modules=['detection', 'recognition'])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    print("Models loaded successfully!")

    train_dir = "Chimeric_Dataset/training"
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Chimeric training directory not found at {train_dir}")

    embeddings = []
    subject_ids = []
    image_ids = []

    processed_count = 0
    failed_count = 0

    print("Extracting training face embeddings...")
    for i in tqdm(range(1, 101), desc="Extracting Face Embeddings"):
        person_name = f"Person_{i:03d}"
        # We extract from the preprocessed face.jpg in the Chimeric Dataset
        img_path = os.path.join(train_dir, person_name, "face.jpg")
        
        if not os.path.exists(img_path):
            print(f"\nWarning: Missing training face image for {person_name} at {img_path}")
            failed_count += 1
            continue

        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Failed to read image at {img_path}")

            # Run detection + recognition
            faces = app.get(img)

            if len(faces) == 0:
                # If detection fails on the already cropped 112x112 image, 
                # we can try to extract features directly from the image without detection 
                # or fallback to the original face image.
                # Let's try a fallback: load the original face image from Face dataset
                # subject i neutral frontal face index 11
                print(f"\nDetection failed on cropped image for {person_name}, attempting fallback to original image...")
                orig_found = False
                for part in ["originalimages_part1", "originalimages_part2", "originalimages_part3", "originalimages_part4"]:
                    orig_path = os.path.join("Face dataset/extracted", part, f"{i}-11.jpg")
                    if os.path.exists(orig_path):
                        orig_img = cv2.imread(orig_path)
                        orig_faces = app.get(orig_img)
                        if len(orig_faces) > 0:
                            # Use the best face detected on original image
                            best_face = max(orig_faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
                            embedding = best_face.embedding
                            orig_found = True
                            break
                if not orig_found:
                    raise ValueError("Face detection failed on both cropped and original images.")
            else:
                # Select the face
                best_face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
                embedding = best_face.embedding

            # L2 Normalize the embedding vector
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding_l2 = embedding / norm
            else:
                embedding_l2 = np.zeros_like(embedding)

            embeddings.append(embedding_l2)
            subject_ids.append(person_name)
            image_ids.append("face.jpg")

            processed_count += 1

        except Exception as e:
            print(f"\nError processing {img_path}: {e}")
            failed_count += 1

    print(f"\nExtraction complete! Processed: {processed_count}, Failed: {failed_count}")

    if processed_count > 0:
        # Save NPZ file
        output_filename = "fei_face_embeddings.npz"
        np.savez_compressed(
            output_filename,
            embeddings=np.array(embeddings, dtype=np.float32),
            subject_ids=np.array(subject_ids, dtype=object),
            image_ids=np.array(image_ids, dtype=object)
        )
        print(f"Embeddings successfully saved to {output_filename}!")

        # Verification check
        data = np.load(output_filename, allow_pickle=True)
        print("\nVerification of saved NPZ:")
        print("  Keys:", list(data.keys()))
        print("  embeddings shape:", data['embeddings'].shape)
        print("  subject_ids shape:", data['subject_ids'].shape)
        print("  image_ids shape:", data['image_ids'].shape)
    else:
        print("\nError: No embeddings were extracted.")

if __name__ == "__main__":
    main()

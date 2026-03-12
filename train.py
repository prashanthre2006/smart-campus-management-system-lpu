import os
import cv2
import numpy as np

from utils import ensure_dir, save_labels


DATA_DIR = "data"
MODEL_PATH = os.path.join("models", "face_model.yml")
LABELS_PATH = os.path.join("models", "labels.json")


def main():
    if not os.path.exists(DATA_DIR):
        raise SystemExit("No data directory found. Run enroll.py first.")

    faces = []
    labels = []
    id_to_name = {}

    current_id = 0
    for name in sorted(os.listdir(DATA_DIR)):
        person_dir = os.path.join(DATA_DIR, name)
        if not os.path.isdir(person_dir):
            continue

        id_to_name[str(current_id)] = name
        for file_name in os.listdir(person_dir):
            if not file_name.lower().endswith(".jpg"):
                continue
            img_path = os.path.join(person_dir, file_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            faces.append(img)
            labels.append(current_id)

        current_id += 1

    if not faces:
        raise SystemExit("No training images found. Enroll at least one person.")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))

    ensure_dir(os.path.dirname(MODEL_PATH))
    recognizer.save(MODEL_PATH)
    save_labels(LABELS_PATH, id_to_name)

    print(f"Trained model saved to {MODEL_PATH}")
    print(f"Labels saved to {LABELS_PATH}")


if __name__ == "__main__":
    main()

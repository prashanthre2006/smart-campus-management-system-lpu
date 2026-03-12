import json
import os
import cv2
import numpy as np

from config import DATA_DIR, EMBEDDING_DB_PATH, EMBEDDING_MODEL_PATH
from utils import ensure_dir
from dnn_recognizer import DNNRecognizer


def main():
    if not os.path.exists(EMBEDDING_MODEL_PATH):
        raise SystemExit("Embedding model file not found.")

    recognizer = DNNRecognizer(EMBEDDING_MODEL_PATH, EMBEDDING_DB_PATH)
    embeddings = {}

    for name in sorted(os.listdir(DATA_DIR)):
        person_dir = os.path.join(DATA_DIR, name)
        if not os.path.isdir(person_dir):
            continue

        vecs = []
        for file_name in os.listdir(person_dir):
            if not file_name.lower().endswith(".jpg"):
                continue
            img_path = os.path.join(person_dir, file_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            vecs.append(recognizer._embedding(img))

        if vecs:
            embeddings[name] = np.mean(np.array(vecs), axis=0).tolist()

    ensure_dir(os.path.dirname(EMBEDDING_DB_PATH))
    with open(EMBEDDING_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(embeddings, f)

    print(f"Saved embeddings to {EMBEDDING_DB_PATH}")


if __name__ == "__main__":
    main()

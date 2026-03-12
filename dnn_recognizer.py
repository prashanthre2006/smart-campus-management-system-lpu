import json
import os
from typing import Dict, Tuple

import cv2
import numpy as np


class DNNRecognizer:
    def __init__(self, model_path: str, db_path: str):
        self.model_path = model_path
        self.db_path = db_path
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.embeddings = self._load_db(db_path)

    def _load_db(self, db_path: str) -> Dict[str, list]:
        if not os.path.exists(db_path):
            return {}
        with open(db_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        face = cv2.resize(face_bgr, (112, 112))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        blob = face.astype("float32") / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)
        return blob

    def _embedding(self, face_bgr: np.ndarray) -> np.ndarray:
        blob = self._preprocess(face_bgr)
        self.net.setInput(blob)
        emb = self.net.forward().flatten().astype("float32")
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    def predict(self, face_bgr: np.ndarray, threshold: float = 0.6) -> Tuple[str, float]:
        if not self.embeddings:
            return "Unknown", 0.0

        emb = self._embedding(face_bgr)
        best_name = "Unknown"
        best_score = -1.0
        for name, vec in self.embeddings.items():
            score = self._cosine(emb, np.array(vec, dtype="float32"))
            if score > best_score:
                best_score = score
                best_name = name

        if best_score < threshold:
            return "Unknown", best_score
        return best_name, best_score

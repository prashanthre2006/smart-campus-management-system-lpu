import csv
import json
import os
from datetime import datetime
from typing import Tuple

import cv2
import numpy as np


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_labels(labels_path: str, id_to_name: dict) -> None:
    ensure_dir(os.path.dirname(labels_path))
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(id_to_name, f, indent=2)


def load_labels(labels_path: str) -> dict:
    if not os.path.exists(labels_path):
        return {}
    with open(labels_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def mark_attendance(name: str, csv_path: str) -> None:
    ensure_dir(os.path.dirname(csv_path) or ".")
    today = datetime.now().strftime("%Y-%m-%d")
    rows = []
    exists = os.path.exists(csv_path)

    if exists:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)

        for row in rows[1:]:
            if len(row) >= 2 and row[0] == name and row[1].startswith(today):
                return

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["name", "timestamp"])
        writer.writerow([name, get_timestamp()])


def list_today_attendance(csv_path: str) -> list:
    if not os.path.exists(csv_path):
        return []
    today = datetime.now().strftime("%Y-%m-%d")
    names = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
        for row in rows[1:]:
            if len(row) >= 2 and row[1].startswith(today):
                names.append(row[0])
    return names


def is_blurry(gray_img, blur_threshold: float) -> Tuple[bool, float]:
    variance = cv2.Laplacian(gray_img, cv2.CV_64F).var()
    return variance < blur_threshold, variance


def brightness_level(gray_img) -> float:
    return float(np.mean(gray_img))


def enhance_low_light(gray_img):
    # Improve low-light faces using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_img)

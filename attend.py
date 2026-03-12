import os
import time

import cv2

from utils import load_labels, mark_attendance, list_today_attendance


MODEL_PATH = os.path.join("models", "face_model.yml")
LABELS_PATH = os.path.join("models", "labels.json")
ATTENDANCE_CSV = "attendance.csv"
UNKNOWN_DIR = os.path.join("logs", "unknown")


def main():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
        raise SystemExit("Model not found. Run train.py first.")

    labels = load_labels(LABELS_PATH)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Unable to open camera.")

    face_cascade = cv2.CascadeClassifier(
        os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    )

    last_marked = {}
    cooldown_seconds = 60
    confidence_threshold = 70
    required_hits = 3
    max_unknown_saves = 5
    unknown_saved = 0
    recent_hits = {}
    last_status = "Waiting..."
    last_status_time = 0.0

    print("Starting attendance. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_roi = gray[y : y + h, x : x + w]
            label_id, confidence = recognizer.predict(face_roi)
            name = labels.get(str(label_id), "Unknown")

            if confidence <= confidence_threshold and name != "Unknown":
                recent_hits[name] = recent_hits.get(name, 0) + 1
                if recent_hits[name] >= required_hits:
                    now = time.time()
                    last_time = last_marked.get(name, 0)
                    if now - last_time > cooldown_seconds:
                        mark_attendance(name, ATTENDANCE_CSV)
                        last_marked[name] = now
                        last_status = f"Marked: {name}"
                        last_status_time = now
                    recent_hits[name] = 0
                color = (0, 255, 0)
            else:
                name = "Unknown"
                color = (0, 0, 255)
                if unknown_saved < max_unknown_saves:
                    os.makedirs(UNKNOWN_DIR, exist_ok=True)
                    snap_path = os.path.join(UNKNOWN_DIR, f"unknown_{int(time.time())}.jpg")
                    cv2.imwrite(snap_path, frame)
                    unknown_saved += 1
                if time.time() - last_status_time > 2:
                    last_status = "Unknown person"

            label_text = f"{name} ({int(confidence)})"
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
                label_text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

        # Status and today's attendance overlay
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 70), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"Status: {last_status}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        today_names = list_today_attendance(ATTENDANCE_CSV)
        shown = ", ".join(today_names[-5:]) if today_names else "None"
        cv2.putText(
            frame,
            f"Today: {shown}",
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            2,
        )

        cv2.imshow("Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

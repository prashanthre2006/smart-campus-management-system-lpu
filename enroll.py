import argparse
import os
import time

import cv2

from utils import ensure_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Enroll a person by capturing face samples.")
    parser.add_argument("--name", required=True, help="Person name")
    parser.add_argument("--samples", type=int, default=30, help="Number of face samples to capture")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    return parser.parse_args()


def main():
    args = parse_args()
    name = args.name.strip()
    if not name:
        raise SystemExit("Name is required.")

    save_dir = os.path.join("data", name)
    ensure_dir(save_dir)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Unable to open camera.")

    face_cascade = cv2.CascadeClassifier(
        os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    )

    count = 0
    last_saved = 0.0

    print(f"Enrolling {name}. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if count < args.samples and (time.time() - last_saved) > 0.2:
                face_roi = gray[y : y + h, x : x + w]
                face_path = os.path.join(save_dir, f"{count:03d}.jpg")
                cv2.imwrite(face_path, face_roi)
                count += 1
                last_saved = time.time()

        cv2.putText(
            frame,
            f"Samples: {count}/{args.samples}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Enroll", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if count >= args.samples:
            time.sleep(0.5)
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {count} samples to {save_dir}")


if __name__ == "__main__":
    main()

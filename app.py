import os
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
from datetime import datetime

import cv2
import openpyxl

from alerts import send_alert
from config import (
    ADMIN_PASSWORD,
    ADMIN_USERNAME,
    ATTENDANCE_CSV,
    DATA_DIR,
    DB_PATH,
    MODEL_DIR,
    PERIODS,
    BRIGHTNESS_MAX,
    BRIGHTNESS_MIN,
    BLUR_THRESHOLD,
    CONFIDENCE_THRESHOLD,
    REQUIRED_HITS,
    USE_DNN_RECOG,
    EMBEDDING_MODEL_PATH,
    EMBEDDING_DB_PATH,
)
from db import (
    add_attendance,
    add_student,
    attendance_exists,
    get_student_by_name,
    get_students,
    init_db,
    list_attendance_for_date,
)
from utils import brightness_level, ensure_dir, is_blurry, load_labels, mark_attendance
from dnn_recognizer import DNNRecognizer


MODEL_PATH = os.path.join(MODEL_DIR, "face_model.yml")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")


class AttendanceApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Face Attendance System")
        self.geometry("900x600")
        self.resizable(False, False)

        ensure_dir(DATA_DIR)
        ensure_dir(MODEL_DIR)
        init_db(DB_PATH)

        self.current_user = None
        self.attend_thread = None
        self.attend_stop = threading.Event()
        self.camera_index = tk.IntVar(value=0)
        self.camera_label_var = tk.StringVar(value="Camera 0")
        self.period_var = tk.StringVar(value=PERIODS[0])
        self.status_var = tk.StringVar(value="Ready")
        self.classroom_mode = tk.BooleanVar(value=True)
        self.camera_indices = []
        self.use_quality_filter = tk.BooleanVar(value=True)
        self.use_dnn = tk.BooleanVar(value=USE_DNN_RECOG)

        self._build_login()

    def _build_login(self):
        self.login_frame = tk.Frame(self)
        self.login_frame.pack(expand=True)

        tk.Label(self.login_frame, text="Admin Login", font=("Segoe UI", 18, "bold")).pack(pady=20)

        form = tk.Frame(self.login_frame)
        form.pack()

        tk.Label(form, text="Username").grid(row=0, column=0, padx=8, pady=6, sticky="e")
        tk.Label(form, text="Password").grid(row=1, column=0, padx=8, pady=6, sticky="e")

        self.user_entry = tk.Entry(form, width=30)
        self.pass_entry = tk.Entry(form, width=30, show="*")
        self.user_entry.grid(row=0, column=1, padx=8, pady=6)
        self.pass_entry.grid(row=1, column=1, padx=8, pady=6)

        tk.Button(self.login_frame, text="Login", width=20, command=self._login).pack(pady=12)

        self.user_entry.insert(0, ADMIN_USERNAME)
        self.pass_entry.focus_set()

    def _login(self):
        user = self.user_entry.get().strip()
        pw = self.pass_entry.get().strip()
        if user == ADMIN_USERNAME and pw == ADMIN_PASSWORD:
            self.current_user = user
            self.login_frame.destroy()
            self._build_main()
        else:
            messagebox.showerror("Login failed", "Invalid credentials.")

    def _build_main(self):
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        self.tab_enroll = ttk.Frame(self.notebook)
        self.tab_train = ttk.Frame(self.notebook)
        self.tab_attend = ttk.Frame(self.notebook)
        self.tab_reports = ttk.Frame(self.notebook)
        self.tab_settings = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_enroll, text="Enrollment")
        self.notebook.add(self.tab_train, text="Training")
        self.notebook.add(self.tab_attend, text="Attendance")
        self.notebook.add(self.tab_reports, text="Reports")
        self.notebook.add(self.tab_settings, text="Settings")

        self._build_enroll_tab()
        self._build_train_tab()
        self._build_attend_tab()
        self._build_reports_tab()
        self._build_settings_tab()

        status_bar = tk.Label(self, textvariable=self.status_var, anchor="w")
        status_bar.pack(fill="x")

    def _build_enroll_tab(self):
        container = tk.Frame(self.tab_enroll)
        container.pack(padx=20, pady=20, fill="x")

        tk.Label(container, text="Student Name").grid(row=0, column=0, sticky="e", padx=8, pady=6)
        self.enroll_name = tk.Entry(container, width=40)
        self.enroll_name.grid(row=0, column=1, padx=8, pady=6)

        tk.Label(container, text="Samples").grid(row=1, column=0, sticky="e", padx=8, pady=6)
        self.enroll_samples = tk.Spinbox(container, from_=10, to=100, width=10)
        self.enroll_samples.delete(0, "end")
        self.enroll_samples.insert(0, "30")
        self.enroll_samples.grid(row=1, column=1, sticky="w", padx=8, pady=6)

        tk.Button(container, text="Start Enrollment", command=self._enroll_student).grid(
            row=2, column=1, sticky="w", padx=8, pady=10
        )

        self.enroll_list = tk.Listbox(self.tab_enroll, height=12, width=50)
        self.enroll_list.pack(padx=20, pady=10)
        self._refresh_student_list()

    def _build_train_tab(self):
        container = tk.Frame(self.tab_train)
        container.pack(padx=20, pady=20, fill="x")

        tk.Label(container, text="Train the model after enrollment.").pack(anchor="w")
        tk.Button(container, text="Train Model", command=self._train_model).pack(pady=12, anchor="w")
        tk.Button(container, text="Build Embeddings (Deep)", command=self._build_embeddings).pack(
            pady=4, anchor="w"
        )

        self.train_status = tk.Text(self.tab_train, height=10, width=80)
        self.train_status.pack(padx=20, pady=10)
        self.train_status.insert("end", "Ready to train.\n")
        self.train_status.config(state="disabled")

    def _build_attend_tab(self):
        container = tk.Frame(self.tab_attend)
        container.pack(padx=20, pady=20, fill="x")

        tk.Label(container, text="Camera Index").grid(row=0, column=0, sticky="e", padx=8, pady=6)
        self.camera_combo = ttk.Combobox(container, textvariable=self.camera_label_var, state="readonly")
        self.camera_combo.grid(row=0, column=1, sticky="w", padx=8, pady=6)
        tk.Button(container, text="Rescan", command=self._scan_cameras).grid(
            row=0, column=2, sticky="w", padx=8, pady=6
        )

        tk.Label(container, text="Period").grid(row=1, column=0, sticky="e", padx=8, pady=6)
        period_menu = ttk.Combobox(container, textvariable=self.period_var, values=PERIODS, state="readonly")
        period_menu.grid(row=1, column=1, sticky="w", padx=8, pady=6)

        tk.Checkbutton(
            container, text="Classroom mode (multiple faces)", variable=self.classroom_mode
        ).grid(row=2, column=1, sticky="w", padx=8, pady=6)

        tk.Checkbutton(
            container, text="Quality filter (blur/lighting)", variable=self.use_quality_filter
        ).grid(row=3, column=1, sticky="w", padx=8, pady=6)

        tk.Checkbutton(
            container, text="Use deep model (if configured)", variable=self.use_dnn
        ).grid(row=4, column=1, sticky="w", padx=8, pady=6)

        tk.Button(container, text="Start Attendance", command=self._start_attendance).grid(
            row=5, column=1, sticky="w", padx=8, pady=8
        )
        tk.Button(container, text="Stop Attendance", command=self._stop_attendance).grid(
            row=6, column=1, sticky="w", padx=8, pady=4
        )

        self._scan_cameras()

    def _build_reports_tab(self):
        container = tk.Frame(self.tab_reports)
        container.pack(padx=20, pady=20, fill="x")

        tk.Label(container, text="Export date (YYYY-MM-DD)").grid(row=0, column=0, sticky="e", padx=8, pady=6)
        self.export_date = tk.Entry(container, width=20)
        self.export_date.grid(row=0, column=1, sticky="w", padx=8, pady=6)
        self.export_date.insert(0, datetime.now().strftime("%Y-%m-%d"))

        tk.Button(container, text="Export to Excel", command=self._export_excel).grid(
            row=1, column=1, sticky="w", padx=8, pady=10
        )

        self.report_text = tk.Text(self.tab_reports, height=15, width=80)
        self.report_text.pack(padx=20, pady=10)
        self.report_text.insert("end", "Export report will appear here.\n")
        self.report_text.config(state="disabled")

    def _build_settings_tab(self):
        container = tk.Frame(self.tab_settings)
        container.pack(padx=20, pady=20, fill="x")

        tk.Label(container, text="Admin user:").grid(row=0, column=0, sticky="e", padx=8, pady=6)
        tk.Label(container, text=ADMIN_USERNAME, font=("Segoe UI", 10, "bold")).grid(
            row=0, column=1, sticky="w", padx=8, pady=6
        )

        tk.Label(container, text="Periods:").grid(row=1, column=0, sticky="ne", padx=8, pady=6)
        tk.Label(container, text=", ".join(PERIODS), wraplength=500, justify="left").grid(
            row=1, column=1, sticky="w", padx=8, pady=6
        )

    def _refresh_student_list(self):
        self.enroll_list.delete(0, "end")
        for _, name in get_students(DB_PATH):
            self.enroll_list.insert("end", name)

    def _enroll_student(self):
        name = self.enroll_name.get().strip()
        if not name:
            messagebox.showwarning("Missing name", "Please enter a student name.")
            return

        add_student(DB_PATH, name)
        self._refresh_student_list()
        samples = int(self.enroll_samples.get())

        self.status_var.set(f"Enrolling {name}...")
        self._run_enrollment_capture(name, samples)
        self.status_var.set(f"Enrollment finished for {name}.")

    def _run_enrollment_capture(self, name: str, samples: int):
        save_dir = os.path.join(DATA_DIR, name)
        ensure_dir(save_dir)

        cap = cv2.VideoCapture(self.camera_index.get())
        if not cap.isOpened():
            messagebox.showerror("Camera error", "Unable to open camera.")
            return

        face_cascade = cv2.CascadeClassifier(
            os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        )

        count = 0
        last_saved = 0.0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if count < samples and (time.time() - last_saved) > 0.2:
                    face_roi = gray[y : y + h, x : x + w]
                    if self.use_quality_filter.get():
                        blurry, _ = is_blurry(face_roi, BLUR_THRESHOLD)
                        bright = brightness_level(face_roi)
                        if blurry or bright < BRIGHTNESS_MIN or bright > BRIGHTNESS_MAX:
                            continue

                    face_path = os.path.join(save_dir, f"{count:03d}.jpg")
                    cv2.imwrite(face_path, face_roi)
                    count += 1
                    last_saved = time.time()

            cv2.putText(
                frame,
                f"Samples: {count}/{samples}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Enrollment", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if count >= samples:
                time.sleep(0.5)
                break

        cap.release()
        cv2.destroyAllWindows()

    def _train_model(self):
        from train import main as train_main

        self.train_status.config(state="normal")
        self.train_status.insert("end", "Training started...\n")
        self.train_status.config(state="disabled")
        self.status_var.set("Training model...")

        try:
            train_main()
            msg = "Training completed.\n"
        except SystemExit as exc:
            msg = f"Training failed: {exc}\n"
        except Exception as exc:
            msg = f"Training failed: {exc}\n"

        self.train_status.config(state="normal")
        self.train_status.insert("end", msg)
        self.train_status.config(state="disabled")
        self.status_var.set("Ready")

    def _build_embeddings(self):
        from build_embeddings import main as build_main

        self.train_status.config(state="normal")
        self.train_status.insert("end", "Building embeddings...\n")
        self.train_status.config(state="disabled")
        self.status_var.set("Building embeddings...")

        try:
            build_main()
            msg = "Embeddings built.\n"
        except SystemExit as exc:
            msg = f"Embeddings failed: {exc}\n"
        except Exception as exc:
            msg = f"Embeddings failed: {exc}\n"

        self.train_status.config(state="normal")
        self.train_status.insert("end", msg)
        self.train_status.config(state="disabled")
        self.status_var.set("Ready")

    def _start_attendance(self):
        if self.attend_thread and self.attend_thread.is_alive():
            messagebox.showinfo("Attendance", "Attendance is already running.")
            return

        if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
            messagebox.showwarning("Missing model", "Train the model first.")
            return

        self.attend_stop.clear()
        self.attend_thread = threading.Thread(target=self._run_attendance, daemon=True)
        self.attend_thread.start()
        self.status_var.set("Attendance running...")

    def _stop_attendance(self):
        self.attend_stop.set()
        self.status_var.set("Attendance stopped.")

    def _run_attendance(self):
        labels = load_labels(LABELS_PATH)
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(MODEL_PATH)

        cap = cv2.VideoCapture(self.camera_index.get())
        if not cap.isOpened():
            messagebox.showerror("Camera error", "Unable to open camera.")
            return

        face_cascade = cv2.CascadeClassifier(
            os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        )

        last_marked = {}
        recent_hits = {}
        cooldown_seconds = 60
        confidence_threshold = CONFIDENCE_THRESHOLD
        required_hits = REQUIRED_HITS
        camera_label = self.camera_label_var.get()
        last_alerted = {}
        unknown_alert_time = 0.0

        dnn_recognizer = None
        if self.use_dnn.get() and os.path.exists(EMBEDDING_MODEL_PATH) and os.path.exists(EMBEDDING_DB_PATH):
            dnn_recognizer = DNNRecognizer(EMBEDDING_MODEL_PATH, EMBEDDING_DB_PATH)

        while not self.attend_stop.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

            for (x, y, w, h) in faces:
                face_roi = gray[y : y + h, x : x + w]

                if self.use_quality_filter.get():
                    blurry, _ = is_blurry(face_roi, BLUR_THRESHOLD)
                    bright = brightness_level(face_roi)
                    if blurry or bright < BRIGHTNESS_MIN or bright > BRIGHTNESS_MAX:
                        continue

                if self.use_dnn.get() and dnn_recognizer:
                    name, score = dnn_recognizer.predict(frame[y : y + h, x : x + w], threshold=0.6)
                    label_id, confidence = 0, int((1.0 - score) * 100)
                else:
                    if self.use_dnn.get() and not dnn_recognizer:
                        self.status_var.set("Deep model missing, using LBPH")
                    label_id, confidence = recognizer.predict(face_roi)
                    name = labels.get(str(label_id), "Unknown")

                if confidence <= confidence_threshold and name != "Unknown":
                    recent_hits[name] = recent_hits.get(name, 0) + 1
                    if recent_hits[name] >= required_hits:
                        now = time.time()
                        last_time = last_marked.get(name, 0)
                        if now - last_time > cooldown_seconds:
                            record_student = get_student_by_name(DB_PATH, name)
                            if record_student:
                                student_id = record_student[0]
                                period = self.period_var.get()
                                date_str = datetime.now().strftime("%Y-%m-%d")
                                if not attendance_exists(DB_PATH, student_id, period, date_str):
                                    add_attendance(DB_PATH, student_id, period, camera_label)
                                    mark_attendance(name, ATTENDANCE_CSV)
                                    if now - last_alerted.get(name, 0) > 60:
                                        send_alert(
                                            "marked",
                                            f"Marked attendance: {name} | {period} | {camera_label}",
                                        )
                                        last_alerted[name] = now
                                last_marked[name] = now
                        recent_hits[name] = 0
                    color = (0, 255, 0)
                else:
                    name = "Unknown"
                    color = (0, 0, 255)
                    now = time.time()
                    if now - unknown_alert_time > 60:
                        send_alert("unknown", f"Unknown person detected | {camera_label}")
                        unknown_alert_time = now

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

                if not self.classroom_mode.get():
                    break

            cv2.imshow("Attendance", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    def _export_excel(self):
        date_str = self.export_date.get().strip()
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            messagebox.showwarning("Invalid date", "Use format YYYY-MM-DD.")
            return

        records = list_attendance_for_date(DB_PATH, date_str)
        if not records:
            messagebox.showinfo("No data", "No attendance found for the selected date.")
            return

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Attendance"
        ws.append(["Name", "Period", "Timestamp", "Camera"])

        for row in records:
            ws.append(list(row))

        out_path = os.path.join("reports", f"attendance_{date_str}.xlsx")
        ensure_dir(os.path.dirname(out_path))
        wb.save(out_path)

        self.report_text.config(state="normal")
        self.report_text.delete("1.0", "end")
        self.report_text.insert("end", f"Exported to {out_path}\n")
        self.report_text.config(state="disabled")

    def _scan_cameras(self):
        self.camera_indices = []
        max_index = 6
        for idx in range(max_index):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    self.camera_indices.append(idx)
            cap.release()

        if not self.camera_indices:
            self.camera_indices = [0]

        options = [f"Camera {i}" for i in self.camera_indices]
        self.camera_combo["values"] = options
        self.camera_label_var.set(options[0])
        self.camera_index.set(self.camera_indices[0])

        def on_select(_event):
            label = self.camera_label_var.get()
            if label.startswith("Camera "):
                idx = int(label.split(" ")[1])
                self.camera_index.set(idx)

        self.camera_combo.bind("<<ComboboxSelected>>", on_select)


if __name__ == "__main__":
    app = AttendanceApp()
    app.mainloop()

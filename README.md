# Face Attendance System (Local Webcam)

This project is a simple offline face attendance system using OpenCV LBPH.

## Quick start
1. Create a virtual environment (optional) and install deps:
   - `python -m venv .venv`
   - `.\.venv\Scripts\activate`
   - `pip install -r requirements.txt`
2. Enroll a person:
   - `python enroll.py --name "Alice"`
3. Train the model:
   - `python train.py`
4. Start attendance:
   - `python attend.py`

Attendance will be saved to `attendance.csv`.

## GUI app (recommended)
Run the full app with admin login, database, multi-camera selection, and Excel export:
- `python app.py`

Admin login:
- Set `ADMIN_USERNAME` and `ADMIN_PASSWORD` in `.env` (see `.env.example`).

## Alerts (SMS/Email)
Alerts are supported via environment variables. If variables are not set, alerts are silently skipped.

Email (SMTP):
- `SMTP_HOST`
- `SMTP_PORT` (default 587)
- `SMTP_USER`
- `SMTP_PASS`
- `SMTP_FROM` (optional)
- `SMTP_TO`

SMS (Twilio):
- `TWILIO_ACCOUNT_SID`
- `TWILIO_AUTH_TOKEN`
- `TWILIO_FROM`
- `TWILIO_TO`

## Deep-learning recognition (optional)
This app supports an optional deep embedding model. You must provide:
- `models/face_embedder.onnx`
- `models/embeddings.json` (build using `python build_embeddings.py`)

In the GUI, enable "Use deep model (if configured)" to use it. If files are missing, it falls back to LBPH.

## Tips
- Enroll each person with 20-40 samples in different angles and lighting.
- Keep the face centered and well lit.
- Press `q` to quit any camera window.

## Files
- `enroll.py`: Capture face samples for a person.
- `train.py`: Train the LBPH model.
- `attend.py`: Recognize faces and mark attendance.
- `app.py`: GUI app with admin login, periods, reports, and exports.
- `models/face_model.yml`: Trained model.
- `models/labels.json`: Label mapping.
- `data/<person_name>/`: Captured face samples.

## Configuration
- Copy `.env.example` to `.env` and set secrets locally.
- Do not commit `.env` to Git.

## Deployment notes
This repo contains:
- A local Tkinter desktop app (`app.py`) that cannot run on Vercel.
- A Django web app (use `python manage.py runserver` locally).

If you want Vercel deployment, we can add a Vercel-specific setup for the Django web app, but the desktop webcam features will not work in a serverless environment.

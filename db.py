import os
import sqlite3
from datetime import datetime

from utils import ensure_dir


def get_conn(db_path: str):
    ensure_dir(os.path.dirname(db_path))
    return sqlite3.connect(db_path)


def init_db(db_path: str) -> None:
    with get_conn(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                period TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                camera TEXT,
                FOREIGN KEY(student_id) REFERENCES students(id)
            )
            """
        )
        conn.commit()


def add_student(db_path: str, name: str) -> int:
    with get_conn(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT OR IGNORE INTO students (name, created_at) VALUES (?, ?)",
            (name, datetime.now().isoformat(timespec="seconds")),
        )
        conn.commit()
        cur.execute("SELECT id FROM students WHERE name = ?", (name,))
        row = cur.fetchone()
        return int(row[0]) if row else -1


def get_students(db_path: str):
    with get_conn(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, name FROM students ORDER BY name")
        return cur.fetchall()


def get_student_by_name(db_path: str, name: str):
    with get_conn(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, name FROM students WHERE name = ?", (name,))
        return cur.fetchone()


def add_attendance(db_path: str, student_id: int, period: str, camera: str) -> None:
    with get_conn(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO attendance (student_id, period, timestamp, camera) VALUES (?, ?, ?, ?)",
            (
                student_id,
                period,
                datetime.now().isoformat(timespec="seconds"),
                camera,
            ),
        )
        conn.commit()


def attendance_exists(db_path: str, student_id: int, period: str, date_str: str) -> bool:
    with get_conn(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT COUNT(1) FROM attendance
            WHERE student_id = ? AND period = ? AND timestamp LIKE ?
            """,
            (student_id, period, f"{date_str}%"),
        )
        row = cur.fetchone()
        return bool(row and row[0] > 0)


def list_attendance_for_date(db_path: str, date_str: str):
    with get_conn(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT s.name, a.period, a.timestamp, a.camera
            FROM attendance a
            JOIN students s ON s.id = a.student_id
            WHERE a.timestamp LIKE ?
            ORDER BY a.timestamp
            """,
            (f"{date_str}%",),
        )
        return cur.fetchall()


def clear_attendance_for_date(db_path: str, date_str: str):
    with get_conn(db_path) as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM attendance WHERE timestamp LIKE ?", (f"{date_str}%",))
        conn.commit()

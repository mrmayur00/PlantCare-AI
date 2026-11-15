"""
database.py - SQLite storage for scan history.

Table: scan_history
    id              INTEGER PRIMARY KEY
    plant_name      TEXT
    disease_name    TEXT
    disease_status  TEXT     ("Healthy", "Diseased" or "Uncertain")
    confidence      REAL     (0-100)
    image_path      TEXT     (file name inside static/uploads)
    scan_date       TEXT     (ISO date-time, local time)
    result_data     TEXT     (full JSON result shown on the result page)
"""
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime

import config


class DatabaseError(Exception):
    """Raised when a database operation fails."""


@contextmanager
def _connect():
    """Open a connection, commit on success, roll back on error, always close."""
    os.makedirs(os.path.dirname(config.DATABASE_PATH), exist_ok=True)
    conn = sqlite3.connect(config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row  # rows behave like dicts
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Create the scan_history table if it does not exist yet."""
    try:
        with _connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS scan_history (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    plant_name     TEXT NOT NULL,
                    disease_name   TEXT NOT NULL,
                    disease_status TEXT NOT NULL,
                    confidence     REAL NOT NULL,
                    image_path     TEXT NOT NULL,
                    scan_date      TEXT NOT NULL,
                    result_data    TEXT
                )
                """
            )
    except sqlite3.Error as exc:
        raise DatabaseError(f"Could not initialise the database: {exc}")


def save_scan(result, image_filename):
    """Store a completed scan and return its new id."""
    try:
        with _connect() as conn:
            cur = conn.execute(
                """INSERT INTO scan_history
                   (plant_name, disease_name, disease_status, confidence,
                    image_path, scan_date, result_data)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    result["plant"],
                    result["disease"],
                    result["status"],
                    result["confidence"],
                    image_filename,
                    datetime.now().isoformat(timespec="seconds"),
                    json.dumps(result),
                ),
            )
            return cur.lastrowid
    except sqlite3.Error as exc:
        raise DatabaseError(f"Could not save the scan: {exc}")


def _row_summary(row):
    try:
        local_name = json.loads(row["result_data"] or "{}").get("local_name")
    except ValueError:
        local_name = None
    return {
        "id": row["id"],
        "plant": row["plant_name"],
        "local_name": local_name,
        "disease": row["disease_name"],
        "status": row["disease_status"],
        "confidence": row["confidence"],
        "image_url": f"/static/uploads/{row['image_path']}",
        "scan_date": row["scan_date"],
    }


def get_all_scans():
    """Return a list of all scans, newest first (summary fields only)."""
    try:
        with _connect() as conn:
            rows = conn.execute("SELECT * FROM scan_history ORDER BY id DESC").fetchall()
        return [_row_summary(r) for r in rows]
    except sqlite3.Error as exc:
        raise DatabaseError(f"Could not read scan history: {exc}")


def get_scan(scan_id):
    """Return the complete stored result for one scan, or None if not found."""
    try:
        with _connect() as conn:
            row = conn.execute("SELECT * FROM scan_history WHERE id = ?", (scan_id,)).fetchone()
    except sqlite3.Error as exc:
        raise DatabaseError(f"Could not read the scan: {exc}")
    if row is None:
        return None
    data = json.loads(row["result_data"]) if row["result_data"] else {}
    data.update(_row_summary(row))  # id, image_url and scan_date always come from the row
    return data


def delete_scan(scan_id):
    """
    Delete a scan record. Returns the image file name that belonged to it
    (so the caller can delete the file), or None if the id was not found.
    """
    try:
        with _connect() as conn:
            row = conn.execute("SELECT image_path FROM scan_history WHERE id = ?", (scan_id,)).fetchone()
            if row is None:
                return None
            conn.execute("DELETE FROM scan_history WHERE id = ?", (scan_id,))
            return row["image_path"]
    except sqlite3.Error as exc:
        raise DatabaseError(f"Could not delete the scan: {exc}")

"""
database/database.py
====================
All SQLite code lives here. No other file writes SQL.

Tables
------
users                 registered people (name, registration ID, dataset folder)
recognition_history   every recognition / verification result
training_history      every completed training run (real metrics only)
app_settings          small key/value store (e.g. confidence threshold)

Design notes
------------
* One short-lived connection per operation. This is safe when training runs
  in a background thread while the GUI thread reads the database.
* Foreign keys are switched on for every connection (SQLite has them off
  by default).
* Deleting a user keeps their old recognition history rows (the name stays
  as text, the user link becomes NULL).
* Only the minimum personal data is stored: name and registration ID.
"""

from __future__ import annotations

import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

from config import settings

SCHEMA_VERSION = 1


class DatabaseError(Exception):
    """Base class for database problems we want to show to the user."""


class ValidationError(DatabaseError):
    """The user typed something invalid (empty name, bad ID format...)."""


class DuplicateRegistrationIDError(DatabaseError):
    """This registration ID is already used by another person."""


def _now() -> str:
    """Current local time as text, e.g. '2026-01-31T14:05:09'."""
    return datetime.now().isoformat(timespec="seconds")


class Database:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else settings.DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Connection handling
    # ------------------------------------------------------------------
    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()          # save if everything went well
        except Exception:
            conn.rollback()        # undo half-finished changes
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def initialize(self) -> None:
        """Create all tables if they do not exist. Safe to call every start."""
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    registration_id TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    full_name       TEXT NOT NULL,
                    folder_name     TEXT NOT NULL,
                    registered_at   TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS recognition_history (
                    id                INTEGER PRIMARY KEY AUTOINCREMENT,
                    recognized_at     TEXT NOT NULL,
                    source            TEXT NOT NULL
                        CHECK (source IN ('live', 'image', 'verification')),
                    predicted_user_id INTEGER
                        REFERENCES users(id) ON DELETE SET NULL,
                    predicted_name    TEXT NOT NULL,
                    confidence        REAL NOT NULL,
                    is_unknown        INTEGER NOT NULL DEFAULT 0,
                    model_version     TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_recognition_time
                    ON recognition_history(recognized_at);

                CREATE TABLE IF NOT EXISTS training_history (
                    id                INTEGER PRIMARY KEY AUTOINCREMENT,
                    trained_at        TEXT NOT NULL,
                    model_version     TEXT NOT NULL,
                    model_path        TEXT NOT NULL,
                    dataset_signature TEXT,
                    num_classes       INTEGER NOT NULL,
                    train_samples     INTEGER,
                    val_samples       INTEGER,
                    test_samples      INTEGER,
                    epochs_run        INTEGER,
                    batch_size        INTEGER,
                    config_json       TEXT,
                    train_accuracy    REAL,
                    val_accuracy      REAL,
                    test_accuracy     REAL,
                    train_loss        REAL,
                    val_loss          REAL,
                    test_loss         REAL
                );

                CREATE TABLE IF NOT EXISTS app_settings (
                    key   TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def schema_version(self) -> int:
        with self._connect() as conn:
            return conn.execute("PRAGMA user_version").fetchone()[0]

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _clean_name(full_name: str) -> str:
        name = " ".join((full_name or "").split())    # trim + single spaces
        if not name:
            raise ValidationError("Full name cannot be empty.")
        if len(name) > settings.NAME_MAX_LENGTH:
            raise ValidationError(
                f"Full name is too long (max {settings.NAME_MAX_LENGTH} characters).")
        return name

    @staticmethod
    def _clean_registration_id(registration_id: str) -> str:
        reg_id = (registration_id or "").strip()
        if not reg_id:
            raise ValidationError("Registration ID cannot be empty.")
        if len(reg_id) > settings.REGISTRATION_ID_MAX_LENGTH:
            raise ValidationError(
                f"Registration ID is too long (max "
                f"{settings.REGISTRATION_ID_MAX_LENGTH} characters).")
        if not re.match(settings.REGISTRATION_ID_PATTERN, reg_id):
            raise ValidationError(
                "Registration ID may only contain letters, digits, '-', '_' "
                "and '.', and must start with a letter or digit.")
        return reg_id

    # ------------------------------------------------------------------
    # Users
    # ------------------------------------------------------------------
    def add_user(self, registration_id: str, full_name: str) -> dict:
        """
        Register a person. The dataset folder name (person_001 ...) is
        created from the new database id, so it never depends on what the
        user typed. Returns the saved user as a dict.
        """
        reg_id = self._clean_registration_id(registration_id)
        name = self._clean_name(full_name)
        try:
            with self._connect() as conn:
                cur = conn.execute(
                    "INSERT INTO users (registration_id, full_name, folder_name, "
                    "registered_at) VALUES (?, ?, ?, ?)",
                    (reg_id, name, "", _now()),
                )
                user_id = cur.lastrowid
                conn.execute(
                    "UPDATE users SET folder_name = ? WHERE id = ?",
                    (settings.person_folder_name(user_id), user_id),
                )
                row = conn.execute(
                    "SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
                return dict(row)
        except sqlite3.IntegrityError as exc:
            raise DuplicateRegistrationIDError(
                f"Registration ID '{reg_id}' is already registered.") from exc

    def registration_id_exists(self, registration_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM users WHERE registration_id = ?",
                ((registration_id or "").strip(),)).fetchone()
            return row is not None

    def get_user(self, user_id: int) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            return dict(row) if row else None

    def get_user_by_registration_id(self, registration_id: str) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE registration_id = ?",
                ((registration_id or "").strip(),)).fetchone()
            return dict(row) if row else None

    def list_users(self, search: str = "") -> list:
        """All users (oldest first). `search` matches name or registration ID."""
        query = "SELECT * FROM users"
        params: tuple = ()
        search = (search or "").strip()
        if search:
            # escape LIKE wildcards so searching for '%' or '_' is literal
            safe = search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            query += (" WHERE full_name LIKE ? ESCAPE '\\'"
                      " OR registration_id LIKE ? ESCAPE '\\'")
            params = (f"%{safe}%", f"%{safe}%")
        query += " ORDER BY id"
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(query, params).fetchall()]

    def count_users(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]

    def delete_user(self, user_id: int) -> bool:
        """
        Remove the user row. Returns True if a row was deleted.
        Deleting the image folder is dataset_manager's job (Stage 4).
        After any delete the model is outdated and must be retrained.
        """
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
            return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Recognition history
    # ------------------------------------------------------------------
    def log_recognition(self, source: str, predicted_name: str, confidence: float,
                        is_unknown: bool, predicted_user_id: Optional[int] = None,
                        model_version: Optional[str] = None) -> int:
        """Save one recognition result. `source`: 'live', 'image' or 'verification'."""
        if source not in ("live", "image", "verification"):
            raise ValidationError(f"Unknown recognition source: {source!r}")
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO recognition_history (recognized_at, source, "
                "predicted_user_id, predicted_name, confidence, is_unknown, "
                "model_version) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (_now(), source, predicted_user_id, predicted_name,
                 float(confidence), 1 if is_unknown else 0, model_version),
            )
            return cur.lastrowid

    def get_recent_recognitions(self, limit: int = 10) -> list:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM recognition_history "
                "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
            return [dict(r) for r in rows]

    def count_recognitions(self) -> int:
        with self._connect() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM recognition_history").fetchone()[0]

    # ------------------------------------------------------------------
    # Training history
    # ------------------------------------------------------------------
    def add_training_record(self, model_version: str, model_path: str,
                            num_classes: int, **optional) -> int:
        """
        Save one finished training run. Optional keyword values:
        dataset_signature, train_samples, val_samples, test_samples,
        epochs_run, batch_size, config_json, train_accuracy, val_accuracy,
        test_accuracy, train_loss, val_loss, test_loss.
        Pass only REAL values measured during training.
        """
        allowed = ("dataset_signature", "train_samples", "val_samples",
                   "test_samples", "epochs_run", "batch_size", "config_json",
                   "train_accuracy", "val_accuracy", "test_accuracy",
                   "train_loss", "val_loss", "test_loss")
        unknown = set(optional) - set(allowed)
        if unknown:
            raise ValidationError(f"Unknown training fields: {sorted(unknown)}")
        columns = ["trained_at", "model_version", "model_path", "num_classes"]
        values = [_now(), model_version, str(model_path), num_classes]
        for key in allowed:
            if key in optional:
                columns.append(key)
                values.append(optional[key])
        placeholders = ", ".join("?" for _ in columns)
        with self._connect() as conn:
            cur = conn.execute(
                f"INSERT INTO training_history ({', '.join(columns)}) "
                f"VALUES ({placeholders})", values)
            return cur.lastrowid

    def get_latest_training(self) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM training_history ORDER BY id DESC LIMIT 1").fetchone()
            return dict(row) if row else None

    def list_training_history(self) -> list:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM training_history ORDER BY id DESC").fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # App settings (key/value)
    # ------------------------------------------------------------------
    def get_setting(self, key: str, default: Optional[str] = None) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM app_settings WHERE key = ?", (key,)).fetchone()
            return row["value"] if row else default

    def set_setting(self, key: str, value) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO app_settings (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, str(value)))

    def get_confidence_threshold(self) -> float:
        """Saved threshold, or the default if none saved / value is corrupt."""
        raw = self.get_setting("confidence_threshold")
        lo, hi = settings.CONFIDENCE_THRESHOLD_RANGE
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return settings.DEFAULT_CONFIDENCE_THRESHOLD
        return value if lo <= value <= hi else settings.DEFAULT_CONFIDENCE_THRESHOLD

    def set_confidence_threshold(self, value: float) -> None:
        lo, hi = settings.CONFIDENCE_THRESHOLD_RANGE
        if not (lo <= value <= hi):
            raise ValidationError(f"Threshold must be between {lo} and {hi}.")
        self.set_setting("confidence_threshold", value)

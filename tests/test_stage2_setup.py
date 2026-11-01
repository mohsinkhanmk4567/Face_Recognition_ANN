"""
tests/test_stage2_setup.py
==========================
Tests for Stage 2 (settings + SQLite). No webcam, GUI or TensorFlow needed.

Run from the project root:

    python -m unittest discover -s tests -v
"""

import sqlite3
import sys
import tempfile
import threading
import unittest
from contextlib import closing
from pathlib import Path

# make "config" and "database" importable when run from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings
from config.settings import ANNConfig
from database.database import (Database, DuplicateRegistrationIDError,
                               ValidationError)


class TestSettings(unittest.TestCase):
    def test_default_settings_are_valid(self):
        settings.validate_settings()

    def test_split_ratios_sum_to_one(self):
        self.assertAlmostEqual(
            settings.TRAIN_RATIO + settings.VAL_RATIO + settings.TEST_RATIO, 1.0)

    def test_input_dim_matches_image_size(self):
        self.assertEqual(settings.INPUT_DIM, settings.IMAGE_SIZE ** 2)

    def test_person_folder_name(self):
        self.assertEqual(settings.person_folder_name(7), "person_007")
        self.assertEqual(settings.person_folder_name(123), "person_123")

    def test_ann_config_roundtrip_through_json_types(self):
        cfg = ANNConfig(hidden_layers=(256, 64), dropout_rates=(0.5, 0.2),
                        l2_regularization=0.001)
        restored = ANNConfig.from_dict(cfg.to_dict())   # lists -> tuples again
        self.assertEqual(restored, cfg)
        self.assertIsInstance(restored.hidden_layers, tuple)

    def test_ann_config_rejects_bad_values(self):
        bad = [
            dict(hidden_layers=()),
            dict(hidden_layers=(64, 32), dropout_rates=(0.3,)),       # length mismatch
            dict(hidden_layers=(64,), dropout_rates=(1.0,)),          # dropout too high
            dict(hidden_layers=(0,), dropout_rates=(0.1,)),           # zero neurons
            dict(activation="banana"),
            dict(optimizer="magic"),
            dict(learning_rate=0),
            dict(epochs=0),
            dict(batch_size=0),
        ]
        for kwargs in bad:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    ANNConfig(**kwargs).validate()


class DatabaseTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db = Database(Path(self._tmp.name) / "test.db")
        self.db.initialize()

    def tearDown(self):
        self._tmp.cleanup()


class TestDatabaseSetup(DatabaseTestCase):
    def test_tables_created(self):
        with closing(sqlite3.connect(self.db.db_path)) as conn:
            names = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
        for table in ("users", "recognition_history", "training_history",
                      "app_settings"):
            self.assertIn(table, names)

    def test_initialize_twice_keeps_data(self):
        self.db.add_user("A1", "Ali Khan")
        self.db.initialize()
        self.assertEqual(self.db.count_users(), 1)

    def test_schema_version(self):
        self.assertEqual(self.db.schema_version(), 1)


class TestUsers(DatabaseTestCase):
    def test_add_user_and_folder_name(self):
        user = self.db.add_user("CS-2201", "  Ayesha   Noor ")
        self.assertEqual(user["full_name"], "Ayesha Noor")      # spaces cleaned
        self.assertEqual(user["folder_name"],
                         settings.person_folder_name(user["id"]))
        self.assertTrue(user["folder_name"].startswith("person_"))

    def test_folder_names_are_unique_and_sequential(self):
        a = self.db.add_user("A1", "A")
        b = self.db.add_user("B1", "B")
        self.assertNotEqual(a["folder_name"], b["folder_name"])
        self.assertEqual(b["id"], a["id"] + 1)

    def test_duplicate_registration_id_rejected_case_insensitive(self):
        self.db.add_user("cs-1", "First")
        with self.assertRaises(DuplicateRegistrationIDError):
            self.db.add_user("CS-1", "Second")
        self.assertEqual(self.db.count_users(), 1)
        self.assertTrue(self.db.registration_id_exists("CS-1"))

    def test_failed_insert_does_not_leave_ghost_row(self):
        self.db.add_user("X1", "X")
        try:
            self.db.add_user("x1", "Y")
        except DuplicateRegistrationIDError:
            pass
        self.assertEqual([u["registration_id"] for u in self.db.list_users()],
                         ["X1"])

    def test_validation_errors(self):
        for reg, name in [("", "Name"), ("   ", "Name"), ("ID1", ""),
                          ("ID1", "   "), ("bad id!", "Name"), ("-start", "Name"),
                          ("A" * 31, "Name"), ("ID1", "N" * 61)]:
            with self.subTest(reg=reg, name=name):
                with self.assertRaises(ValidationError):
                    self.db.add_user(reg, name)
        self.assertEqual(self.db.count_users(), 0)

    def test_search_by_name_and_id_and_wildcards_are_literal(self):
        self.db.add_user("CS-1", "Ali Raza")
        self.db.add_user("EE-2", "Sara Ahmed")
        self.assertEqual(len(self.db.list_users("raza")), 1)
        self.assertEqual(len(self.db.list_users("ee-")), 1)
        self.assertEqual(len(self.db.list_users("")), 2)
        self.assertEqual(self.db.list_users("%"), [])      # '%' is not a wildcard
        self.assertEqual(self.db.list_users("zzz"), [])

    def test_get_user_helpers(self):
        u = self.db.add_user("Q9", "Qasim")
        self.assertEqual(self.db.get_user(u["id"])["full_name"], "Qasim")
        self.assertEqual(self.db.get_user_by_registration_id("q9")["id"], u["id"])
        self.assertIsNone(self.db.get_user(9999))

    def test_delete_user(self):
        u = self.db.add_user("D1", "Delete Me")
        self.assertTrue(self.db.delete_user(u["id"]))
        self.assertFalse(self.db.delete_user(u["id"]))      # already gone
        self.assertEqual(self.db.count_users(), 0)

    def test_deleted_id_is_not_reused_for_folder_name(self):
        a = self.db.add_user("A1", "A")
        self.db.delete_user(a["id"])
        b = self.db.add_user("B1", "B")
        self.assertNotEqual(a["folder_name"], b["folder_name"])


class TestHistory(DatabaseTestCase):
    def test_log_and_read_recognition(self):
        u = self.db.add_user("R1", "Rida")
        self.db.log_recognition("live", "Rida", 0.93, False, u["id"], "model_x")
        self.db.log_recognition("image", "Unknown", 0.41, True)
        recent = self.db.get_recent_recognitions(10)
        self.assertEqual(len(recent), 2)
        self.assertEqual(recent[0]["predicted_name"], "Unknown")   # newest first
        self.assertEqual(recent[0]["is_unknown"], 1)
        self.assertEqual(self.db.count_recognitions(), 2)

    def test_invalid_source_rejected(self):
        with self.assertRaises(ValidationError):
            self.db.log_recognition("telepathy", "X", 0.5, False)

    def test_history_survives_user_deletion(self):
        u = self.db.add_user("H1", "Hina")
        self.db.log_recognition("live", "Hina", 0.9, False, u["id"])
        self.db.delete_user(u["id"])
        row = self.db.get_recent_recognitions(1)[0]
        self.assertEqual(row["predicted_name"], "Hina")     # name kept as text
        self.assertIsNone(row["predicted_user_id"])         # link cleared

    def test_foreign_keys_enforced(self):
        with self.assertRaises(sqlite3.IntegrityError):
            self.db.log_recognition("live", "Ghost", 0.9, False,
                                    predicted_user_id=424242)

    def test_training_record_roundtrip(self):
        self.assertIsNone(self.db.get_latest_training())
        self.db.add_training_record("model_1", "models/model_1", 3,
                                    val_accuracy=0.8, epochs_run=12)
        self.db.add_training_record("model_2", "models/model_2", 4)
        latest = self.db.get_latest_training()
        self.assertEqual(latest["model_version"], "model_2")
        self.assertIsNone(latest["val_accuracy"])           # not invented
        self.assertEqual(len(self.db.list_training_history()), 2)

    def test_training_record_rejects_unknown_field(self):
        with self.assertRaises(ValidationError):
            self.db.add_training_record("m", "p", 2, fake_metric=1.0)


class TestAppSettings(DatabaseTestCase):
    def test_default_threshold_when_nothing_saved(self):
        self.assertEqual(self.db.get_confidence_threshold(),
                         settings.DEFAULT_CONFIDENCE_THRESHOLD)

    def test_save_and_reload_threshold(self):
        self.db.set_confidence_threshold(0.85)
        self.assertAlmostEqual(self.db.get_confidence_threshold(), 0.85)

    def test_threshold_out_of_range_rejected(self):
        with self.assertRaises(ValidationError):
            self.db.set_confidence_threshold(0.05)
        with self.assertRaises(ValidationError):
            self.db.set_confidence_threshold(1.5)

    def test_corrupt_threshold_falls_back_to_default(self):
        self.db.set_setting("confidence_threshold", "not-a-number")
        self.assertEqual(self.db.get_confidence_threshold(),
                         settings.DEFAULT_CONFIDENCE_THRESHOLD)

    def test_generic_setting_overwrite(self):
        self.db.set_setting("camera_index", 0)
        self.db.set_setting("camera_index", 1)
        self.assertEqual(self.db.get_setting("camera_index"), "1")
        self.assertEqual(self.db.get_setting("missing", "fallback"), "fallback")


class TestThreadSafety(DatabaseTestCase):
    def test_writes_from_background_thread(self):
        errors = []

        def worker():
            try:
                for i in range(20):
                    self.db.add_user(f"T{i}", f"Thread User {i}")
            except Exception as exc:         # pragma: no cover
                errors.append(exc)

        t = threading.Thread(target=worker)
        t.start()
        for _ in range(20):                  # GUI thread reads meanwhile
            self.db.list_users()
        t.join()
        self.assertEqual(errors, [])
        self.assertEqual(self.db.count_users(), 20)


if __name__ == "__main__":
    unittest.main(verbosity=2)

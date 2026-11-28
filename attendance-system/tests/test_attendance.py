import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from attendance import backup_database, combine_samples, connect, correct_attendance, enroll, identify, mark, students


class AttendanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = connect(Path(self.temp.name) / "test.db")

    def tearDown(self):
        self.db.close()
        self.temp.cleanup()

    def test_match_rejects_unknown_and_ambiguous(self):
        enroll(self.db, "S01", "A", "10", [1, 0, 0])
        enroll(self.db, "S02", "B", "10", [0, 1, 0])
        roster = students(self.db)
        self.assertEqual(identify([0.99, 0.01, 0], roster)[0], "S01")
        self.assertIsNone(identify([0, 0, 1], roster))
        self.assertIsNone(identify([1, 1, 0], roster))

    def test_once_per_day_and_delete_removes_records(self):
        enroll(self.db, "S01", "A", "10", [1, 0])
        time = datetime(2026, 11, 28, 9, 0, tzinfo=timezone.utc)
        self.assertTrue(mark(self.db, "S01", 0.88, time))
        self.assertFalse(mark(self.db, "S01", 0.91, time))
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM attendance").fetchone()[0], 1)
        with self.db:
            self.db.execute("DELETE FROM students WHERE student_id = ?", ("S01",))
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM attendance").fetchone()[0], 0)

    def test_duplicate_id_is_rejected(self):
        enroll(self.db, "S01", "A", "10", [1, 0])
        with self.assertRaises(sqlite3.IntegrityError):
            enroll(self.db, "S01", "Other", "10", [0, 1])

    def test_correction_requires_reason_and_is_audited(self):
        enroll(self.db, "S01", "A", "10", [1, 0])
        with self.assertRaises(ValueError):
            correct_attendance(self.db, "S01", "2026-11-28", True, " ")
        correct_attendance(self.db, "S01", "2026-11-28", True, "Paper register")
        self.assertEqual(self.db.execute("SELECT score FROM attendance").fetchone()[0], -1)
        correct_attendance(self.db, "S01", "2026-11-28", False, "Entered in error")
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM attendance").fetchone()[0], 0)
        self.assertEqual(self.db.execute("SELECT COUNT(*) FROM audit").fetchone()[0], 2)

    def test_three_samples_and_duplicate_identity(self):
        feature = combine_samples([[1, 0, 0], [0.95, 0.1, 0], [0.9, -0.1, 0]])
        enroll(self.db, "S01", "A", "10-A", feature)
        with self.assertRaisesRegex(ValueError, "already enrolled"):
            enroll(self.db, "S02", "A again", "10-A", [1, 0, 0])
        with self.assertRaisesRegex(ValueError, "differ too much"):
            combine_samples([[1, 0], [0, 1], [1, 0]])

    def test_backup_is_consistent_and_does_not_overwrite(self):
        enroll(self.db, "S01", "A", "10-A", [1, 0])
        backup = Path(self.temp.name) / "backups" / "copy.db"
        backup_database(self.db, backup)
        with sqlite3.connect(backup) as copy:
            self.assertEqual(copy.execute("SELECT name FROM students").fetchone()[0], "A")
            self.assertEqual(copy.execute("PRAGMA integrity_check").fetchone()[0], "ok")
        with self.assertRaises(FileExistsError):
            backup_database(self.db, backup)


if __name__ == "__main__":
    unittest.main()

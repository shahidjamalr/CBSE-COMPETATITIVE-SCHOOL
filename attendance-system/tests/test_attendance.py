import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from attendance import connect, correct_attendance, enroll, identify, mark, students


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


if __name__ == "__main__":
    unittest.main()

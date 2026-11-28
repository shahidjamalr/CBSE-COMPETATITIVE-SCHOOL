"""Local, consent-based face attendance prototype."""

import argparse
import csv
from datetime import datetime
import os
import sqlite3
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_DB = ROOT / "data" / "attendance.db"
DETECTOR = ROOT / "models" / "face_detection_yunet_2023mar.onnx"
RECOGNIZER = ROOT / "models" / "face_recognition_sface_2021dec.onnx"
THRESHOLD = 0.50  # Conservative starting point; calibrate with consented local data.


def connect(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.execute("PRAGMA foreign_keys = ON")
    db.execute("PRAGMA busy_timeout = 5000")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS students (
            student_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            class_name TEXT NOT NULL,
            embedding BLOB NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS attendance (
            student_id TEXT NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
            day TEXT NOT NULL,
            marked_at TEXT NOT NULL,
            score REAL NOT NULL,
            PRIMARY KEY (student_id, day)
        );
        CREATE TABLE IF NOT EXISTS audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
            day TEXT NOT NULL,
            action TEXT NOT NULL,
            at TEXT NOT NULL,
            note TEXT NOT NULL
        );
    """)
    return db


def normalize(vector):
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    size = np.linalg.norm(value)
    if not np.isfinite(value).all() or size < 1e-8:
        raise ValueError("Invalid face embedding")
    return value / size


def enroll(db, student_id, name, class_name, feature):
    if not all(x.strip() for x in (student_id, name, class_name)):
        raise ValueError("ID, name and class are required")
    embedding = normalize(feature)
    for existing_id, _, _, existing in students(db):
        if existing.shape == embedding.shape and float(np.dot(embedding, existing)) >= 0.70:
            raise ValueError(f"Face looks like already enrolled student {existing_id}; review before continuing")
    with db:
        db.execute(
            "INSERT INTO students VALUES (?, ?, ?, ?, ?)",
            (student_id.strip(), name.strip(), class_name.strip(),
             embedding.tobytes(), datetime.now().astimezone().isoformat(timespec="seconds")),
        )


def combine_samples(features, minimum_similarity=0.55):
    """Reject inconsistent captures and average three consented face samples."""
    if len(features) != 3:
        raise ValueError("Capture three face samples")
    samples = [normalize(feature) for feature in features]
    if any(sample.shape != samples[0].shape for sample in samples):
        raise ValueError("Face samples have incompatible dimensions")
    if any(float(np.dot(samples[i], samples[j])) < minimum_similarity
           for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Face samples differ too much; recapture the same student in good light")
    return normalize(np.mean(samples, axis=0))


def backup_database(db, destination):
    """Make a consistent, non-overwriting copy while the database is open."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    try:
        with sqlite3.connect(destination) as target:
            db.backup(target)
            if target.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise RuntimeError("Backup integrity check failed")
        return destination
    except Exception:
        destination.unlink(missing_ok=True)
        raise


def students(db):
    return [(sid, name, class_name, np.frombuffer(blob, dtype=np.float32))
            for sid, name, class_name, blob in db.execute(
                "SELECT student_id, name, class_name, embedding FROM students ORDER BY student_id")]


def identify(feature, roster, threshold=THRESHOLD, margin=0.05):
    """Reject ambiguous nearest neighbours as well as low similarity."""
    query = normalize(feature)
    scores = []
    for sid, name, class_name, vector in roster:
        if vector.shape != query.shape:
            continue
        scores.append((float(np.dot(query, vector)), sid, name, class_name))
    scores.sort(reverse=True)
    if not scores or scores[0][0] < threshold:
        return None
    if len(scores) > 1 and scores[0][0] - scores[1][0] < margin:
        return None
    score, sid, name, class_name = scores[0]
    return sid, name, class_name, score


def mark(db, student_id, score, when=None):
    when = when or datetime.now().astimezone()
    with db:
        cursor = db.execute(
            "INSERT OR IGNORE INTO attendance VALUES (?, ?, ?, ?)",
            (student_id, when.date().isoformat(), when.isoformat(timespec="seconds"), score),
        )
    return cursor.rowcount == 1


def correct_attendance(db, student_id, day, present, note):
    """Record an operator correction and keep an audit trail."""
    if not note.strip():
        raise ValueError("A reason is required for an attendance correction")
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    with db:
        if present:
            db.execute("""INSERT INTO attendance VALUES (?, ?, ?, ?)
                ON CONFLICT(student_id, day) DO UPDATE SET marked_at=excluded.marked_at,
                score=excluded.score""", (student_id, day, now, -1.0))
        else:
            db.execute("DELETE FROM attendance WHERE student_id=? AND day=?", (student_id, day))
        db.execute("INSERT INTO audit(student_id, day, action, at, note) VALUES (?, ?, ?, ?, ?)",
                   (student_id, day, "present" if present else "absent", now, note.strip()))


class Vision:
    def __init__(self):
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("Install dependencies: pip install -r requirements.txt") from exc
        for model in (DETECTOR, RECOGNIZER):
            if not model.is_file():
                raise RuntimeError(f"Missing model: {model.name}. See README.md.")
        self.cv = cv2
        self.detector = cv2.FaceDetectorYN.create(str(DETECTOR), "", (320, 320))
        self.recognizer = cv2.FaceRecognizerSF.create(str(RECOGNIZER), "")

    def faces(self, frame):
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)
        return [] if faces is None else list(faces)

    def feature(self, frame, face):
        cropped = self.recognizer.alignCrop(frame, face)
        return normalize(self.recognizer.feature(cropped))


def open_camera(cv, index):
    camera = cv.VideoCapture(index)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"Could not open camera {index}")
    return camera


def enroll_camera(vision, index):
    cv = vision.cv
    camera = open_camera(cv, index)
    print("Look at the camera. Press SPACE three times with small angle changes; ESC to cancel.")
    samples = []
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Camera stopped returning frames")
            faces = vision.faces(frame)
            preview = frame.copy()
            for face in faces:
                x, y, w, h = map(int, face[:4])
                cv.rectangle(preview, (x, y), (x+w, y+h), (0, 200, 255), 2)
            cv.imshow("Enroll - SPACE to capture", preview)
            key = cv.waitKey(1) & 0xFF
            if key == 27:
                raise RuntimeError("Enrollment cancelled")
            if key == 32:
                if len(faces) != 1:
                    print("Exactly one face must be visible; try again.")
                    continue
                samples.append(vision.feature(frame, faces[0]))
                print(f"Captured {len(samples)}/3")
                if len(samples) == 3:
                    return combine_samples(samples)
    finally:
        camera.release()
        cv.destroyAllWindows()


def scan_camera(vision, db, index, threshold):
    cv = vision.cv
    roster = students(db)
    if not roster:
        raise RuntimeError("Enroll at least one student first")
    camera = open_camera(cv, index)
    consecutive = {}  # Requires three consecutive confident frames per identity.
    print("Scanning. Press Q or ESC to stop.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Camera stopped returning frames")
            visible = set()
            preview = frame.copy()
            for face in vision.faces(frame):
                match = identify(vision.feature(frame, face), roster, threshold)
                x, y, w, h = map(int, face[:4])
                label = "Unknown / uncertain"
                color = (0, 0, 255)
                if match:
                    sid, name, _, score = match
                    if sid not in visible:
                        consecutive[sid] = consecutive.get(sid, 0) + 1
                    visible.add(sid)
                    label = f"{name} ({score:.2f})"
                    color = (0, 180, 0)
                    if consecutive[sid] == 3 and mark(db, sid, score):
                        print(f"Present: {name} ({sid}), score {score:.3f}")
                cv.rectangle(preview, (x, y), (x+w, y+h), color, 2)
                cv.putText(preview, label, (x, max(20, y-8)),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            consecutive = {sid: count for sid, count in consecutive.items() if sid in visible}
            cv.imshow("Attendance - Q to quit", preview)
            if (cv.waitKey(1) & 0xFF) in (ord("q"), 27):
                break
    finally:
        camera.release()
        cv.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(DEFAULT_DB), help="Path to local SQLite database")
    sub = parser.add_subparsers(dest="command", required=True)
    e = sub.add_parser("enroll", help="Capture three samples of one consenting student's face")
    e.add_argument("student_id")
    e.add_argument("name")
    e.add_argument("class_name")
    e.add_argument("--camera", type=int, default=0)
    s = sub.add_parser("scan", help="Record today's attendance from webcam")
    s.add_argument("--camera", type=int, default=0)
    s.add_argument("--threshold", type=float, default=THRESHOLD)
    sub.add_parser("students", help="List enrolled students")
    report = sub.add_parser("report", help="Show or export daily attendance")
    report.add_argument("--day", default=datetime.now().astimezone().date().isoformat())
    report.add_argument("--csv", type=Path, help="Export to CSV")
    delete = sub.add_parser("delete", help="Delete a student's embedding and attendance")
    delete.add_argument("student_id")
    backup = sub.add_parser("backup", help="Make a verified database backup")
    backup.add_argument("--output", type=Path, help="New backup filename; must not exist")
    args = parser.parse_args()
    try:
        with connect(args.db) as db:
            if args.command == "enroll":
                feature = enroll_camera(Vision(), args.camera)
                enroll(db, args.student_id, args.name, args.class_name, feature)
                print(f"Enrolled {args.name} ({args.student_id})")
            elif args.command == "scan":
                if not 0 < args.threshold <= 1:
                    parser.error("--threshold must be greater than 0 and at most 1")
                scan_camera(Vision(), db, args.camera, args.threshold)
            elif args.command == "students":
                for sid, name, class_name, _ in students(db):
                    print(f"{sid}\t{name}\t{class_name}")
            elif args.command == "report":
                try:
                    args.day = datetime.strptime(args.day, "%Y-%m-%d").date().isoformat()
                except ValueError:
                    parser.error("--day must be YYYY-MM-DD")
                rows = db.execute("""SELECT s.student_id, s.name, s.class_name,
                    CASE WHEN a.student_id IS NULL THEN 'Absent' ELSE 'Present' END,
                    COALESCE(a.marked_at, ''), COALESCE(a.score, '')
                    FROM students s LEFT JOIN attendance a
                    ON s.student_id = a.student_id AND a.day = ?
                    ORDER BY s.class_name, s.name""", (args.day,)).fetchall()
                header = ("student_id", "name", "class", "status", "marked_at", "score")
                if args.csv:
                    args.csv.parent.mkdir(parents=True, exist_ok=True)
                    with args.csv.open("w", newline="", encoding="utf-8") as output:
                        writer = csv.writer(output)
                        writer.writerow(header)
                        writer.writerows(rows)
                    print(f"Exported {len(rows)} rows to {args.csv}")
                else:
                    print("\t".join(header))
                    for row in rows:
                        print("\t".join(map(str, row)))
            elif args.command == "delete":
                with db:
                    result = db.execute("DELETE FROM students WHERE student_id = ?", (args.student_id,))
                print("Deleted student and attendance" if result.rowcount else "Student not found")
            elif args.command == "backup":
                name = datetime.now().astimezone().strftime("attendance-%Y%m%d-%H%M%S.db")
                destination = args.output or (Path(args.db).parent / "backups" / name)
                print(f"Backup saved: {backup_database(db, destination)}")
    except (RuntimeError, ValueError, OSError, sqlite3.Error) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()

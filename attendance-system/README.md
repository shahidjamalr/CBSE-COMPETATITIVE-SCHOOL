# Face Attendance — local prototype

Enroll a student with a webcam, recognize enrolled faces in a live camera feed, and mark each student present at most once per day. Attendance is saved in a local SQLite database. The program does not upload images or keep the original camera frames. It stores face embeddings, which are still sensitive biometric data.

## Set up

Python 3.10+ and a webcam are needed. From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python download_models.py
```

The model download is a one-time step; recognition runs offline afterward. The models come from the [OpenCV Zoo YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) and [SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) projects. If the script cannot reach GitHub, download `face_detection_yunet_2023mar.onnx` and `face_recognition_sface_2021dec.onnx` from those pages and place them in `models/`.

## School staff interface

Set a long local staff password and start the browser interface:

```bash
export ATTENDANCE_ADMIN_PASSWORD='replace-this-with-a-long-private-password'
python webapp.py
```

Open `http://127.0.0.1:5000` on the **same computer**. Sign in, enroll students using their official IDs and class sections (for example, `10-A`), scan at the supervised entrance, and review or correct the daily register. Enrollment captures three samples and rejects a close match to an existing student ID. Corrections require a reason and are recorded in a local audit log. Attendance can be filtered by class and exported as CSV. The staff password is set in your terminal and is not committed to GitHub. Use a restricted school computer account and keep the database securely backed up. The interface binds only to `127.0.0.1`; it is not an internet service or a phone-accessible server.

Make a consistent backup before software updates and at least weekly:

```bash
python attendance.py backup
# Or write a new backup to a secure external drive:
python attendance.py backup --output /path/to/secure-drive/attendance-2026-11-29.db
```

Backups contain biometric embeddings and student records. The default backup stays on the same computer, so it does not protect against disk failure. Keep a second encrypted copy under school control, test restoration on a separate machine, and set a retention schedule. Do not put the database or any backup on GitHub.

## Command-line use

```bash
python attendance.py enroll STU001 "Asha Kumar" "Class 10"
python attendance.py enroll STU002 "Rahul Singh" "Class 10" --camera 1
python attendance.py students
python attendance.py scan
python attendance.py report
python attendance.py report --day 2026-11-28 --csv data/attendance-2026-11-28.csv
python attendance.py delete STU001
```

During enrollment, keep exactly one face visible and press **Space** three times with small angle changes. During scanning, press **Q** or **Esc** to stop. `--camera 1` selects another camera. Unknown or uncertain faces are not marked. A confident match must persist for three frames, and a student receives only one entry per calendar day according to the computer's local time. The daily report shows all enrolled students as Present or Absent. `delete` removes the student's embedding and attendance history.

For a separate database, pass `--db /path/to/attendance.db` before the command. The default is `data/attendance.db`. Keep that file restricted to authorized staff and back it up securely. Obtain informed consent (and guardian consent where appropriate) before enrollment. Provide a non-biometric attendance option and a process to correct mistakes.

## Limits and next steps

This is a prototype for a supervised check-in point, not a production attendance system. A printed photo or phone screen may fool face matching; the three-frame check is **not** liveness detection. A staff member should review attendance and correct false matches and missed matches before relying on records. Lighting, camera angle, masks, and similar-looking people affect accuracy. The `0.50` similarity threshold and `0.05` margin are starting values, not validated accuracy guarantees. Test with consenting volunteers and tune them before broader use. For deployment, add liveness checks, access control, encryption, audit logs, retention rules, and an administrator interface.

Run local logic tests with `python -m unittest discover -s tests -v`. Camera and model inference need to be tested on a machine with the installed dependencies, downloaded models, and an attached camera. The confidence threshold must be calibrated against consenting students in school lighting before official attendance use.

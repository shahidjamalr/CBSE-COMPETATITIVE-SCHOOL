"""Local staff interface for CBSE Competitive School face attendance."""

import base64
import csv
from datetime import date, datetime
from functools import wraps
import io
import os
import secrets
import sqlite3

from flask import Flask, Response, abort, jsonify, redirect, render_template_string, request, session, url_for
import numpy as np

from attendance import DEFAULT_DB, THRESHOLD, Vision, combine_samples, connect, correct_attendance, enroll, identify, mark, students


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>School Attendance</title><style>
body{font:16px system-ui,sans-serif;margin:0;background:#f3f8fc;color:#17324c}
header{background:#123b66;color:white;padding:16px max(20px,calc((100% - 960px)/2))}
main{max-width:960px;margin:24px auto;padding:0 18px}h1{margin:0;font-size:23px}
nav{display:flex;gap:16px;margin-top:12px;flex-wrap:wrap}a{color:#075996}header a{color:white}
.card{background:white;border:1px solid #d8e5ef;border-radius:12px;padding:20px;margin-bottom:20px}
input,select,button{font:inherit;padding:9px;border:1px solid #a9bfce;border-radius:6px;margin:4px}
button{background:#0864a2;color:white;border:0;cursor:pointer}button:disabled{opacity:.5}
video{width:100%;max-width:640px;background:#172b3d;border-radius:8px}
table{border-collapse:collapse;width:100%;overflow:auto}td,th{padding:9px;text-align:left;border-bottom:1px solid #dce6ee}
.notice{background:#e9f5e7;padding:12px;border-radius:7px}.error{background:#ffe9e7;padding:12px;border-radius:7px}
.scroll{overflow:auto}.small{font-size:13px;color:#49627a}
</style></head><body><header><h1>CBSE Competitive School · Mohania</h1>
{% if session.get('staff') %}<nav><a href="{{ url_for('home') }}">Dashboard</a>
<a href="{{ url_for('enroll_page') }}">Enroll</a><a href="{{ url_for('scan_page') }}">Scan</a>
<a href="{{ url_for('report_page') }}">Attendance</a><a href="{{ url_for('logout') }}">Log out</a></nav>{% endif %}
</header><main>{% if message %}<p class="{{ 'error' if error else 'notice' }}">{{ message }}</p>{% endif %}
{{ content|safe }}</main></body></html>"""

LOGIN = """<section class="card"><h2>Staff sign in</h2><form method="post">
<input type="password" name="password" required placeholder="Staff password" autocomplete="current-password">
<button>Sign in</button></form><p class="small">This application runs on this computer only.</p></section>"""

HOME = """<section class="card"><h2>Today's attendance</h2>
<p><strong>{{ total }}</strong> enrolled · <strong>{{ present }}</strong> present ·
<strong>{{ total - present }}</strong> pending or absent</p>
<p>1. Enroll students with permission. 2. Scan at the supervised entrance.
3. Review and correct the daily attendance before using it.</p>
<p class="small">Face matching can make mistakes and a photo may fool the camera. Keep a manual option.</p></section>"""

ENROLL = """<section class="card"><h2>Enroll a student</h2>
<p>Only one student should be in the frame. We capture three samples with small angle changes. Use the student's official school ID.</p>
<form id="enroll"><input id="sid" required maxlength="60" placeholder="Student ID">
<input id="name" required maxlength="120" placeholder="Full name">
<input id="class" required maxlength="60" placeholder="Class / section (e.g. 10-A)">
<label><input type="checkbox" id="consent" required> Student/guardian consent is recorded</label><br>
<button type="button" onclick="startCamera()">Start camera</button>
<button type="submit">Capture 3 samples and enroll</button></form>
<video id="video" autoplay playsinline muted></video><p id="result" role="status"></p></section>
<section class="card scroll"><h2>Enrolled students</h2><table><tr><th>ID</th><th>Name</th><th>Class</th><th>Action</th></tr>
{% for sid, name, cls, _ in roster %}<tr><td>{{ sid }}</td><td>{{ name }}</td><td>{{ cls }}</td>
<td><form method="post" action="{{ url_for('delete_student') }}" onsubmit="return confirm('Delete this student and attendance history?')">
<input type="hidden" name="csrf" value="{{ csrf }}"><input type="hidden" name="student_id" value="{{ sid }}">
<button>Delete</button></form></td></tr>{% endfor %}</table></section>"""

SCAN = """<section class="card"><h2>Live check-in · {{ today }}</h2>
<p>Keep the camera at the attended school entrance. A match must appear in three scans.</p>
<button onclick="startCamera()">Start camera</button><button onclick="stopCamera()">Stop camera</button><br>
<video id="video" autoplay playsinline muted></video><p id="result" role="status"></p>
<p class="small">A face match is provisional. Review the attendance register after scanning.</p></section>"""

REPORT = """<section class="card"><h2>Attendance register</h2>
<form method="get"><label>Date <input name="day" type="date" value="{{ day }}"></label>
<label>Class <select name="class"><option value="">All classes</option>
{% for cls in classes %}<option value="{{ cls }}" {{ 'selected' if cls == selected_class else '' }}>{{ cls }}</option>{% endfor %}
</select></label><button>View</button></form>
<a href="{{ url_for('export_csv', day=day, **{'class': selected_class}) }}">Export CSV</a>
<div class="scroll"><table><tr><th>ID</th><th>Name</th><th>Class</th><th>Status</th><th>Time</th><th>Correction</th></tr>
{% for sid, name, cls, status, marked_at, score in rows %}<tr><td>{{ sid }}</td><td>{{ name }}</td><td>{{ cls }}</td>
<td>{{ status }}{{ ' (staff)' if score == -1 else '' }}</td><td>{{ marked_at[11:19] if marked_at else '' }}</td><td>
<form method="post" action="{{ url_for('correction') }}">
<input type="hidden" name="csrf" value="{{ csrf }}"><input type="hidden" name="student_id" value="{{ sid }}">
<input type="hidden" name="day" value="{{ day }}"><input type="hidden" name="class" value="{{ selected_class }}">
<input name="reason" required maxlength="200" placeholder="Reason" aria-label="Reason for correction">
<button name="status" value="present">Present</button><button name="status" value="absent">Absent</button>
</form></td></tr>{% endfor %}</table></div></section>
<section class="card"><h3>Corrections log · {{ day }}</h3><div class="scroll"><table>
<tr><th>Time</th><th>ID</th><th>Change</th><th>Reason</th></tr>
{% for at, sid, action, note in audit %}<tr><td>{{ at }}</td><td>{{ sid }}</td><td>{{ action }}</td><td>{{ note }}</td></tr>{% endfor %}
</table></div></section>"""

CAMERA_JS = """<script>
let stream=null, timer=null;
const video=document.querySelector('#video'), result=document.querySelector('#result');
const csrf={{ csrf|tojson }};
async function startCamera(){
 try {if(stream) return; stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:'user',width:640},audio:false});
 video.srcObject=stream; result.textContent='Camera ready';
 if(location.pathname==='/scan') timer=setInterval(scan,850);
 } catch(e){result.textContent='Camera unavailable: '+e.message;}
}
function stopCamera(){clearInterval(timer);timer=null;if(stream){stream.getTracks().forEach(t=>t.stop());stream=null;video.srcObject=null;}}
function snapshot(){if(!stream||!video.videoWidth) throw Error('Start camera first');
 const c=document.createElement('canvas');c.width=video.videoWidth;c.height=video.videoHeight;
 c.getContext('2d').drawImage(video,0,0);return c.toDataURL('image/jpeg',.75);}
async function postFrame(path,payload){const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json','X-CSRF-Token':csrf},body:JSON.stringify(payload)});
 const data=await r.json();if(!r.ok)throw Error(data.error||'Request failed');return data;}
async function scan(){if(!stream||window.busy)return;window.busy=true;
 try{let d=await postFrame('/api/scan',{image:snapshot()});
 result.textContent=d.matches.length?d.matches.map(x=>x.name+' — '+x.status).join(' | '):'No confident enrolled face';
 }catch(e){result.textContent=e.message;}finally{window.busy=false;}}
const form=document.querySelector('#enroll');if(form) form.addEventListener('submit',async e=>{e.preventDefault();
 if(!document.querySelector('#consent').checked)return;
 try{let images=[];for(let i=0;i<3;i++){images.push(snapshot());result.textContent='Captured '+(i+1)+'/3';
 if(i<2)await new Promise(resolve=>setTimeout(resolve,650));}
 let d=await postFrame('/api/enroll',{images,student_id:document.querySelector('#sid').value,
 name:document.querySelector('#name').value,class_name:document.querySelector('#class').value,
 consent:document.querySelector('#consent').checked});
 result.textContent=d.message;stopCamera();setTimeout(()=>location.reload(),1200);
 }catch(err){result.textContent=err.message;}});
window.addEventListener('pagehide',stopCamera);
</script>"""


def create_app(db_path=DEFAULT_DB, vision_factory=Vision):
    app = Flask(__name__)
    password = os.environ.get("ATTENDANCE_ADMIN_PASSWORD", "")
    if len(password) < 12:
        raise RuntimeError("Set ATTENDANCE_ADMIN_PASSWORD to at least 12 characters")
    app.secret_key = os.environ.get("ATTENDANCE_SECRET_KEY") or secrets.token_hex(32)
    app.config.update(MAX_CONTENT_LENGTH=2 * 1024 * 1024, SESSION_COOKIE_HTTPONLY=True,
                      SESSION_COOKIE_SAMESITE="Strict")
    counters = {}
    vision = None

    def get_vision():
        nonlocal vision
        if vision is None:
            vision = vision_factory()
        return vision

    def page(body, message=None, error=False, **context):
        token = session.setdefault("csrf", secrets.token_urlsafe(32))
        context["csrf"] = token
        return render_template_string(PAGE, content=render_template_string(body, **context),
                                      message=message, error=error)

    def staff(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            if not session.get("staff"):
                if request.path.startswith("/api/"):
                    return jsonify(error="Sign in first"), 401
                return redirect(url_for("login"))
            return fn(*args, **kwargs)
        return wrapped

    def check_csrf():
        supplied = request.headers.get("X-CSRF-Token") or request.form.get("csrf", "")
        if not secrets.compare_digest(supplied, session.get("csrf", "")):
            abort(403)

    def valid_day(value):
        try:
            return date.fromisoformat(value).isoformat()
        except (TypeError, ValueError):
            abort(400)

    def report_rows(db, day, cls):
        return db.execute("""SELECT s.student_id, s.name, s.class_name,
            CASE WHEN a.student_id IS NULL THEN 'Absent' ELSE 'Present' END,
            COALESCE(a.marked_at,''), COALESCE(a.score,'')
            FROM students s LEFT JOIN attendance a ON s.student_id=a.student_id AND a.day=?
            WHERE (?='' OR s.class_name=?) ORDER BY s.class_name,s.name""",
                          (day, cls, cls)).fetchall()

    def decode_image():
        payload = request.get_json(silent=True) or {}
        encoded = payload.get("image", "")
        if not isinstance(encoded, str) or not encoded.startswith("data:image/jpeg;base64,"):
            abort(400)
        try:
            raw = base64.b64decode(encoded.split(",", 1)[1], validate=True)
            if len(raw) > 1_500_000:
                abort(413)
            cv = get_vision().cv
            frame = cv.imdecode(np.frombuffer(raw, dtype=np.uint8), cv.IMREAD_COLOR)
            if frame is None:
                abort(400)
            return payload, frame
        except (ValueError, base64.binascii.Error):
            abort(400)

    def decode_samples():
        payload = request.get_json(silent=True) or {}
        encoded = payload.get("images")
        if not isinstance(encoded, list) or len(encoded) != 3:
            abort(400)
        frames = []
        cv = get_vision().cv
        for image in encoded:
            if not isinstance(image, str) or not image.startswith("data:image/jpeg;base64,"):
                abort(400)
            try:
                raw = base64.b64decode(image.split(",", 1)[1], validate=True)
            except (ValueError, base64.binascii.Error):
                abort(400)
            frame = cv.imdecode(np.frombuffer(raw, dtype=np.uint8), cv.IMREAD_COLOR)
            if frame is None:
                abort(400)
            frames.append(frame)
        return payload, frames

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            if secrets.compare_digest(request.form.get("password", ""), password):
                session.clear()
                session["staff"] = True
                session["csrf"] = secrets.token_urlsafe(32)
                session["key"] = secrets.token_urlsafe(16)
                return redirect(url_for("home"))
            return page(LOGIN, message="Incorrect password", error=True)
        return page(LOGIN)

    @app.route("/logout")
    def logout():
        counters.pop(session.get("key"), None)
        session.clear()
        return redirect(url_for("login"))

    @app.route("/")
    @staff
    def home():
        with connect(db_path) as db:
            total = db.execute("SELECT COUNT(*) FROM students").fetchone()[0]
            present = db.execute("SELECT COUNT(*) FROM attendance WHERE day=?", (date.today().isoformat(),)).fetchone()[0]
        return page(HOME, total=total, present=present)

    @app.route("/enroll")
    @staff
    def enroll_page():
        with connect(db_path) as db:
            roster = students(db)
        return page(ENROLL + CAMERA_JS, roster=roster)

    @app.route("/scan")
    @staff
    def scan_page():
        return page(SCAN + CAMERA_JS, today=date.today().isoformat())

    @app.route("/attendance")
    @staff
    def report_page():
        day = valid_day(request.args.get("day", date.today().isoformat()))
        cls = request.args.get("class", "")
        with connect(db_path) as db:
            rows = report_rows(db, day, cls)
            classes = [r[0] for r in db.execute("SELECT DISTINCT class_name FROM students ORDER BY class_name")]
            audit = db.execute("SELECT at,student_id,action,note FROM audit WHERE day=? ORDER BY id DESC", (day,)).fetchall()
        return page(REPORT, day=day, selected_class=cls, rows=rows, classes=classes, audit=audit)

    @app.route("/attendance.csv")
    @staff
    def export_csv():
        day = valid_day(request.args.get("day", date.today().isoformat()))
        cls = request.args.get("class", "")
        with connect(db_path) as db:
            rows = report_rows(db, day, cls)
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(("student_id", "name", "class", "status", "marked_at", "score"))
        writer.writerows(rows)
        return Response(output.getvalue(), mimetype="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="attendance-{day}.csv"'})

    @app.route("/correct", methods=["POST"])
    @staff
    def correction():
        check_csrf()
        day = valid_day(request.form.get("day"))
        status = request.form.get("status")
        if status not in ("present", "absent"):
            abort(400)
        try:
            with connect(db_path) as db:
                correct_attendance(db, request.form.get("student_id", ""), day,
                                   status == "present", request.form.get("reason", ""))
        except (ValueError, sqlite3.IntegrityError):
            abort(400)
        return redirect(url_for("report_page", day=day, **{"class": request.form.get("class", "")}))

    @app.route("/delete", methods=["POST"])
    @staff
    def delete_student():
        check_csrf()
        with connect(db_path) as db:
            with db:
                db.execute("DELETE FROM students WHERE student_id=?", (request.form.get("student_id", ""),))
        return redirect(url_for("enroll_page"))

    @app.route("/api/enroll", methods=["POST"])
    @staff
    def api_enroll():
        check_csrf()
        payload, frames = decode_samples()
        if payload.get("consent") is not True:
            return jsonify(error="Record consent before enrolling"), 400
        try:
            features = []
            for frame in frames:
                faces = get_vision().faces(frame)
                if len(faces) != 1:
                    return jsonify(error="Exactly one face must be visible in each sample"), 400
                features.append(get_vision().feature(frame, faces[0]))
            with connect(db_path) as db:
                enroll(db, str(payload.get("student_id", "")), str(payload.get("name", "")),
                       str(payload.get("class_name", "")), combine_samples(features))
        except sqlite3.IntegrityError:
            return jsonify(error="Student ID already exists"), 409
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        return jsonify(message="Student enrolled. Review attendance after scanning.")

    @app.route("/api/scan", methods=["POST"])
    @staff
    def api_scan():
        check_csrf()
        _, frame = decode_image()
        key = session["key"]
        today = date.today().isoformat()
        state = counters.setdefault(key, {"day": today, "counts": {}})
        if state["day"] != today:
            state = counters[key] = {"day": today, "counts": {}}
        previous = state["counts"]
        found = set()
        matches = []
        with connect(db_path) as db:
            roster = students(db)
            for face in get_vision().faces(frame):
                match = identify(get_vision().feature(frame, face), roster, THRESHOLD)
                if match is None:
                    continue
                sid, name, _, score = match
                if sid in found:
                    continue
                found.add(sid)
                previous[sid] = previous.get(sid, 0) + 1
                status = "verifying"
                if previous[sid] >= 3:
                    status = "marked present" if mark(db, sid, score) else "already present"
                matches.append({"name": name, "status": status})
        state["counts"] = {sid: count for sid, count in previous.items() if sid in found}
        return jsonify(matches=matches)

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000, debug=False)

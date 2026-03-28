"""
CIS3425 Smart Home Security System
Combined Server — Face Recognition + Number Plate Detection
Features: Login, Multi-Camera, Settings Panel, Enrolment, Dev Panel

Run:
  python main_server.py
  Open: http://localhost:5000
  Default login: admin / admin123
"""

import cv2
import face_recognition
import easyocr
import numpy as np
import sqlite3
import os, re, time, logging, threading, shutil, hashlib, json
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
from flask import (Flask, Response, render_template_string, jsonify,
                   request, send_file, session, redirect, url_for)
from functools import wraps

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Default Configuration (saved/loaded from config.json) ─────────────────────
DEFAULT_CONFIG = {
    "face_model":            "yolo26n.pt",
    "plate_model":           "plate_detector.pt",
    "db_path":               "security.db",
    "log_frames_dir":        "logged_frames",
    "known_faces_dir":       "known_faces",
    "unknowns_dir":          "unknowns",
    "known_plates_file":     "known_plates.txt",
    "face_confidence":       0.50,
    "plate_confidence":      0.45,
    "recognition_tolerance": 0.50,
    "face_frame_skip":       2,
    "plate_frame_skip":      3,
    "unknown_cooldown":      15,
    "plate_cooldown":        15,
    "ocr_languages":         ["en"],
    "ocr_min_chars":         4,
    "ocr_confidence":        0.25,
    "flask_port":            5000,
    "flask_host":            "0.0.0.0",
    "flask_secret":          "cis3425-security-secret-key",
    # Multi-camera: list of {id, name, source, role}
    # source: 0,1,2 for USB cameras, or "rtsp://..." for IP cameras
    "cameras": [
        {"id": "cam0", "name": "Front Door",  "source": 0,   "role": "face+plate", "enabled": True},
    ],
    # Login credentials (hashed passwords)
    "users": {
        "admin": {"password_hash": "", "role": "admin"},
    },
}

CONFIG_FILE = "config.json"


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def load_config() -> dict:
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            cfg = json.load(f)
        # Merge with defaults to catch new keys
        for k, v in DEFAULT_CONFIG.items():
            if k not in cfg:
                cfg[k] = v
        return cfg
    # First run — set default admin password
    cfg = dict(DEFAULT_CONFIG)
    cfg["users"]["admin"]["password_hash"] = hash_password("admin123")
    save_config(cfg)
    return cfg


def save_config(cfg: dict):
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)
    log.info("Configuration saved.")


CONFIG = load_config()


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED STATE
# ═══════════════════════════════════════════════════════════════════════════════

class SharedState:
    def __init__(self):
        self.lock             = threading.Lock()
        # Multi-camera frames: {cam_id: frame}
        self.raw_frames       = {}
        self.combined_frames  = {}
        self.feed_frames      = {}  # Dedicated feed buffer — updated by camera thread directly
        self.face_detections  = []
        self.plate_detections = []
        self.unknown_faces    = []
        self.unknown_plates   = []
        self.running          = True
        self.known_encodings  = []
        self.known_plates     = set()
        self.perf             = {
            "face_fps": 0.0, "plate_fps": 0.0,
            "face_inference": 0.0, "plate_inference": 0.0,
            "total_faces": 0, "total_plates": 0,
            "known_faces_count": 0, "unknown_faces_count": 0,
            "uptime_start": time.time(),
            "active_cameras": 0,
        }

    def update_raw(self, cam_id, frame):
        with self.lock:
            self.raw_frames[cam_id] = frame.copy()
            # Only set feed frame if no annotated frame exists yet
            if cam_id not in self.feed_frames:
                self.feed_frames[cam_id] = frame.copy()

    def get_raw(self, cam_id):
        with self.lock:
            f = self.raw_frames.get(cam_id)
            return f.copy() if f is not None else None

    def get_any_raw(self):
        with self.lock:
            for f in self.raw_frames.values():
                if f is not None:
                    return f.copy()
            return None

    def update_combined(self, cam_id, frame):
        with self.lock:
            self.combined_frames[cam_id] = frame.copy()
            # Update feed buffer with annotated frame
            self.feed_frames[cam_id] = frame.copy()

    def get_feed_frame(self, cam_id):
        with self.lock:
            f = self.feed_frames.get(cam_id)
            return f.copy() if f is not None else None

    def get_combined_any(self):
        with self.lock:
            for f in self.combined_frames.values():
                if f is not None:
                    return f.copy()
            return None

    def add_face_event(self, event):
        with self.lock:
            self.face_detections.insert(0, event)
            self.face_detections = self.face_detections[:50]

    def add_plate_event(self, event):
        with self.lock:
            self.plate_detections.insert(0, event)
            self.plate_detections = self.plate_detections[:50]

    def add_unknown_face(self, uid, path, timestamp, cam_id):
        with self.lock:
            self.unknown_faces.insert(0, {"id": uid, "path": path,
                                          "time": timestamp, "camera": cam_id})
            self.unknown_faces = self.unknown_faces[:20]

    def add_unknown_plate(self, uid, plate, path, timestamp, cam_id):
        with self.lock:
            self.unknown_plates.insert(0, {"id": uid, "plate": plate,
                                           "path": path, "time": timestamp,
                                           "camera": cam_id})
            self.unknown_plates = self.unknown_plates[:20]

    def remove_unknown_face(self, uid):
        with self.lock:
            self.unknown_faces = [f for f in self.unknown_faces if f["id"] != uid]

    def remove_unknown_plate(self, uid):
        with self.lock:
            self.unknown_plates = [p for p in self.unknown_plates if p["id"] != uid]

    def reload_encodings(self, encodings):
        with self.lock:
            self.known_encodings = encodings

    def reload_plates(self, plates):
        with self.lock:
            self.known_plates = plates

    def update_perf(self, key, value):
        with self.lock:
            self.perf[key] = value

    def increment_perf(self, key):
        with self.lock:
            self.perf[key] = self.perf.get(key, 0) + 1

    def get_perf(self):
        with self.lock:
            p = dict(self.perf)
            s = int(time.time() - p["uptime_start"])
            p["uptime"] = f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}"
            return p

    def get_recent_events(self):
        with self.lock:
            return {"faces": self.face_detections[:10],
                    "plates": self.plate_detections[:10]}

    def get_unknowns(self):
        with self.lock:
            return {"faces": list(self.unknown_faces),
                    "plates": list(self.unknown_plates)}


# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════════════════════════════════════════

class SecurityDatabase:
    def __init__(self, db_path):
        self.db_path = db_path
        self._init_db()

    def _conn(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _init_db(self):
        c = self._conn()
        c.executescript("""
            CREATE TABLE IF NOT EXISTS known_faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_name TEXT NOT NULL, encoding BLOB NOT NULL,
                source_img TEXT, enrolled_at TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS face_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL, person_name TEXT NOT NULL,
                confidence REAL, frame_path TEXT, camera_id TEXT
            );
            CREATE TABLE IF NOT EXISTS plate_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL, plate_text TEXT NOT NULL,
                ocr_conf REAL, is_known INTEGER DEFAULT 0,
                frame_path TEXT, camera_id TEXT
            );
            CREATE TABLE IF NOT EXISTS known_plates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_text TEXT UNIQUE NOT NULL,
                added_at TEXT DEFAULT (datetime('now'))
            );
        """)
        c.commit(); c.close()
        log.info(f"Database ready: {self.db_path}")

    def save_face_encoding(self, name, encoding, source=""):
        c = self._conn()
        c.execute("INSERT INTO known_faces (person_name,encoding,source_img) VALUES (?,?,?)",
                  (name, encoding.tobytes(), source))
        c.commit(); c.close()

    def load_face_encodings(self):
        c = self._conn()
        rows = c.execute("SELECT person_name,encoding FROM known_faces").fetchall()
        c.close()
        return [{"name": n, "encoding": np.frombuffer(e, dtype=np.float64)} for n,e in rows]

    def clear_face_encodings(self):
        c = self._conn(); c.execute("DELETE FROM known_faces"); c.commit(); c.close()

    def log_face(self, name, confidence, frame_path, cam_id=""):
        c = self._conn()
        c.execute("INSERT INTO face_log (timestamp,person_name,confidence,frame_path,camera_id) VALUES (?,?,?,?,?)",
                  (datetime.now().isoformat(), name, confidence, frame_path, cam_id))
        c.commit(); c.close()

    def log_plate(self, plate, conf, is_known, frame_path, cam_id=""):
        c = self._conn()
        c.execute("INSERT INTO plate_log (timestamp,plate_text,ocr_conf,is_known,frame_path,camera_id) VALUES (?,?,?,?,?,?)",
                  (datetime.now().isoformat(), plate, conf, int(is_known), frame_path, cam_id))
        c.commit(); c.close()

    def load_known_plates(self):
        c = self._conn()
        rows = c.execute("SELECT plate_text FROM known_plates").fetchall()
        c.close()
        return {r[0].upper() for r in rows}

    def add_known_plate(self, plate):
        c = self._conn()
        c.execute("INSERT OR IGNORE INTO known_plates (plate_text) VALUES (?)",
                  (plate.upper().strip(),))
        c.commit(); c.close()

    def get_recent_logs(self, limit=50):
        c = self._conn()
        faces  = c.execute("SELECT timestamp,person_name,confidence,camera_id FROM face_log ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        plates = c.execute("SELECT timestamp,plate_text,ocr_conf,is_known,camera_id FROM plate_log ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        c.close()
        return {
            "faces":  [{"time":r[0][:19],"name":r[1],"conf":round(r[2] or 0,3),"camera":r[3]} for r in faces],
            "plates": [{"time":r[0][:19],"plate":r[1],"conf":round(r[2] or 0,3),"known":bool(r[3]),"camera":r[4]} for r in plates]
        }


# ═══════════════════════════════════════════════════════════════════════════════
# ENROLMENT
# ═══════════════════════════════════════════════════════════════════════════════

def enrol_faces_from_dir(db, known_faces_dir, force=False):
    if force:
        db.clear_face_encodings()
    enrolled = 0
    path = Path(known_faces_dir)
    if not path.exists():
        return 0
    for person_dir in sorted(path.iterdir()):
        if not person_dir.is_dir():
            continue
        name = person_dir.name
        for img_path in (list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png"))):
            try:
                img  = face_recognition.load_image_file(str(img_path))
                encs = face_recognition.face_encodings(img)
                if encs:
                    db.save_face_encoding(name, encs[0], str(img_path))
                    enrolled += 1
            except Exception as e:
                log.warning(f"Enrolment failed {img_path.name}: {e}")
    log.info(f"Enrolled {enrolled} face encodings.")
    return enrolled


def enrol_face_from_crop(db, state, crop_path, name):
    try:
        img_bgr = cv2.imread(crop_path)
        if img_bgr is None:
            return False, "Could not read image"
        h, w = img_bgr.shape[:2]
        if w < 150 or h < 150:
            scale   = max(150/w, 150/h)
            img_bgr = cv2.resize(img_bgr, (int(w*scale), int(h*scale)),
                                 interpolation=cv2.INTER_CUBIC)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        encs    = face_recognition.face_encodings(img_rgb)
        if not encs:
            encs = face_recognition.face_encodings(img_rgb, num_jitters=3, model="large")
        if not encs:
            return False, "No face found — try a clearer photo or stand closer"
        person_dir = Path(CONFIG["known_faces_dir"]) / name
        person_dir.mkdir(parents=True, exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = str(person_dir / f"{name}_enrolled_{ts}.jpg")
        shutil.copy2(crop_path, dest)
        db.save_face_encoding(name, encs[0], dest)
        state.reload_encodings(db.load_face_encodings())
        log.info(f"Enrolled: {name}")
        return True, f"✅ {name} enrolled — will be recognised immediately"
    except Exception as e:
        return False, str(e)


# ═══════════════════════════════════════════════════════════════════════════════
# CAMERA THREAD (one per camera)
# ═══════════════════════════════════════════════════════════════════════════════

def camera_thread(state, cam_cfg):
    cam_id  = cam_cfg["id"]
    source  = cam_cfg["source"]
    name    = cam_cfg["name"]

    # Use DirectShow backend on Windows for better performance
    if isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        log.error(f"Cannot open camera [{name}]: {source}")
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # ── Recording setup ───────────────────────────────────────────────────────
    recordings_dir = os.path.join("recordings", cam_id)
    os.makedirs(recordings_dir, exist_ok=True)

    fps            = 20                        # Recording FPS
    chunk_minutes  = 10                        # Start new file every N minutes
    chunk_seconds  = chunk_minutes * 60
    fourcc         = cv2.VideoWriter_fourcc(*"mp4v")
    writer         = None
    chunk_start    = time.time()
    frame_w, frame_h = 640, 480

    def new_writer():
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(recordings_dir, f"{cam_id}_{ts}.mp4")
        w        = cv2.VideoWriter(filepath, fourcc, fps, (frame_w, frame_h))
        log.info(f"Recording started: {filepath}")
        return w, time.time()

    writer, chunk_start = new_writer()

    state.increment_perf("active_cameras")
    log.info(f"Camera [{name}] started — source: {source}")

    while state.running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        state.update_raw(cam_id, frame)

        # ── Write to recording ─────────────────────────────────────────────
        if writer is not None:
            # Add timestamp overlay to recording
            rec_frame = frame.copy()
            ts_text   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(rec_frame, f"{name} | {ts_text}",
                        (8, frame_h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255,255,255), 1)
            writer.write(rec_frame)

            # Rotate file every chunk_minutes
            if time.time() - chunk_start >= chunk_seconds:
                writer.release()
                writer, chunk_start = new_writer()

    # Finalise recording
    if writer is not None:
        writer.release()
        log.info(f"Recording finalised for [{name}]")

    cap.release()
    log.info(f"Camera [{name}] stopped.")


# ═══════════════════════════════════════════════════════════════════════════════
# FACE RECOGNITION THREAD
# ═══════════════════════════════════════════════════════════════════════════════

def face_thread(state, db, cfg):
    log.info("Starting face recognition thread (dlib HOG — no YOLO needed)...")
    os.makedirs(cfg["log_frames_dir"], exist_ok=True)
    os.makedirs(os.path.join(cfg["unknowns_dir"], "faces"), exist_ok=True)

    cooldowns   = {}
    fps_counter = 0
    fps_timer   = time.time()

    log.info("Face recognition thread started.")
    log.info("Face thread waiting for camera frames...")
    while state.running:
        if state.raw_frames:
            break
        time.sleep(0.1)
    log.info("Face thread got first frame — starting recognition.")

    cam_frame_counts = {}
    last_process_time = {}

    while state.running:
        active_cams = [c for c in cfg["cameras"] if c.get("enabled", True)]
        if not active_cams:
            time.sleep(0.1)
            continue

        for cam in active_cams:
            cam_id = cam["id"]
            frame  = state.get_raw(cam_id)
            if frame is None:
                continue

            now  = time.time()
            last = last_process_time.get(cam_id, 0)
            if now - last < 0.2:  # max 5fps for dlib
                continue
            last_process_time[cam_id] = now

            try:
                annotated = frame.copy()
                rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # dlib finds faces directly — no YOLO needed
                face_locs = face_recognition.face_locations(rgb, model="hog")

                if face_locs:
                    face_encs = face_recognition.face_encodings(
                        rgb, face_locs, num_jitters=1, model="small"
                    )
                    with state.lock:
                        known = list(state.known_encodings)

                    for (top, right, bottom, left), enc in zip(face_locs, face_encs):
                        name, similarity = "Unknown", 0.0
                        if known:
                            known_encs  = [e["encoding"] for e in known]
                            known_names = [e["name"]     for e in known]
                            dists       = face_recognition.face_distance(known_encs, enc)
                            best_idx    = int(np.argmin(dists))
                            similarity  = round(1.0 - float(dists[best_idx]), 3)
                            matches     = face_recognition.compare_faces(
                                known_encs, enc,
                                tolerance=cfg["recognition_tolerance"]
                            )
                            if matches[best_idx]:
                                name = known_names[best_idx]

                        # Draw tight face box only — no person box
                        colour = (0,200,0) if name != "Unknown" else (0,0,220)
                        label  = f"{name} {similarity:.2f}"
                        cv2.rectangle(annotated,(left,top),(right,bottom),colour,2)
                        label_y  = top - 5 if top > 30 else bottom + 20
                        lbg_y1   = top - 22 if top > 30 else bottom
                        lbg_y2   = top if top > 30 else bottom + 22
                        label_x2 = min(frame.shape[1], left + len(label)*11)
                        cv2.rectangle(annotated,(left,lbg_y1),(label_x2,lbg_y2),colour,-1)
                        cv2.putText(annotated,label,(left+2,label_y),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

                        # Rate-limited logging and saving
                        key      = f"{cam_id}_{left//50}_{top//50}"
                        last_log = cooldowns.get(key, 0)
                        if time.time() - last_log < cfg["unknown_cooldown"]:
                            continue
                        cooldowns[key] = time.time()

                        ts         = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        frame_path = os.path.join(cfg["log_frames_dir"],
                                                  f"face_{name}_{ts}.jpg")
                        cv2.imwrite(frame_path, annotated)
                        db.log_face(name, similarity, frame_path, cam_id)

                        # Save padded face crop
                        pad  = 20
                        spx1 = max(0, left-pad);   spy1 = max(0, top-pad)
                        spx2 = min(frame.shape[1], right+pad)
                        spy2 = min(frame.shape[0], bottom+pad)
                        padded = frame[spy1:spy2, spx1:spx2]

                        if name == "Unknown":
                            uid       = ts
                            crop_path = os.path.join(cfg["unknowns_dir"], "faces",
                                                     f"unknown_{uid}.jpg")
                            cv2.imwrite(crop_path, padded)
                            log.info(f"Saved unknown face: {crop_path}")
                            state.add_unknown_face(uid, crop_path,
                                                   datetime.now().strftime("%H:%M:%S"),
                                                   cam_id)
                            state.increment_perf("unknown_faces_count")
                        else:
                            person_dir = os.path.join(cfg["known_faces_dir"], name)
                            os.makedirs(person_dir, exist_ok=True)
                            cv2.imwrite(os.path.join(person_dir,
                                        f"{name}_auto_{ts}.jpg"), padded)
                            state.increment_perf("known_faces_count")

                        state.add_face_event({
                            "time":       datetime.now().strftime("%H:%M:%S"),
                            "name":       name,
                            "confidence": similarity,
                            "known":      name != "Unknown",
                            "camera":     cam["name"]
                        })
                        state.increment_perf("total_faces")

                state.update_combined(cam_id, annotated)

            except Exception as e:
                log.error(f"Face thread error [{cam_id}]: {e}")

        fps_counter += 1
        if time.time() - fps_timer >= 1.0:
            state.update_perf("face_fps", fps_counter)
            fps_counter = 0
            fps_timer   = time.time()
        time.sleep(0.01)

    log.info("Face recognition thread stopped.")


# ═══════════════════════════════════════════════════════════════════════════════
# NUMBER PLATE THREAD
# ═══════════════════════════════════════════════════════════════════════════════

def plate_thread(state, db, cfg):
    log.info("Loading plate model...")
    yolo = YOLO(cfg["plate_model"])
    log.info("Loading EasyOCR...")
    reader = easyocr.Reader(cfg["ocr_languages"], gpu=True)
    os.makedirs(os.path.join(cfg["unknowns_dir"], "plates"), exist_ok=True)
    os.makedirs(os.path.join(cfg["log_frames_dir"], "plates"), exist_ok=True)

    cooldowns   = {}
    frame_count = 0
    fps_counter = 0
    fps_timer   = time.time()

    def read_plate(crop):
        h, w = crop.shape[:2]
        if w < 100:
            scale = 100/w
            crop  = cv2.resize(crop,(int(w*scale),int(h*scale)),
                               interpolation=cv2.INTER_CUBIC)
        grey  = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        grey  = clahe.apply(grey)
        _, binary = cv2.threshold(grey,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        try:
            results = reader.readtext(binary)
            texts, confs = [], []
            for (_,text,conf) in results:
                clean = re.sub(r"[^A-Za-z0-9]","",text).upper()
                if clean: texts.append(clean); confs.append(conf)
            if texts:
                return "".join(texts), float(np.mean(confs))
        except Exception:
            pass
        return "", 0.0

    log.info("Number plate thread started.")
    # Wait until at least one camera has a frame
    log.info("Plate thread waiting for camera frames...")
    while state.running:
        if any(f is not None for f in state.raw_frames.values()):
            break
        time.sleep(0.1)
    log.info("Plate thread got first frame — starting detection.")
    while state.running:
        active_cams = [c for c in cfg["cameras"] if c.get("enabled", True)
                       and c.get("role","") in ("plate","face+plate")]
        if not active_cams:
            time.sleep(0.1); continue

        for cam in active_cams:
            cam_id = cam["id"]
            frame  = state.get_raw(cam_id)
            if frame is None:
                continue

            frame_count += 1
            if frame_count % cfg["plate_frame_skip"] != 0:
                continue

            t0        = time.time()
            results   = yolo(frame, verbose=False)[0]
            combined  = state.get_feed_frame(cam_id)
            annotated = combined if combined is not None else frame.copy()

            for box in results.boxes:
                conf = float(box.conf[0])
                if conf < cfg["plate_confidence"]:
                    continue
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                x1=max(0,x1); y1=max(0,y1)
                x2=min(frame.shape[1],x2); y2=min(frame.shape[0],y2)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                plate_text, ocr_conf = read_plate(crop)
                if (len(plate_text) < cfg["ocr_min_chars"] or
                        ocr_conf < cfg["ocr_confidence"]):
                    continue

                with state.lock:
                    is_known = plate_text.upper().replace(" ","") in state.known_plates
                colour = (0,200,0) if is_known else (0,0,220)
                label  = f"{plate_text} [{'KNOWN' if is_known else 'UNKNOWN'}]"
                cv2.rectangle(annotated,(x1,y1),(x2,y2),colour,2)
                cv2.rectangle(annotated,(x1,y2),(x1+len(label)*10,y2+22),colour,-1)
                cv2.putText(annotated,label,(x1+2,y2+16),
                            cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),2)

                key  = plate_text.upper().replace(" ","")
                last = cooldowns.get(f"{cam_id}_{key}", 0)
                if time.time() - last < cfg["plate_cooldown"]:
                    continue
                cooldowns[f"{cam_id}_{key}"] = time.time()

                ts         = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                frame_path = os.path.join(cfg["log_frames_dir"],"plates",
                                          f"{plate_text}_{ts}.jpg")
                cv2.imwrite(frame_path, annotated)
                db.log_plate(plate_text, ocr_conf, is_known, frame_path, cam_id)

                if not is_known:
                    uid       = ts
                    crop_path = os.path.join(cfg["unknowns_dir"],"plates",
                                             f"plate_{uid}.jpg")
                    cv2.imwrite(crop_path, crop)
                    state.add_unknown_plate(uid, plate_text, crop_path,
                                            datetime.now().strftime("%H:%M:%S"), cam_id)

                state.add_plate_event({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "plate": plate_text, "confidence": ocr_conf,
                    "known": is_known, "camera": cam["name"]
                })
                state.increment_perf("total_plates")

            state.update_combined(cam_id, annotated)
            state.update_perf("plate_inference", round((time.time()-t0)*1000,1))

        fps_counter += 1
        if time.time() - fps_timer >= 1.0:
            state.update_perf("plate_fps", fps_counter)
            fps_counter = 0; fps_timer = time.time()
        time.sleep(0.01)

    log.info("Number plate thread stopped.")


# ═══════════════════════════════════════════════════════════════════════════════
# FLASK APP
# ═══════════════════════════════════════════════════════════════════════════════

def make_flask_app(state, db, cfg):
    app = Flask(__name__)
    app.secret_key = cfg.get("flask_secret", "cis3425-key")

    # ── Auth decorator ────────────────────────────────────────────────────────
    def login_required(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if not session.get("logged_in"):
                return redirect(url_for("login"))
            return f(*args, **kwargs)
        return decorated

    # ── Login / Logout ────────────────────────────────────────────────────────
    @app.route("/login", methods=["GET","POST"])
    def login():
        error = ""
        if request.method == "POST":
            username = request.form.get("username","").strip()
            password = request.form.get("password","")
            users    = cfg.get("users", {})
            if username in users:
                stored = users[username].get("password_hash","")
                if stored == hash_password(password):
                    session["logged_in"] = True
                    session["username"]  = username
                    session["role"]      = users[username].get("role","viewer")
                    return redirect(url_for("dashboard"))
            error = "Invalid username or password"
        return render_template_string(LOGIN_HTML, error=error)

    @app.route("/logout")
    def logout():
        session.clear()
        return redirect(url_for("login"))

    # ── Main dashboard ────────────────────────────────────────────────────────
    @app.route("/")
    @login_required
    def dashboard():
        cams = [c for c in cfg["cameras"] if c.get("enabled", True)]
        return render_template_string(DASHBOARD_HTML,
                                      cameras=cams,
                                      username=session.get("username",""),
                                      role=session.get("role","viewer"))

    # ── Settings page ─────────────────────────────────────────────────────────
    @app.route("/settings", methods=["GET","POST"])
    @login_required
    def settings():
        if session.get("role") != "admin":
            return "Access denied — admin only", 403
        msg = ""
        if request.method == "POST":
            action = request.form.get("action")

            if action == "save_thresholds":
                cfg["face_confidence"]       = float(request.form.get("face_conf", 0.5))
                cfg["plate_confidence"]      = float(request.form.get("plate_conf", 0.45))
                cfg["recognition_tolerance"] = float(request.form.get("rec_tol", 0.5))
                cfg["unknown_cooldown"]      = int(request.form.get("unk_cool", 15))
                cfg["plate_cooldown"]        = int(request.form.get("plate_cool", 15))
                cfg["face_frame_skip"]       = int(request.form.get("face_skip", 2))
                cfg["plate_frame_skip"]      = int(request.form.get("plate_skip", 3))
                save_config(cfg)
                msg = "✅ Thresholds saved. Restart server to apply."

            elif action == "add_camera":
                new_cam = {
                    "id":      f"cam{len(cfg['cameras'])}",
                    "name":    request.form.get("cam_name", "Camera"),
                    "source":  request.form.get("cam_source", "0"),
                    "role":    request.form.get("cam_role", "face+plate"),
                    "enabled": True
                }
                # Convert source to int if it's a digit
                try:
                    new_cam["source"] = int(new_cam["source"])
                except ValueError:
                    pass
                cfg["cameras"].append(new_cam)
                save_config(cfg)
                msg = f"✅ Camera '{new_cam['name']}' added. Restart server to activate."

            elif action == "remove_camera":
                cam_id = request.form.get("cam_id")
                cfg["cameras"] = [c for c in cfg["cameras"] if c["id"] != cam_id]
                save_config(cfg)
                msg = "✅ Camera removed."

            elif action == "toggle_camera":
                cam_id = request.form.get("cam_id")
                for c in cfg["cameras"]:
                    if c["id"] == cam_id:
                        c["enabled"] = not c.get("enabled", True)
                save_config(cfg)
                msg = "✅ Camera toggled."

            elif action == "change_password":
                username    = session.get("username")
                old_pw      = request.form.get("old_password","")
                new_pw      = request.form.get("new_password","")
                confirm_pw  = request.form.get("confirm_password","")
                users       = cfg.get("users",{})
                if username in users and users[username]["password_hash"] == hash_password(old_pw):
                    if new_pw == confirm_pw and len(new_pw) >= 6:
                        cfg["users"][username]["password_hash"] = hash_password(new_pw)
                        save_config(cfg)
                        msg = "✅ Password changed."
                    else:
                        msg = "❌ Passwords don't match or too short (min 6 chars)."
                else:
                    msg = "❌ Current password incorrect."

            elif action == "add_user":
                new_username = request.form.get("new_username","").strip()
                new_password = request.form.get("new_password","")
                new_role     = request.form.get("new_role","viewer")
                if new_username and new_password:
                    cfg["users"][new_username] = {
                        "password_hash": hash_password(new_password),
                        "role": new_role
                    }
                    save_config(cfg)
                    msg = f"✅ User '{new_username}' added."
                else:
                    msg = "❌ Username and password required."

        return render_template_string(SETTINGS_HTML,
                                      cfg=cfg, msg=msg,
                                      username=session.get("username",""))

    # ── Live feed ─────────────────────────────────────────────────────────────
    @app.route("/feed/<cam_id>")
    def feed(cam_id):
        # Check session outside generator (context safe)
        if not session.get("logged_in"):
            return redirect(url_for("login"))

        def gen():
            while state.running:
                frame = state.get_feed_frame(cam_id)
                if frame is None:
                    time.sleep(0.05)
                    continue
                try:
                    _, buf = cv2.imencode(".jpg", frame,
                                         [cv2.IMWRITE_JPEG_QUALITY, 75])
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
                           buf.tobytes() + b"\r\n")
                except Exception as e:
                    log.debug(f"Feed encode error: {e}")
                time.sleep(0.04)

        return Response(gen(),
                        mimetype="multipart/x-mixed-replace; boundary=frame",
                        headers={"Cache-Control": "no-cache, no-store",
                                 "Pragma": "no-cache"})

    # ── API endpoints ─────────────────────────────────────────────────────────
    @app.route("/api/events")
    @login_required
    def api_events():
        return jsonify(state.get_recent_events())

    @app.route("/api/perf")
    @login_required
    def api_perf():
        return jsonify(state.get_perf())

    @app.route("/api/unknowns")
    @login_required
    def api_unknowns():
        return jsonify(state.get_unknowns())

    @app.route("/api/logs")
    @login_required
    def api_logs():
        return jsonify(db.get_recent_logs())

    @app.route("/recordings")
    @login_required
    def recordings():
        rec_dir = "recordings"
        files   = []
        if os.path.exists(rec_dir):
            for cam_dir in sorted(os.listdir(rec_dir)):
                cam_path = os.path.join(rec_dir, cam_dir)
                if os.path.isdir(cam_path):
                    for f in sorted(os.listdir(cam_path), reverse=True):
                        if f.endswith(".mp4"):
                            fp   = os.path.join(cam_path, f)
                            size = os.path.getsize(fp) / (1024*1024)
                            files.append({
                                "camera": cam_dir,
                                "filename": f,
                                "path": fp,
                                "size_mb": round(size, 1),
                                "url": f"/recordings/download/{cam_dir}/{f}"
                            })
        return render_template_string(RECORDINGS_HTML,
                                      files=files,
                                      username=session.get("username",""))

    @app.route("/recordings/download/<cam_id>/<filename>")
    @login_required
    def download_recording(cam_id, filename):
        filepath = os.path.join("recordings", cam_id, filename)
        if os.path.exists(filepath):
            return send_file(os.path.abspath(filepath),
                             as_attachment=True,
                             download_name=filename)
        return "File not found", 404

    @app.route("/unknown_img/face/<uid>")
    @login_required
    def unknown_face_img(uid):
        for f in state.get_unknowns()["faces"]:
            if f["id"] == uid:
                return send_file(os.path.abspath(f["path"]), mimetype="image/jpeg")
        return "", 404

    @app.route("/unknown_img/plate/<uid>")
    @login_required
    def unknown_plate_img(uid):
        for p in state.get_unknowns()["plates"]:
            if p["id"] == uid:
                return send_file(os.path.abspath(p["path"]), mimetype="image/jpeg")
        return "", 404

    @app.route("/enrol/face", methods=["POST"])
    @login_required
    def enrol_face():
        try:
            data = request.get_json()
            if not data:
                return jsonify({"success": False, "message": "No data received"})
            uid  = data.get("uid","").strip()
            name = data.get("name","").strip()
            if not uid or not name:
                return jsonify({"success": False, "message": "Missing name or uid"})
            crop_path = next((f["path"] for f in state.get_unknowns()["faces"]
                             if f["id"]==uid), None)
            if not crop_path:
                return jsonify({"success": False, "message": "Image not found — try again"})
            if not os.path.exists(crop_path):
                return jsonify({"success": False, "message": "Image file missing from disk"})

            # Run encoding in background thread to avoid blocking Flask
            result = {"ok": False, "msg": "Processing..."}
            done   = threading.Event()

            def do_enrol():
                ok, msg = enrol_face_from_crop(db, state, crop_path, name)
                result["ok"]  = ok
                result["msg"] = msg
                done.set()

            t = threading.Thread(target=do_enrol, daemon=True)
            t.start()
            done.wait(timeout=30)  # Wait up to 30s

            if result["ok"]:
                state.remove_unknown_face(uid)
            return jsonify({"success": result["ok"], "message": result["msg"]})

        except Exception as e:
            log.error(f"Enrol face error: {e}")
            return jsonify({"success": False, "message": f"Error: {str(e)}"})

    @app.route("/enrol/plate", methods=["POST"])
    @login_required
    def enrol_plate():
        data  = request.get_json()
        uid   = data.get("uid")
        plate = data.get("plate","").upper().replace(" ","").strip()
        if not uid or not plate:
            return jsonify({"success": False, "message": "Missing data"})
        db.add_known_plate(plate)
        with state.lock:
            state.known_plates.add(plate)
        state.remove_unknown_plate(uid)
        return jsonify({"success": True, "message": f"✅ {plate} added to whitelist"})

    return app


# ═══════════════════════════════════════════════════════════════════════════════
# HTML TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════════

LOGIN_HTML = """
<!DOCTYPE html>
<html>
<head>
<title>VertexGuard — Sign In</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:       #0b0f1a;
    --surface:  #111827;
    --border:   #1e2a3a;
    --accent:   #3b82f6;
    --accent2:  #06b6d4;
    --text:     #e2e8f0;
    --muted:    #64748b;
    --danger:   #ef4444;
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body {
    background:var(--bg);
    min-height:100vh;
    display:flex;
    align-items:center;
    justify-content:center;
    font-family:'DM Sans', sans-serif;
    position:relative;
    overflow:hidden;
  }
  body::before {
    content:'';
    position:absolute;
    width:600px; height:600px;
    background:radial-gradient(circle, rgba(59,130,246,0.08) 0%, transparent 70%);
    top:-100px; left:-100px;
    pointer-events:none;
  }
  body::after {
    content:'';
    position:absolute;
    width:400px; height:400px;
    background:radial-gradient(circle, rgba(6,182,212,0.06) 0%, transparent 70%);
    bottom:-50px; right:-50px;
    pointer-events:none;
  }
  .card {
    background:var(--surface);
    border:1px solid var(--border);
    border-radius:16px;
    padding:48px 40px;
    width:380px;
    position:relative;
    z-index:1;
    box-shadow:0 25px 50px rgba(0,0,0,0.4);
  }
  .logo {
    display:flex;
    align-items:center;
    gap:10px;
    margin-bottom:32px;
  }
  .logo-icon {
    width:36px; height:36px;
    background:linear-gradient(135deg, var(--accent), var(--accent2));
    border-radius:8px;
    display:flex; align-items:center; justify-content:center;
  }
  .logo-icon svg { width:18px; height:18px; fill:white; }
  .logo-text { font-size:1.1rem; font-weight:600; color:var(--text); letter-spacing:-0.02em; }
  h1 { font-size:1.5rem; font-weight:600; color:var(--text); margin-bottom:6px; letter-spacing:-0.03em; }
  p  { font-size:0.85rem; color:var(--muted); margin-bottom:32px; }
  label {
    display:block;
    font-size:0.75rem;
    font-weight:500;
    color:var(--muted);
    text-transform:uppercase;
    letter-spacing:0.08em;
    margin-bottom:6px;
  }
  input {
    width:100%;
    background:#0f172a;
    border:1px solid var(--border);
    color:var(--text);
    padding:11px 14px;
    border-radius:8px;
    font-family:'DM Sans', sans-serif;
    font-size:0.9rem;
    margin-bottom:20px;
    transition:border-color 0.2s;
    outline:none;
  }
  input:focus { border-color:var(--accent); }
  button {
    width:100%;
    background:linear-gradient(135deg, var(--accent), var(--accent2));
    border:none;
    color:#fff;
    padding:12px;
    border-radius:8px;
    font-family:'DM Sans', sans-serif;
    font-size:0.9rem;
    font-weight:500;
    cursor:pointer;
    transition:opacity 0.2s, transform 0.1s;
    letter-spacing:0.01em;
  }
  button:hover { opacity:0.9; }
  button:active { transform:scale(0.99); }
  .error {
    background:rgba(239,68,68,0.08);
    border:1px solid rgba(239,68,68,0.3);
    color:#fca5a5;
    padding:10px 14px;
    border-radius:8px;
    font-size:0.82rem;
    margin-bottom:20px;
  }
</style>
</head>
<body>
<div class="card">
  <div class="logo">
    <div class="logo-icon">
      <svg viewBox="0 0 24 24"><path d="M12 1L3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5l-9-4z"/></svg>
    </div>
    <span class="logo-text">VertexGuard</span>
  </div>
  <h1>Welcome back</h1>
  <p>Sign in to your security dashboard</p>
  {% if error %}<div class="error">{{ error }}</div>{% endif %}
  <form method="POST">
    <label>Username</label>
    <input type="text" name="username" autofocus autocomplete="username">
    <label>Password</label>
    <input type="password" name="password" autocomplete="current-password">
    <button type="submit">Sign In</button>
  </form>
</div>
</body></html>
"""

SETTINGS_HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Settings — VertexGuard</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:#0b0f1a; --surface:#111827; --surface2:#0f172a;
    --border:#1e2a3a; --border2:#243040;
    --accent:#3b82f6; --accent2:#06b6d4;
    --text:#e2e8f0; --muted:#64748b; --muted2:#94a3b8;
    --success:#10b981; --danger:#ef4444; --warning:#f59e0b;
    --sidebar:220px;
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:var(--bg); color:var(--text); font-family:'DM Sans',sans-serif; display:flex; min-height:100vh; }

  /* Sidebar */
  .sidebar {
    width:var(--sidebar); background:var(--surface); border-right:1px solid var(--border);
    display:flex; flex-direction:column; padding:24px 0; position:fixed; top:0; left:0; height:100vh;
  }
  .sidebar-logo { padding:0 20px 24px; border-bottom:1px solid var(--border); margin-bottom:16px; }
  .logo-row { display:flex; align-items:center; gap:10px; }
  .logo-icon { width:32px; height:32px; background:linear-gradient(135deg,var(--accent),var(--accent2)); border-radius:7px; display:flex; align-items:center; justify-content:center; }
  .logo-icon svg { width:16px; height:16px; fill:white; }
  .logo-text { font-size:1rem; font-weight:600; letter-spacing:-0.02em; }
  .nav-item { display:flex; align-items:center; gap:10px; padding:9px 20px; font-size:0.85rem; color:var(--muted2); text-decoration:none; transition:all 0.15s; cursor:pointer; border-left:2px solid transparent; }
  .nav-item:hover { color:var(--text); background:rgba(255,255,255,0.04); }
  .nav-item.active { color:var(--accent); border-left-color:var(--accent); background:rgba(59,130,246,0.08); }
  .nav-icon { width:16px; height:16px; opacity:0.7; }
  .nav-section { padding:16px 20px 6px; font-size:0.65rem; text-transform:uppercase; letter-spacing:0.1em; color:var(--muted); }
  .sidebar-bottom { margin-top:auto; padding:16px 20px; border-top:1px solid var(--border); }
  .user-row { display:flex; align-items:center; gap:10px; }
  .avatar { width:30px; height:30px; background:linear-gradient(135deg,var(--accent),var(--accent2)); border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:0.75rem; font-weight:600; }
  .user-name { font-size:0.82rem; color:var(--muted2); }

  /* Main */
  .main { margin-left:var(--sidebar); flex:1; padding:32px; max-width:calc(100% - var(--sidebar)); }
  .page-title { font-size:1.4rem; font-weight:600; letter-spacing:-0.03em; margin-bottom:4px; }
  .page-sub { font-size:0.83rem; color:var(--muted); margin-bottom:28px; }

  /* Cards */
  .card { background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:24px; margin-bottom:20px; }
  .card-title { font-size:0.78rem; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; color:var(--muted2); margin-bottom:20px; }

  /* Form */
  .form-grid { display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-bottom:16px; }
  label { display:block; font-size:0.72rem; font-weight:500; text-transform:uppercase; letter-spacing:0.07em; color:var(--muted); margin-bottom:5px; }
  input, select { width:100%; background:var(--surface2); border:1px solid var(--border2); color:var(--text); padding:9px 12px; border-radius:7px; font-family:'DM Sans',sans-serif; font-size:0.85rem; outline:none; transition:border-color 0.15s; }
  input:focus, select:focus { border-color:var(--accent); }
  select option { background:var(--surface2); }

  /* Buttons */
  .btn { border:none; padding:9px 18px; border-radius:7px; font-family:'DM Sans',sans-serif; font-size:0.82rem; font-weight:500; cursor:pointer; transition:opacity 0.15s; }
  .btn-primary { background:linear-gradient(135deg,var(--accent),var(--accent2)); color:#fff; }
  .btn-danger  { background:rgba(239,68,68,0.15); color:#fca5a5; border:1px solid rgba(239,68,68,0.3); }
  .btn-secondary { background:rgba(99,102,241,0.15); color:#a5b4fc; border:1px solid rgba(99,102,241,0.3); }
  .btn:hover { opacity:0.85; }

  /* Table */
  table { width:100%; border-collapse:collapse; font-size:0.82rem; }
  th { text-align:left; color:var(--muted); font-size:0.7rem; text-transform:uppercase; letter-spacing:0.07em; padding:8px 10px; border-bottom:1px solid var(--border); font-weight:500; }
  td { padding:10px; border-bottom:1px solid rgba(30,42,58,0.5); color:var(--muted2); }
  tr:last-child td { border-bottom:none; }

  /* Badge */
  .badge { padding:3px 10px; border-radius:20px; font-size:0.7rem; font-weight:500; }
  .badge-on  { background:rgba(16,185,129,0.12); color:#6ee7b7; border:1px solid rgba(16,185,129,0.25); }
  .badge-off { background:rgba(239,68,68,0.1); color:#fca5a5; border:1px solid rgba(239,68,68,0.2); }
  .badge-role { background:rgba(59,130,246,0.1); color:#93c5fd; border:1px solid rgba(59,130,246,0.2); }

  /* Alert */
  .alert { padding:12px 16px; border-radius:8px; font-size:0.83rem; margin-bottom:20px; }
  .alert-ok  { background:rgba(16,185,129,0.08); border:1px solid rgba(16,185,129,0.25); color:#6ee7b7; }
  .alert-err { background:rgba(239,68,68,0.08); border:1px solid rgba(239,68,68,0.25); color:#fca5a5; }
  .divider { height:1px; background:var(--border); margin:20px 0; }
</style>
</head>
<body>
<aside class="sidebar">
  <div class="sidebar-logo">
    <div class="logo-row">
      <div class="logo-icon"><svg viewBox="0 0 24 24"><path d="M12 1L3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5l-9-4z"/></svg></div>
      <span class="logo-text">VertexGuard</span>
    </div>
  </div>
  <span class="nav-section">Navigation</span>
  <a href="/" class="nav-item">
    <svg class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/></svg>
    Dashboard
  </a>
  <a href="/recordings" class="nav-item">
    <svg class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polygon points="10,8 16,12 10,16"/></svg>
    Recordings
  </a>
  <span class="nav-section">System</span>
  <a href="/settings" class="nav-item active">
    <svg class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14M4.93 4.93a10 10 0 0 0 0 14.14"/></svg>
    Settings
  </a>
  <div class="sidebar-bottom">
    <div class="user-row">
      <div class="avatar">{{ username[0].upper() }}</div>
      <div>
        <div style="font-size:0.82rem;font-weight:500">{{ username }}</div>
        <a href="/logout" style="font-size:0.72rem;color:var(--muted);text-decoration:none">Sign out</a>
      </div>
    </div>
  </div>
</aside>

<main class="main">
  <div class="page-title">Settings</div>
  <div class="page-sub">Manage cameras, detection thresholds and user accounts</div>

  {% if msg %}
  <div class="alert {{ 'alert-ok' if msg.startswith('') else 'alert-err' }}">{{ msg }}</div>
  {% endif %}

  <!-- Camera Management -->
  <div class="card">
    <div class="card-title">Camera Management</div>
    <table>
      <tr><th>Name</th><th>Source</th><th>Role</th><th>Status</th><th>Actions</th></tr>
      {% for cam in cfg.cameras %}
      <tr>
        <td style="color:var(--text);font-weight:500">{{ cam.name }}</td>
        <td><code style="font-family:'DM Mono',monospace;font-size:0.78rem;color:var(--accent2)">{{ cam.source }}</code></td>
        <td>{{ cam.role }}</td>
        <td><span class="badge {{ 'badge-on' if cam.get('enabled', True) else 'badge-off' }}">{{ 'Active' if cam.get('enabled', True) else 'Disabled' }}</span></td>
        <td style="display:flex;gap:8px">
          <form method="POST" style="display:inline">
            <input type="hidden" name="action" value="toggle_camera">
            <input type="hidden" name="cam_id" value="{{ cam.id }}">
            <button class="btn btn-secondary" style="padding:5px 12px;font-size:0.75rem">Toggle</button>
          </form>
          {% if cam.id != 'cam0' %}
          <form method="POST" style="display:inline">
            <input type="hidden" name="action" value="remove_camera">
            <input type="hidden" name="cam_id" value="{{ cam.id }}">
            <button class="btn btn-danger" style="padding:5px 12px;font-size:0.75rem">Remove</button>
          </form>
          {% endif %}
        </td>
      </tr>
      {% endfor %}
    </table>
    <div class="divider"></div>
    <form method="POST">
      <input type="hidden" name="action" value="add_camera">
      <div class="form-grid">
        <div><label>Camera Name</label><input type="text" name="cam_name" placeholder="e.g. Back Door"></div>
        <div><label>Source (0, 1 or RTSP URL)</label><input type="text" name="cam_source" placeholder="e.g. 1 or rtsp://..."></div>
      </div>
      <div class="form-grid" style="grid-template-columns:1fr auto">
        <div><label>Role</label>
          <select name="cam_role">
            <option value="face+plate">Face + Plate Detection</option>
            <option value="face">Face Detection Only</option>
            <option value="plate">Plate Detection Only</option>
          </select>
        </div>
        <div style="display:flex;align-items:flex-end"><button class="btn btn-primary" type="submit">Add Camera</button></div>
      </div>
    </form>
  </div>

  <!-- Detection Thresholds -->
  <div class="card">
    <div class="card-title">Detection Thresholds</div>
    <form method="POST">
      <input type="hidden" name="action" value="save_thresholds">
      <div class="form-grid">
        <div><label>Face Confidence</label><input type="number" name="face_conf" step="0.05" min="0.1" max="1.0" value="{{ cfg.face_confidence }}"></div>
        <div><label>Plate Confidence</label><input type="number" name="plate_conf" step="0.05" min="0.1" max="1.0" value="{{ cfg.plate_confidence }}"></div>
      </div>
      <div class="form-grid">
        <div><label>Recognition Tolerance (lower = stricter)</label><input type="number" name="rec_tol" step="0.05" min="0.1" max="1.0" value="{{ cfg.recognition_tolerance }}"></div>
        <div><label>Unknown Face Cooldown (seconds)</label><input type="number" name="unk_cool" min="1" max="300" value="{{ cfg.unknown_cooldown }}"></div>
      </div>
      <div class="form-grid">
        <div><label>Face Frame Skip</label><input type="number" name="face_skip" min="1" max="10" value="{{ cfg.face_frame_skip }}"></div>
        <div><label>Plate Frame Skip</label><input type="number" name="plate_skip" min="1" max="10" value="{{ cfg.plate_frame_skip }}"></div>
      </div>
      <button class="btn btn-primary" type="submit">Save Thresholds</button>
    </form>
  </div>

  <!-- User Management -->
  <div class="card">
    <div class="card-title">User Accounts</div>
    <table>
      <tr><th>Username</th><th>Role</th></tr>
      {% for uname, udata in cfg.users.items() %}
      <tr>
        <td style="color:var(--text);font-weight:500">{{ uname }}</td>
        <td><span class="badge badge-role">{{ udata.role }}</span></td>
      </tr>
      {% endfor %}
    </table>
    <div class="divider"></div>
    <form method="POST">
      <input type="hidden" name="action" value="add_user">
      <div class="form-grid">
        <div><label>Username</label><input type="text" name="new_username" placeholder="username"></div>
        <div><label>Password</label><input type="password" name="new_password" placeholder="password"></div>
      </div>
      <div class="form-grid" style="grid-template-columns:1fr auto">
        <div><label>Role</label>
          <select name="new_role">
            <option value="viewer">Viewer</option>
            <option value="admin">Admin</option>
          </select>
        </div>
        <div style="display:flex;align-items:flex-end"><button class="btn btn-primary" type="submit">Add User</button></div>
      </div>
    </form>
    <div class="divider"></div>
    <form method="POST">
      <input type="hidden" name="action" value="change_password">
      <div class="form-grid">
        <div><label>Current Password</label><input type="password" name="old_password"></div>
        <div><label>New Password</label><input type="password" name="new_password"></div>
      </div>
      <div class="form-grid" style="grid-template-columns:1fr auto">
        <div><label>Confirm New Password</label><input type="password" name="confirm_password"></div>
        <div style="display:flex;align-items:flex-end"><button class="btn btn-secondary" type="submit">Change Password</button></div>
      </div>
    </form>
  </div>
</main>
</body></html>
"""

RECORDINGS_HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Recordings — VertexGuard</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:#0b0f1a; --surface:#111827; --surface2:#0f172a;
    --border:#1e2a3a; --border2:#243040;
    --accent:#3b82f6; --accent2:#06b6d4;
    --text:#e2e8f0; --muted:#64748b; --muted2:#94a3b8;
    --success:#10b981; --danger:#ef4444;
    --sidebar:220px;
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:var(--bg); color:var(--text); font-family:'DM Sans',sans-serif; display:flex; min-height:100vh; }
  .sidebar { width:var(--sidebar); background:var(--surface); border-right:1px solid var(--border); display:flex; flex-direction:column; padding:24px 0; position:fixed; top:0; left:0; height:100vh; }
  .sidebar-logo { padding:0 20px 24px; border-bottom:1px solid var(--border); margin-bottom:16px; }
  .logo-row { display:flex; align-items:center; gap:10px; }
  .logo-icon { width:32px; height:32px; background:linear-gradient(135deg,var(--accent),var(--accent2)); border-radius:7px; display:flex; align-items:center; justify-content:center; }
  .logo-icon svg { width:16px; height:16px; fill:white; }
  .logo-text { font-size:1rem; font-weight:600; letter-spacing:-0.02em; }
  .nav-item { display:flex; align-items:center; gap:10px; padding:9px 20px; font-size:0.85rem; color:var(--muted2); text-decoration:none; transition:all 0.15s; border-left:2px solid transparent; }
  .nav-item:hover { color:var(--text); background:rgba(255,255,255,0.04); }
  .nav-item.active { color:var(--accent); border-left-color:var(--accent); background:rgba(59,130,246,0.08); }
  .nav-icon { width:16px; height:16px; }
  .nav-section { padding:16px 20px 6px; font-size:0.65rem; text-transform:uppercase; letter-spacing:0.1em; color:var(--muted); }
  .sidebar-bottom { margin-top:auto; padding:16px 20px; border-top:1px solid var(--border); }
  .avatar { width:30px; height:30px; background:linear-gradient(135deg,var(--accent),var(--accent2)); border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:0.75rem; font-weight:600; }
  .main { margin-left:var(--sidebar); flex:1; padding:32px; }
  .page-title { font-size:1.4rem; font-weight:600; letter-spacing:-0.03em; margin-bottom:4px; }
  .page-sub { font-size:0.83rem; color:var(--muted); margin-bottom:28px; }
  .card { background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:24px; }
  .card-title { font-size:0.78rem; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; color:var(--muted2); margin-bottom:20px; }
  table { width:100%; border-collapse:collapse; font-size:0.82rem; }
  th { text-align:left; color:var(--muted); font-size:0.7rem; text-transform:uppercase; letter-spacing:0.07em; padding:8px 10px; border-bottom:1px solid var(--border); font-weight:500; }
  td { padding:10px; border-bottom:1px solid rgba(30,42,58,0.5); color:var(--muted2); }
  tr:last-child td { border-bottom:none; }
  .badge { padding:3px 10px; border-radius:20px; font-size:0.7rem; font-weight:500; background:rgba(59,130,246,0.1); color:#93c5fd; border:1px solid rgba(59,130,246,0.2); }
  .dl-btn { background:linear-gradient(135deg,var(--accent),var(--accent2)); border:none; color:#fff; padding:5px 14px; border-radius:6px; font-size:0.78rem; font-weight:500; text-decoration:none; display:inline-block; cursor:pointer; }
  .dl-btn:hover { opacity:0.85; }
  .empty { color:var(--muted); font-size:0.85rem; padding:32px; text-align:center; }
</style>
</head>
<body>
<aside class="sidebar">
  <div class="sidebar-logo">
    <div class="logo-row">
      <div class="logo-icon"><svg viewBox="0 0 24 24"><path d="M12 1L3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5l-9-4z"/></svg></div>
      <span class="logo-text">VertexGuard</span>
    </div>
  </div>
  <span class="nav-section">Navigation</span>
  <a href="/" class="nav-item">
    <svg class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/></svg>
    Dashboard
  </a>
  <a href="/recordings" class="nav-item active">
    <svg class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polygon points="10,8 16,12 10,16"/></svg>
    Recordings
  </a>
  <span class="nav-section">System</span>
  <a href="/settings" class="nav-item">
    <svg class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14M4.93 4.93a10 10 0 0 0 0 14.14"/></svg>
    Settings
  </a>
  <div class="sidebar-bottom">
    <div style="display:flex;align-items:center;gap:10px">
      <div class="avatar">{{ username[0].upper() }}</div>
      <div>
        <div style="font-size:0.82rem;font-weight:500">{{ username }}</div>
        <a href="/logout" style="font-size:0.72rem;color:var(--muted);text-decoration:none">Sign out</a>
      </div>
    </div>
  </div>
</aside>
<main class="main">
  <div class="page-title">Recordings</div>
  <div class="page-sub">Browse and download saved camera footage</div>
  <div class="card">
    <div class="card-title">Saved Recordings</div>
    {% if files %}
    <table>
      <tr><th>Camera</th><th>Filename</th><th>Size</th><th></th></tr>
      {% for f in files %}
      <tr>
        <td><span class="badge">{{ f.camera }}</span></td>
        <td style="font-family:'DM Mono',monospace;font-size:0.78rem;color:var(--text)">{{ f.filename }}</td>
        <td>{{ f.size_mb }} MB</td>
        <td><a class="dl-btn" href="{{ f.url }}">Download</a></td>
      </tr>
      {% endfor %}
    </table>
    {% else %}
    <div class="empty">No recordings yet — files appear here after 10 minutes of runtime.</div>
    {% endif %}
  </div>
</main>
</body></html>
"""


DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
<title>VertexGuard — Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:#0b0f1a; --surface:#111827; --surface2:#0f172a;
    --border:#1e2a3a; --border2:#243040;
    --accent:#3b82f6; --accent2:#06b6d4;
    --text:#e2e8f0; --muted:#64748b; --muted2:#94a3b8;
    --success:#10b981; --danger:#ef4444; --warning:#f59e0b;
    --sidebar:220px;
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:var(--bg); color:var(--text); font-family:'DM Sans',sans-serif; display:flex; min-height:100vh; overflow-x:hidden; }
  .sidebar { width:var(--sidebar); background:var(--surface); border-right:1px solid var(--border); display:flex; flex-direction:column; padding:24px 0; position:fixed; top:0; left:0; height:100vh; z-index:100; }
  .sidebar-logo { padding:0 20px 24px; border-bottom:1px solid var(--border); margin-bottom:16px; }
  .logo-row { display:flex; align-items:center; gap:10px; }
  .logo-icon { width:32px; height:32px; background:linear-gradient(135deg,var(--accent),var(--accent2)); border-radius:7px; display:flex; align-items:center; justify-content:center; flex-shrink:0; }
  .logo-icon svg { width:16px; height:16px; fill:white; }
  .logo-text { font-size:1rem; font-weight:600; letter-spacing:-0.02em; }
  .nav-item { display:flex; align-items:center; gap:10px; padding:9px 20px; font-size:0.85rem; color:var(--muted2); text-decoration:none; transition:all 0.15s; border-left:2px solid transparent; }
  .nav-item:hover { color:var(--text); background:rgba(255,255,255,0.04); }
  .nav-item.active { color:var(--accent); border-left-color:var(--accent); background:rgba(59,130,246,0.08); }
  .nav-icon { width:16px; height:16px; flex-shrink:0; }
  .nav-section { padding:16px 20px 6px; font-size:0.65rem; text-transform:uppercase; letter-spacing:0.1em; color:var(--muted); }
  .sidebar-bottom { margin-top:auto; padding:16px 20px; border-top:1px solid var(--border); }
  .avatar { width:30px; height:30px; background:linear-gradient(135deg,var(--accent),var(--accent2)); border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:0.75rem; font-weight:600; flex-shrink:0; }
  .main { margin-left:var(--sidebar); flex:1; display:flex; flex-direction:column; min-height:100vh; }
  .topbar { background:var(--surface); border-bottom:1px solid var(--border); padding:14px 24px; display:flex; align-items:center; justify-content:space-between; position:sticky; top:0; z-index:50; }
  .page-title { font-size:1rem; font-weight:600; letter-spacing:-0.02em; }
  .topbar-right { display:flex; align-items:center; gap:12px; }
  .clock { font-size:0.78rem; color:var(--muted); font-family:'DM Mono',monospace; }
  .dev-btn { background:var(--surface2); border:1px solid var(--border2); color:var(--muted2); padding:5px 14px; border-radius:6px; font-family:'DM Sans',sans-serif; font-size:0.78rem; font-weight:500; cursor:pointer; transition:all 0.15s; }
  .dev-btn.active { background:rgba(99,102,241,0.15); border-color:rgba(99,102,241,0.4); color:#a5b4fc; }
  .content { padding:20px 24px; flex:1; }
  .card { background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:20px; }
  .card-title { font-size:0.72rem; font-weight:600; text-transform:uppercase; letter-spacing:0.09em; color:var(--muted); margin-bottom:14px; }
  .grid-main { display:grid; grid-template-columns:1.6fr 1fr; gap:16px; margin-bottom:16px; }
  .grid-logs  { display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-bottom:16px; }
  .grid-full  { margin-bottom:16px; }
  .feed-wrap { position:relative; border-radius:8px; overflow:hidden; background:#000; }
  .feed-wrap img { width:100%; display:block; min-height:180px; }
  .feed-label { position:absolute; top:8px; left:8px; background:rgba(0,0,0,0.65); backdrop-filter:blur(4px); color:#fff; font-size:0.7rem; font-weight:500; padding:3px 8px; border-radius:4px; }
  .feed-rec { position:absolute; top:8px; right:8px; display:flex; align-items:center; gap:5px; background:rgba(239,68,68,0.85); color:#fff; font-size:0.68rem; font-weight:600; padding:3px 8px; border-radius:4px; }
  .rec-dot { width:6px; height:6px; border-radius:50%; background:#fff; animation:pulse 1.2s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }
  .feeds-grid { display:grid; gap:10px; }
  .status-item { display:flex; align-items:center; justify-content:space-between; padding:8px 0; border-bottom:1px solid rgba(30,42,58,0.5); font-size:0.82rem; }
  .status-item:last-child { border-bottom:none; }
  .status-label { color:var(--muted2); }
  .status-dot { width:7px; height:7px; border-radius:50%; margin-right:6px; display:inline-block; }
  .dot-green { background:var(--success); box-shadow:0 0 6px rgba(16,185,129,0.5); }
  .uptime-val { font-family:'DM Mono',monospace; font-size:0.78rem; color:var(--accent2); }
  .event { padding:7px 10px; margin:3px 0; border-radius:6px; font-size:0.78rem; display:flex; align-items:center; gap:6px; }
  .ev-known   { background:rgba(16,185,129,0.07); border-left:2px solid var(--success); }
  .ev-unknown { background:rgba(239,68,68,0.07); border-left:2px solid var(--danger); }
  .ev-time { color:var(--muted); font-family:'DM Mono',monospace; font-size:0.72rem; flex-shrink:0; }
  .ev-name { color:var(--text); font-weight:500; flex:1; }
  .ev-conf { color:var(--muted); font-size:0.72rem; }
  .ev-cam  { background:var(--surface2); color:var(--muted); font-size:0.65rem; padding:1px 6px; border-radius:3px; }
  .empty   { color:var(--muted); font-size:0.8rem; padding:12px 0; }
  .tabs { display:flex; gap:0; margin-bottom:14px; border-bottom:1px solid var(--border); }
  .tab { padding:8px 16px; font-size:0.8rem; font-weight:500; color:var(--muted2); cursor:pointer; border-bottom:2px solid transparent; margin-bottom:-1px; transition:all 0.15s; }
  .tab:hover { color:var(--text); }
  .tab.active { color:var(--accent); border-bottom-color:var(--accent); }
  .enrol-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(155px,1fr)); gap:12px; margin-top:4px; }
  .enrol-card { background:var(--surface2); border:1px solid var(--border2); border-radius:8px; padding:10px; text-align:center; transition:border-color 0.15s; }
  .enrol-card:hover { border-color:var(--accent); }
  .enrol-card img { width:100%; border-radius:5px; margin-bottom:8px; max-height:110px; object-fit:cover; }
  .enrol-card input { width:100%; background:var(--surface); border:1px solid var(--border2); color:var(--text); padding:6px 8px; border-radius:5px; font-family:'DM Sans',sans-serif; font-size:0.78rem; margin-bottom:6px; outline:none; }
  .enrol-card input:focus { border-color:var(--accent); }
  .enrol-btn { background:linear-gradient(135deg,var(--success),#059669); border:none; color:#fff; padding:5px 10px; border-radius:5px; font-size:0.75rem; font-weight:500; width:100%; cursor:pointer; }
  .plate-btn { background:linear-gradient(135deg,var(--accent),var(--accent2)); border:none; color:#fff; padding:5px 10px; border-radius:5px; font-size:0.75rem; font-weight:500; width:100%; cursor:pointer; margin-top:4px; }
  #dev-panel { display:none; margin-bottom:16px; }
  #dev-panel.visible { display:block; }
  .metrics-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(140px,1fr)); gap:10px; }
  .metric-card { background:var(--surface2); border:1px solid var(--border2); border-radius:8px; padding:14px 12px; }
  .metric-val { font-size:1.4rem; font-weight:600; color:var(--accent); font-family:'DM Mono',monospace; }
  .metric-val.green  { color:var(--success); }
  .metric-val.orange { color:var(--warning); }
  .metric-val.red    { color:var(--danger); }
  .metric-label { font-size:0.68rem; color:var(--muted); margin-top:4px; text-transform:uppercase; letter-spacing:0.07em; }
  .perf-bar-wrap { margin-top:8px; background:var(--border); border-radius:2px; height:3px; }
  .perf-bar { height:3px; border-radius:2px; background:linear-gradient(90deg,var(--accent),var(--accent2)); transition:width 0.5s; }
  #msg { position:fixed; top:20px; right:20px; padding:12px 18px; border-radius:8px; font-size:0.82rem; display:none; z-index:999; box-shadow:0 8px 24px rgba(0,0,0,0.4); }
  .msg-ok  { background:rgba(16,185,129,0.15); border:1px solid rgba(16,185,129,0.3); color:#6ee7b7; }
  .msg-err { background:rgba(239,68,68,0.15); border:1px solid rgba(239,68,68,0.3); color:#fca5a5; }
</style>
</head>
<body>
<div id="msg"></div>
<aside class="sidebar">
  <div class="sidebar-logo">
    <div class="logo-row">
      <div class="logo-icon"><svg viewBox="0 0 24 24"><path d="M12 1L3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5l-9-4z"/></svg></div>
      <span class="logo-text">VertexGuard</span>
    </div>
  </div>
  <span class="nav-section">Overview</span>
  <a href="/" class="nav-item active">
    <svg class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>
    Dashboard
  </a>
  <a href="/recordings" class="nav-item">
    <svg class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M15 10l4.553-2.069A1 1 0 0121 8.87v6.26a1 1 0 01-1.447.9L15 14M3 8a2 2 0 012-2h8a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2V8z"/></svg>
    Recordings
  </a>
  <span class="nav-section">System</span>
  {% if role == 'admin' %}
  <a href="/settings" class="nav-item">
    <svg class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-4 0v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 010-4h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 012.83-2.83l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 014 0v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 2.83l-.06.06A1.65 1.65 0 0019.4 9a1.65 1.65 0 001.51 1H21a2 2 0 010 4h-.09a1.65 1.65 0 00-1.51 1z"/></svg>
    Settings
  </a>
  {% endif %}
  <div class="sidebar-bottom">
    <div style="display:flex;align-items:center;gap:10px">
      <div class="avatar">{{ username[0].upper() if username else 'U' }}</div>
      <div>
        <div style="font-size:0.82rem;font-weight:500;color:var(--text)">{{ username }}</div>
        <a href="/logout" style="font-size:0.72rem;color:var(--muted);text-decoration:none">Sign out</a>
      </div>
    </div>
  </div>
</aside>
<div class="main">
  <div class="topbar">
    <div class="page-title">Security Dashboard</div>
    <div class="topbar-right">
      <span class="clock" id="clock">--:--:--</span>
      <button class="dev-btn" id="dev-toggle" onclick="toggleDev()">Dev Panel</button>
    </div>
  </div>
  <div class="content">
    <div id="dev-panel" class="card grid-full">
      <div class="card-title">Performance Monitor</div>
      <div class="metrics-grid">
        <div class="metric-card"><div class="metric-val green" id="m-face-fps">--</div><div class="metric-label">Face FPS</div><div class="perf-bar-wrap"><div class="perf-bar" id="b-face-fps" style="width:0%"></div></div></div>
        <div class="metric-card"><div class="metric-val green" id="m-plate-fps">--</div><div class="metric-label">Plate FPS</div><div class="perf-bar-wrap"><div class="perf-bar" id="b-plate-fps" style="width:0%"></div></div></div>
        <div class="metric-card"><div class="metric-val orange" id="m-face-inf">--</div><div class="metric-label">Face Inference ms</div><div class="perf-bar-wrap"><div class="perf-bar" id="b-face-inf" style="width:0%;background:var(--warning)"></div></div></div>
        <div class="metric-card"><div class="metric-val orange" id="m-plate-inf">--</div><div class="metric-label">Plate Inference ms</div><div class="perf-bar-wrap"><div class="perf-bar" id="b-plate-inf" style="width:0%;background:var(--warning)"></div></div></div>
        <div class="metric-card"><div class="metric-val" id="m-total-faces">--</div><div class="metric-label">Face Events</div></div>
        <div class="metric-card"><div class="metric-val" id="m-total-plates">--</div><div class="metric-label">Plate Events</div></div>
        <div class="metric-card"><div class="metric-val green" id="m-cameras">--</div><div class="metric-label">Active Cameras</div></div>
        <div class="metric-card"><div class="metric-val" id="m-uptime-dev">--</div><div class="metric-label">Uptime</div></div>
      </div>
    </div>
    <div class="grid-main">
      <div class="card">
        <div class="card-title">Live Feeds</div>
        <div class="feeds-grid">
          {% for cam in cameras %}
          <div class="feed-wrap">
            <img src="/feed/{{ cam.id }}" alt="{{ cam.name }}">
            <div class="feed-label">{{ cam.name }}</div>
            <div class="feed-rec"><div class="rec-dot"></div>REC</div>
          </div>
          {% endfor %}
        </div>
      </div>
      <div class="card">
        <div class="card-title">System Status</div>
        <div class="status-item"><span class="status-label">Face Recognition</span><span><span class="status-dot dot-green"></span><span style="font-size:0.78rem;color:var(--success)">Active</span></span></div>
        <div class="status-item"><span class="status-label">Plate Detection</span><span><span class="status-dot dot-green"></span><span style="font-size:0.78rem;color:var(--success)">Active</span></span></div>
        <div class="status-item"><span class="status-label">Database</span><span><span class="status-dot dot-green"></span><span style="font-size:0.78rem;color:var(--success)">Logging</span></span></div>
        <div class="status-item"><span class="status-label">Cameras</span><span style="font-size:0.78rem;color:var(--accent2)">{{ cameras|length }} online</span></div>
        <div class="status-item"><span class="status-label">Uptime</span><span class="uptime-val" id="uptime-main">--</span></div>
      </div>
    </div>
    <div class="grid-logs">
      <div class="card">
        <div class="card-title">Face Detections</div>
        <div id="face-log"><div class="empty">Waiting for detections...</div></div>
      </div>
      <div class="card">
        <div class="card-title">Plate Detections</div>
        <div id="plate-log"><div class="empty">Waiting for detections...</div></div>
      </div>
    </div>
    <div class="card grid-full">
      <div class="tabs">
        <div class="tab active" onclick="showTab('faces')">Unknown Faces</div>
        <div class="tab" onclick="showTab('plates')">Unknown Plates</div>
      </div>
      <div id="tab-faces"><div class="enrol-grid" id="unknown-faces"><div class="empty">No unknown faces detected yet.</div></div></div>
      <div id="tab-plates" style="display:none"><div class="enrol-grid" id="unknown-plates"><div class="empty">No unknown plates detected yet.</div></div></div>
    </div>
  </div>
</div>
<script>
let devOpen=false;
function toggleDev(){devOpen=!devOpen;document.getElementById('dev-panel').classList.toggle('visible',devOpen);document.getElementById('dev-toggle').classList.toggle('active',devOpen);}
function showTab(tab){document.getElementById('tab-faces').style.display=tab==='faces'?'':'none';document.getElementById('tab-plates').style.display=tab==='plates'?'':'none';document.querySelectorAll('.tab').forEach((t,i)=>{t.classList.toggle('active',(tab==='faces'&&i===0)||(tab==='plates'&&i===1));});}
function showMsg(text,ok=true){const m=document.getElementById('msg');m.textContent=text;m.className=ok?'msg-ok':'msg-err';m.style.display='block';setTimeout(()=>m.style.display='none',3000);}
function enrolFace(uid){const name=document.getElementById('name-'+uid).value.trim();if(!name){showMsg('Enter a name first',false);return;}fetch('/enrol/face',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({uid,name})}).then(r=>r.json()).then(d=>{showMsg(d.message,d.success);if(d.success)document.getElementById('card-'+uid).remove();});}
function enrolPlate(uid){const plate=document.getElementById('plate-'+uid).value.trim();if(!plate){showMsg('Enter plate text first',false);return;}fetch('/enrol/plate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({uid,plate})}).then(r=>r.json()).then(d=>{showMsg(d.message,d.success);if(d.success)document.getElementById('card-'+uid).remove();});}
function fpsClass(fps){return fps>=15?'green':fps>=8?'orange':'red';}
function refreshAll(){
  fetch('/api/events').then(r=>r.json()).then(d=>{
    document.getElementById('face-log').innerHTML=d.faces.length?d.faces.map(e=>`<div class="event ${e.known?'ev-known':'ev-unknown'}"><span class="ev-time">${e.time}</span><span class="ev-name">${e.name}</span><span class="ev-conf">${e.confidence}</span><span class="ev-cam">${e.camera||''}</span></div>`).join(''):'<div class="empty">No detections yet.</div>';
    document.getElementById('plate-log').innerHTML=d.plates.length?d.plates.map(e=>`<div class="event ${e.known?'ev-known':'ev-unknown'}"><span class="ev-time">${e.time}</span><span class="ev-name">${e.plate}</span><span class="ev-conf">${(e.confidence||0).toFixed(2)}</span><span class="ev-cam">${e.camera||''}</span></div>`).join(''):'<div class="empty">No detections yet.</div>';
  });
  fetch('/api/perf').then(r=>r.json()).then(p=>{
    const el=document.getElementById('uptime-main');if(el)el.textContent=p.uptime;
    const du=document.getElementById('m-uptime-dev');if(du)du.textContent=p.uptime;
    if(devOpen){
      const ff=p.face_fps||0,pf=p.plate_fps||0;
      const setM=(id,v)=>{const e=document.getElementById(id);if(e)e.textContent=v;};
      const setB=(id,v)=>{const e=document.getElementById(id);if(e)e.style.width=Math.min(100,v)+'%';};
      setM('m-face-fps',ff+' fps');setM('m-plate-fps',pf+' fps');
      setM('m-face-inf',p.face_inference+' ms');setM('m-plate-inf',p.plate_inference+' ms');
      setM('m-total-faces',p.total_faces);setM('m-total-plates',p.total_plates);setM('m-cameras',p.active_cameras);
      setB('b-face-fps',(ff/30)*100);setB('b-plate-fps',(pf/30)*100);
      setB('b-face-inf',(p.face_inference/500)*100);setB('b-plate-inf',(p.plate_inference/500)*100);
      const fc=document.getElementById('m-face-fps');const pc=document.getElementById('m-plate-fps');
      if(fc)fc.className='metric-val '+fpsClass(ff);if(pc)pc.className='metric-val '+fpsClass(pf);
    }
  });
  fetch('/api/unknowns').then(r=>r.json()).then(d=>{
    const uf=document.getElementById('unknown-faces');
    const up=document.getElementById('unknown-plates');
    d.faces.forEach(f=>{if(!document.getElementById('card-'+f.id)){const c=document.createElement('div');c.className='enrol-card';c.id='card-'+f.id;c.innerHTML=`<img src="/unknown_img/face/${f.id}" alt="Unknown"><div style="font-size:0.68rem;color:var(--muted);margin-bottom:6px">${f.time} · ${f.camera||''}</div><input id="name-${f.id}" type="text" placeholder="Enter name..."><button class="enrol-btn" onclick="enrolFace('${f.id}')">Enrol Face</button>`;const em=uf.querySelector('.empty');if(em)em.remove();uf.appendChild(c);}});
    if(!d.faces.length&&!uf.querySelector('.enrol-card'))uf.innerHTML='<div class="empty">No unknown faces detected yet.</div>';
    d.plates.forEach(p=>{if(!document.getElementById('card-'+p.id)){const c=document.createElement('div');c.className='enrol-card';c.id='card-'+p.id;c.innerHTML=`<img src="/unknown_img/plate/${p.id}" alt="Plate"><div style="font-size:0.68rem;color:var(--muted);margin-bottom:6px">${p.time} · ${p.camera||''}</div><input id="plate-${p.id}" type="text" value="${p.plate}" placeholder="Plate..."><button class="plate-btn" onclick="enrolPlate('${p.id}')">Add to Whitelist</button>`;const em=up.querySelector('.empty');if(em)em.remove();up.appendChild(c);}});
    if(!d.plates.length&&!up.querySelector('.enrol-card'))up.innerHTML='<div class="empty">No unknown plates detected yet.</div>';
  });
  document.getElementById('clock').textContent=new Date().toLocaleTimeString();
}
setInterval(refreshAll,2000);
refreshAll();
</script>
</body></html>
"""


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    log.info("="*60)
    log.info("CIS3425 Smart Home Security System")
    log.info("="*60)

    for d in [CONFIG["log_frames_dir"],
              os.path.join(CONFIG["log_frames_dir"],"plates"),
              os.path.join(CONFIG["unknowns_dir"],"faces"),
              os.path.join(CONFIG["unknowns_dir"],"plates"),
              CONFIG["known_faces_dir"]]:
        os.makedirs(d, exist_ok=True)

    db = SecurityDatabase(CONFIG["db_path"])

    log.info("Enrolling known faces...")
    enrol_faces_from_dir(db, CONFIG["known_faces_dir"], force=False)

    state = SharedState()
    state.reload_encodings(db.load_face_encodings())
    state.reload_plates(db.load_known_plates())

    # Load plates from file
    pf = Path(CONFIG["known_plates_file"])
    if pf.exists():
        with open(pf) as f:
            for line in f:
                p = line.strip().upper().replace(" ","")
                if p:
                    state.known_plates.add(p)
                    db.add_known_plate(p)

    # ── Start camera threads (one per camera) ─────────────────────────────────
    threads = []
    for cam in CONFIG["cameras"]:
        if cam.get("enabled", True):
            t = threading.Thread(target=camera_thread,
                                 args=(state, cam),
                                 daemon=True, name=f"Camera-{cam['id']}")
            threads.append(t)

    # ── Start processing threads ───────────────────────────────────────────────
    threads += [
        threading.Thread(target=face_thread,  args=(state, db, CONFIG),
                         daemon=True, name="FaceRecognition"),
        threading.Thread(target=plate_thread, args=(state, db, CONFIG),
                         daemon=True, name="PlateDetection"),
    ]

    for t in threads:
        log.info(f"Starting: {t.name}")
        t.start()

    # ── Flask (blocking — runs in main thread) ────────────────────────────────
    app = make_flask_app(state, db, CONFIG)
    log.info(f"Dashboard: http://localhost:{CONFIG['flask_port']}")
    log.info(f"Default login: admin / admin123")
    log.info("Press Ctrl+C to stop.")

    try:
        app.run(host=CONFIG["flask_host"], port=CONFIG["flask_port"],
                debug=False, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        log.info("Shutting down...")
        state.running = False
        time.sleep(2)
        log.info("Stopped.")
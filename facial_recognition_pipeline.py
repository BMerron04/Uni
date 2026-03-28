"""
CIS3425 Smart Home Security System
Facial Recognition Pipeline — face_recognition (dlib) + YOLOv8
Author: [Your Name]

Pipeline:
  YOLOv8 detects faces/people → face_recognition encodes detected face crops
  → cosine similarity against known encodings → Known / Unknown classification
  → All events logged to SQLite with timestamp, name, confidence, saved frame
"""

import cv2
import face_recognition
import numpy as np
import sqlite3
import os
import time
import logging
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
CONFIG = {
    "known_faces_dir":      "known_faces",       # One sub-folder per person
    "embeddings_db":        "security.db",        # SQLite database
    "log_frames_dir":       "logged_frames",      # Saved detection frames

    "yolo_model":           "yolov8n.pt",         # Swap for your custom weights e.g. yolo26n.pt
    "yolo_confidence":      0.60,                 # Minimum YOLO detection confidence

    "recognition_tolerance": 0.50,                # Lower = stricter. 0.5 works well for dlib
    "frame_skip":           4,                    # Process every Nth frame
    "unknown_cooldown":     10,                   # Seconds before re-logging same unknown
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════════════════════════════════════════

class SecurityDatabase:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()
        log.info(f"Database connected: {db_path}")

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS known_faces (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                person_name TEXT NOT NULL,
                encoding    BLOB NOT NULL,
                source_img  TEXT,
                enrolled_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS detection_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT NOT NULL,
                person_name TEXT NOT NULL,
                confidence  REAL,
                frame_path  TEXT,
                bbox_x      INTEGER,
                bbox_y      INTEGER,
                bbox_w      INTEGER,
                bbox_h      INTEGER
            );
        """)
        self.conn.commit()

    def save_encoding(self, name: str, encoding: np.ndarray, source: str = ""):
        self.conn.execute(
            "INSERT INTO known_faces (person_name, encoding, source_img) VALUES (?,?,?)",
            (name, encoding.tobytes(), source)
        )
        self.conn.commit()

    def load_encodings(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT person_name, encoding FROM known_faces"
        ).fetchall()
        return [
            {"name": n, "encoding": np.frombuffer(e, dtype=np.float64)}
            for n, e in rows
        ]

    def log_detection(self, name: str, confidence: float,
                      frame_path: str, bbox: tuple):
        x, y, w, h = bbox
        self.conn.execute(
            """INSERT INTO detection_log
               (timestamp, person_name, confidence, frame_path,
                bbox_x, bbox_y, bbox_w, bbox_h)
               VALUES (?,?,?,?,?,?,?,?)""",
            (datetime.now().isoformat(), name, confidence,
             frame_path, x, y, w, h)
        )
        self.conn.commit()

    def clear_encodings(self):
        self.conn.execute("DELETE FROM known_faces")
        self.conn.commit()

    def close(self):
        self.conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
# ENROLMENT
# ═══════════════════════════════════════════════════════════════════════════════

class FaceEnroller:
    """
    Walks known_faces/<PersonName>/*.jpg
    Generates 128-d dlib face encodings and stores them in SQLite.

    Folder structure:
        known_faces/
            Alice/
                alice1.jpg
                alice2.jpg
            Bob/
                bob1.jpg
    """

    def __init__(self, db: SecurityDatabase, known_faces_dir: str):
        self.db = db
        self.dir = Path(known_faces_dir)

    def enrol_all(self, force: bool = False):
        if force:
            self.db.clear_encodings()
            log.info("Cleared existing encodings.")

        enrolled = 0
        for person_dir in sorted(self.dir.iterdir()):
            if not person_dir.is_dir():
                continue
            name = person_dir.name
            images = (list(person_dir.glob("*.jpg")) +
                      list(person_dir.glob("*.jpeg")) +
                      list(person_dir.glob("*.png")))

            for img_path in images:
                try:
                    img = face_recognition.load_image_file(str(img_path))
                    encodings = face_recognition.face_encodings(img)
                    if not encodings:
                        log.warning(f"  No face found in {img_path.name} — skipping")
                        continue
                    self.db.save_encoding(name, encodings[0], str(img_path))
                    log.info(f"  Enrolled {name} from {img_path.name}")
                    enrolled += 1
                except Exception as e:
                    log.warning(f"  Failed {img_path.name}: {e}")

        log.info(f"Enrolment complete — {enrolled} encodings stored.")


# ═══════════════════════════════════════════════════════════════════════════════
# RECOGNITION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class FaceRecognitionEngine:
    """
    Identifies a face crop against all enrolled encodings.
    Uses dlib's compare_faces + face_distance for confidence scoring.
    """

    def __init__(self, db: SecurityDatabase, tolerance: float):
        self.tolerance = tolerance
        self.known: list[dict] = []
        self._load(db)

    def _load(self, db: SecurityDatabase):
        self.known = db.load_encodings()
        log.info(f"Loaded {len(self.known)} known face encodings.")

    def reload(self, db: SecurityDatabase):
        self._load(db)

    def identify(self, face_crop_bgr: np.ndarray) -> tuple[str, float]:
        """
        Returns (name, confidence) where confidence is 1 - face_distance.
        Returns ('Unknown', best_confidence) if no match above tolerance.
        """
        if not self.known:
            return "Unknown", 0.0

        # face_recognition expects RGB
        rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb)

        if not encodings:
            return "Unknown", 0.0

        query = encodings[0]
        known_encs = [e["encoding"] for e in self.known]
        known_names = [e["name"] for e in self.known]

        # face_distance: lower = more similar
        distances = face_recognition.face_distance(known_encs, query)
        best_idx = int(np.argmin(distances))
        best_dist = float(distances[best_idx])
        confidence = round(1.0 - best_dist, 3)

        matches = face_recognition.compare_faces(
            known_encs, query, tolerance=self.tolerance
        )

        if matches[best_idx]:
            return known_names[best_idx], confidence
        return "Unknown", confidence


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

class SmartSecurityPipeline:
    """
    Full pipeline:
      1. YOLOv8 detects faces/persons in each frame
      2. Each detection crop passed to FaceRecognitionEngine
      3. Results annotated on frame and logged to SQLite
    """

    def __init__(self, config: dict):
        self.cfg = config
        os.makedirs(config["log_frames_dir"], exist_ok=True)

        log.info("Loading YOLOv8...")
        self.yolo = YOLO(config["yolo_model"])

        log.info("Connecting to database...")
        self.db = SecurityDatabase(config["embeddings_db"])

        log.info("Loading recognition engine...")
        self.engine = FaceRecognitionEngine(self.db, config["recognition_tolerance"])

        self._cooldowns: dict[str, float] = {}

    def enrol(self, force: bool = False):
        enroller = FaceEnroller(self.db, self.cfg["known_faces_dir"])
        enroller.enrol_all(force=force)
        self.engine.reload(self.db)

    def run(self, source: int | str = 0):
        """
        source = 0 for webcam
        source = "path/to/video.mp4" for video file
        source = "rtsp://..." for IP camera

        Keyboard shortcuts:
          q = quit
          r = reload encodings (after adding new photos)
          e = re-enrol from scratch
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            log.error(f"Cannot open source: {source}")
            return

        log.info(f"Pipeline running. Press Q to quit, R to reload, E to re-enrol.")
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                log.warning("Stream ended or frame read failed.")
                break

            frame_count += 1
            if frame_count % self.cfg["frame_skip"] != 0:
                cv2.imshow("CIS3425 Security System", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # ── YOLOv8 Detection ──────────────────────────────────────────────
            results = self.yolo(frame, verbose=False)[0]
            annotated = frame.copy()

            for box in results.boxes:
                conf = float(box.conf[0])
                if conf < self.cfg["yolo_confidence"]:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Clamp to frame bounds
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                # ── Recognition ───────────────────────────────────────────────
                name, similarity = self.engine.identify(crop)

                # ── Draw annotation ───────────────────────────────────────────
                colour = (0, 200, 0) if name != "Unknown" else (0, 0,220)
                label  = f"{name}  {similarity:.2f}"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)
                cv2.rectangle(annotated, (x1, y1 - 22), (x1 + len(label) * 11, y1),
                              colour, -1)
                cv2.putText(annotated, label, (x1 + 2, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # ── Log event ─────────────────────────────────────────────────
                self._log_event(name, similarity, annotated,
                                (x1, y1, x2 - x1, y2 - y1))

            cv2.imshow("CIS3425 Security System", annotated)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('r'):
                log.info("Reloading encodings...")
                self.engine.reload(self.db)
            elif key == ord('e'):
                log.info("Re-enrolling...")
                self.enrol(force=True)

        cap.release()
        cv2.destroyAllWindows()
        self.db.close()
        log.info("Pipeline shut down.")

    def _log_event(self, name: str, confidence: float,
                   frame: np.ndarray, bbox: tuple):
        region = f"{bbox[0] // 50}_{bbox[1] // 50}"  # grid-based region key

        if name == "Unknown":
            last = self._cooldowns.get(region, 0)
            if time.time() - last < self.cfg["unknown_cooldown"]:
                return
            self._cooldowns[region] = time.time()

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        frame_path = os.path.join(self.cfg["log_frames_dir"], f"{name}_{ts}.jpg")
        cv2.imwrite(frame_path, frame)
        self.db.log_detection(name, confidence, frame_path, bbox)
        log.info(f"  → Logged: {name} | confidence={confidence:.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pipeline = SmartSecurityPipeline(CONFIG)

    # Enrol known faces from known_faces/ directory
    # Set force=True to re-enrol (e.g. after adding new photos)
    pipeline.enrol(force=False)

    # Run on webcam
    # Change source to a video path or RTSP URL for IP camera
    pipeline.run(source=0)
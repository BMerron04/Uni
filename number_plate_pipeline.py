"""
CIS3425 Smart Home Security System
Number Plate Detection & OCR Pipeline
Author: [Your Name]

Pipeline:
  YOLOv8 detects vehicles + number plates → EasyOCR reads plate text
  → Known plates checked against whitelist → All events logged to SQLite
  
Run:
  python number_plate_pipeline.py
  
Keyboard shortcuts:
  q = quit
  a = add last detected plate to whitelist
  r = reload whitelist
"""

import cv2
import easyocr
import numpy as np
import sqlite3
import os
import re
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
    # Paths
    "db_path":          "security.db",
    "log_frames_dir":   "logged_frames/plates",
    "whitelist_file":   "known_plates.txt",       # One plate per line e.g. AB12CDE

    # YOLOv8 — general model detects cars; swap for a plate-specific model if available
    "yolo_model":       "yolov8n.pt",
    "yolo_confidence":  0.45,

    # YOLO class IDs to treat as vehicles (COCO dataset IDs)
    # 2=car, 3=motorcycle, 5=bus, 7=truck
    "vehicle_classes":  [2, 3, 5, 7],

    # EasyOCR
    "ocr_languages":    ["en"],                   # Add more e.g. ["en", "fr"] if needed
    "ocr_confidence":   0.25,                     # Minimum OCR confidence to accept text

    # Plate validation
    "min_plate_chars":  4,                        # Minimum characters to be considered a plate
    "max_plate_chars":  10,                       # Maximum characters

    # Performance
    "frame_skip":       3,
    "plate_cooldown":   15,                       # Seconds before re-logging same plate
}

# ── UK plate regex (optional validation) ──────────────────────────────────────
# Matches formats like: AB12CDE, AB12 CDE, A123 BCD etc.
UK_PLATE_PATTERN = re.compile(
    r"^[A-Z]{2}[0-9]{2}\s?[A-Z]{3}$|"       # Current format:  AB12 CDE
    r"^[A-Z][0-9]{1,3}\s?[A-Z]{3}$|"         # Prefix format:   A123 BCD
    r"^[A-Z]{3}\s?[0-9]{1,3}[A-Z]$",         # Suffix format:   ABC 123D
    re.IGNORECASE
)


# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════════════════════════════════════════

class PlateDatabase:
    """
    Extends the shared security.db with a plate_log table.
    Reuses the same database as the facial recognition pipeline.
    """

    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()
        log.info(f"Plate database connected: {db_path}")

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS plate_log (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp    TEXT NOT NULL,
                plate_text   TEXT NOT NULL,
                ocr_conf     REAL,
                is_known     INTEGER DEFAULT 0,   -- 1 = whitelisted, 0 = unknown
                frame_path   TEXT,
                bbox_x       INTEGER,
                bbox_y       INTEGER,
                bbox_w       INTEGER,
                bbox_h       INTEGER
            );

            CREATE TABLE IF NOT EXISTS known_plates (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_text   TEXT UNIQUE NOT NULL,
                label        TEXT,                -- e.g. "Owner's Car", "Visitor"
                added_at     TEXT DEFAULT (datetime('now'))
            );
        """)
        self.conn.commit()

    def log_plate(self, plate_text: str, ocr_conf: float,
                  is_known: bool, frame_path: str, bbox: tuple):
        x, y, w, h = bbox
        self.conn.execute(
            """INSERT INTO plate_log
               (timestamp, plate_text, ocr_conf, is_known, frame_path,
                bbox_x, bbox_y, bbox_w, bbox_h)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (datetime.now().isoformat(), plate_text, ocr_conf,
             int(is_known), frame_path, x, y, w, h)
        )
        self.conn.commit()

    def add_known_plate(self, plate_text: str, label: str = ""):
        try:
            self.conn.execute(
                "INSERT OR IGNORE INTO known_plates (plate_text, label) VALUES (?,?)",
                (plate_text.upper().strip(), label)
            )
            self.conn.commit()
            log.info(f"Added to known plates: {plate_text}")
        except Exception as e:
            log.warning(f"Could not add plate: {e}")

    def load_known_plates(self) -> set[str]:
        rows = self.conn.execute(
            "SELECT plate_text FROM known_plates"
        ).fetchall()
        return {r[0].upper().strip() for r in rows}

    def get_recent_plates(self, limit: int = 20) -> list[dict]:
        rows = self.conn.execute(
            """SELECT timestamp, plate_text, ocr_conf, is_known
               FROM plate_log ORDER BY id DESC LIMIT ?""",
            (limit,)
        ).fetchall()
        return [
            {"timestamp": r[0], "plate": r[1],
             "confidence": r[2], "known": bool(r[3])}
            for r in rows
        ]

    def close(self):
        self.conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
# WHITELIST LOADER
# ═══════════════════════════════════════════════════════════════════════════════

class PlateWhitelist:
    """
    Manages known plates from both:
      - known_plates.txt  (flat file, one plate per line)
      - SQLite known_plates table
    Combining both allows easy manual entry (text file) + programmatic addition.
    """

    def __init__(self, db: PlateDatabase, whitelist_file: str):
        self.db = db
        self.file = Path(whitelist_file)
        self.known: set[str] = set()
        self._load()

    def _load(self):
        # Load from file
        file_plates = set()
        if self.file.exists():
            with open(self.file, "r") as f:
                for line in f:
                    plate = line.strip().upper().replace(" ", "")
                    if plate:
                        file_plates.add(plate)
                        self.db.add_known_plate(plate, "from_file")

        # Load from DB
        db_plates = self.db.load_known_plates()
        self.known = file_plates | db_plates
        log.info(f"Whitelist loaded: {len(self.known)} known plates")

    def reload(self):
        self._load()

    def is_known(self, plate_text: str) -> bool:
        normalised = plate_text.upper().replace(" ", "")
        return normalised in self.known

    def add(self, plate_text: str):
        normalised = plate_text.upper().replace(" ", "")
        self.known.add(normalised)
        self.db.add_known_plate(normalised)
        # Also append to file
        with open(self.file, "a") as f:
            f.write(normalised + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# OCR ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class PlateOCR:
    """
    Wraps EasyOCR for number plate text extraction.
    Applies preprocessing to improve OCR accuracy on plate crops.
    """

    def __init__(self, languages: list[str]):
        log.info("Initialising EasyOCR (first run downloads models ~100MB)...")
        self.reader = easyocr.Reader(languages, gpu=True)
        log.info("EasyOCR ready.")

    def read_plate(self, plate_crop_bgr: np.ndarray) -> tuple[str, float]:
        """
        Returns (plate_text, confidence).
        Returns ('', 0.0) if nothing readable found.
        """
        processed = self._preprocess(plate_crop_bgr)
        try:
            results = self.reader.readtext(processed)
        except Exception as e:
            log.debug(f"OCR failed: {e}")
            return "", 0.0

        if not results:
            return "", 0.0

        # Concatenate all text boxes, take average confidence
        texts = []
        confs = []
        for (_, text, conf) in results:
            clean = self._clean_text(text)
            if clean:
                texts.append(clean)
                confs.append(conf)

        if not texts:
            return "", 0.0

        full_text = "".join(texts).upper()
        avg_conf  = float(np.mean(confs))
        return full_text, avg_conf

    @staticmethod
    def _preprocess(img: np.ndarray) -> np.ndarray:
        """
        Preprocessing pipeline to improve OCR accuracy:
        1. Upscale small crops
        2. Convert to greyscale
        3. Apply CLAHE for contrast normalisation
        4. Threshold to binary
        """
        # Upscale if too small
        h, w = img.shape[:2]
        if w < 200:
            scale = 200 / w
            img = cv2.resize(img, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_CUBIC)

        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # CLAHE — improves contrast in varying lighting conditions
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        grey = clahe.apply(grey)

        # Otsu thresholding — adaptive binarisation
        _, binary = cv2.threshold(grey, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    @staticmethod
    def _clean_text(text: str) -> str:
        """Remove non-alphanumeric characters, keep letters and digits only."""
        return re.sub(r"[^A-Za-z0-9]", "", text).strip()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

class NumberPlatePipeline:
    """
    Full pipeline:
      1. YOLOv8 detects vehicles in each frame
      2. Vehicle crops passed to EasyOCR for plate reading
      3. Plate text checked against whitelist
      4. Results annotated and logged to SQLite
    """

    def __init__(self, config: dict):
        self.cfg = config
        os.makedirs(config["log_frames_dir"], exist_ok=True)

        log.info("Loading YOLOv8...")
        self.yolo = YOLO(config["yolo_model"])

        log.info("Connecting to database...")
        self.db = PlateDatabase(config["db_path"])

        log.info("Loading whitelist...")
        self.whitelist = PlateWhitelist(self.db, config["whitelist_file"])

        log.info("Loading OCR engine...")
        self.ocr = PlateOCR(config["ocr_languages"])

        self._cooldowns: dict[str, float] = {}
        self._last_plate: str = ""

    def run(self, source: int | str = 0):
        """
        source = 0 for webcam
        source = "path/to/video.mp4" for recorded footage

        Keyboard shortcuts:
          q = quit
          a = add last detected plate to whitelist
          r = reload whitelist
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            log.error(f"Cannot open source: {source}")
            return

        log.info("Number plate pipeline running. Press Q to quit.")
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % self.cfg["frame_skip"] != 0:
                cv2.imshow("CIS3425 — Number Plate Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # ── YOLOv8 Detection ──────────────────────────────────────────────
            results = self.yolo(frame, verbose=False)[0]
            annotated = frame.copy()

            for box in results.boxes:
                conf = float(box.conf[0])
                cls  = int(box.cls[0])

                if conf < self.cfg["yolo_confidence"]:
                    continue

                # Only process vehicle classes
                if cls not in self.cfg["vehicle_classes"]:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)

                vehicle_crop = frame[y1:y2, x1:x2]
                if vehicle_crop.size == 0:
                    continue

                # ── OCR ───────────────────────────────────────────────────────
                plate_text, ocr_conf = self.ocr.read_plate(vehicle_crop)

                # Filter out short/invalid reads
                if (len(plate_text) < self.cfg["min_plate_chars"] or
                        len(plate_text) > self.cfg["max_plate_chars"] or
                        ocr_conf < self.cfg["ocr_confidence"]):
                    # Still draw vehicle box without plate text
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (200, 200, 0), 2)
                    cv2.putText(annotated, "Vehicle", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 2)
                    continue

                # ── Whitelist check ───────────────────────────────────────────
                is_known = self.whitelist.is_known(plate_text)
                self._last_plate = plate_text

                # ── Annotate ──────────────────────────────────────────────────
                colour = (0, 200, 0) if is_known else (0, 0, 220)
                status = "KNOWN" if is_known else "UNKNOWN"
                label  = f"{plate_text} [{status}] {ocr_conf:.2f}"

                cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)
                cv2.rectangle(annotated, (x1, y2), (x1 + len(label) * 10, y2 + 22),
                              colour, -1)
                cv2.putText(annotated, label, (x1 + 2, y2 + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

                # ── Log ───────────────────────────────────────────────────────
                self._log_event(plate_text, ocr_conf, is_known,
                                annotated, (x1, y1, x2 - x1, y2 - y1))

            # ── HUD overlay ───────────────────────────────────────────────────
            cv2.putText(annotated,
                        f"Last plate: {self._last_plate}  |  Q=quit  A=add to whitelist",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            cv2.imshow("CIS3425 — Number Plate Detection", annotated)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('a') and self._last_plate:
                self.whitelist.add(self._last_plate)
                log.info(f"Added to whitelist: {self._last_plate}")
            elif key == ord('r'):
                self.whitelist.reload()

        cap.release()
        cv2.destroyAllWindows()
        self.db.close()
        log.info("Number plate pipeline shut down.")

    def _log_event(self, plate_text: str, ocr_conf: float,
                   is_known: bool, frame: np.ndarray, bbox: tuple):
        """Rate-limited logging — avoids re-logging same plate repeatedly."""
        key = plate_text.upper().replace(" ", "")
        last = self._cooldowns.get(key, 0)
        if time.time() - last < self.cfg["plate_cooldown"]:
            return
        self._cooldowns[key] = time.time()

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        frame_path = os.path.join(
            self.cfg["log_frames_dir"], f"{plate_text}_{ts}.jpg"
        )
        cv2.imwrite(frame_path, frame)
        self.db.log_plate(plate_text, ocr_conf, is_known, frame_path, bbox)
        log.info(f"  → Plate logged: {plate_text} | known={is_known} | conf={ocr_conf:.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pipeline = NumberPlatePipeline(CONFIG)

    # Run on webcam (source=0) or pass a video file path
    # For best results, use video footage of vehicles rather than webcam
    pipeline.run(source=0)
"""
CIS3425 Smart Home Security System
Dataset Curation Script
Author: [Your Name]

Does two things:
  1. Scans logged_frames/ for face crops of known people
     and moves the best quality ones into known_faces/Name/
  2. Removes near-duplicate images within each person's folder
     using perceptual hashing (pHash) — keeps variety, removes redundancy

Usage:
  python curate_dataset.py                    # Interactive mode
  python curate_dataset.py --auto             # Auto-curate without prompts
  python curate_dataset.py --dedupe-only      # Only remove duplicates
  python curate_dataset.py --move-only        # Only move logged frames
  python curate_dataset.py --threshold 12     # Set pHash similarity threshold (default 10)
"""

import cv2
import face_recognition
import numpy as np
import os
import shutil
import argparse
import logging
import sqlite3
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
CONFIG = {
    "known_faces_dir":  "known_faces",
    "logged_frames_dir":"logged_frames",
    "db_path":          "security.db",
    "min_face_size":    60,       # Minimum face width/height in pixels
    "blur_threshold":   80.0,     # Laplacian variance — below this = too blurry
    "phash_threshold":  10,       # pHash distance — below this = near duplicate (0-64, lower=stricter)
    "max_auto_move":    10,       # Max images to auto-move per person per run
}


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE QUALITY CHECKS
# ═══════════════════════════════════════════════════════════════════════════════

def blur_score(img_bgr: np.ndarray) -> float:
    """
    Laplacian variance — higher = sharper.
    Below ~80 is typically too blurry for reliable face encoding.
    """
    grey = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(grey, cv2.CV_64F).var())


def brightness_score(img_bgr: np.ndarray) -> float:
    """Mean brightness 0-255. Too dark (<40) or too bright (>220) = poor quality."""
    grey = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.mean(grey))


def face_size(img_bgr: np.ndarray) -> tuple[int, int]:
    """Returns (width, height) of image."""
    h, w = img_bgr.shape[:2]
    return w, h


def quality_score(img_bgr: np.ndarray) -> float:
    """
    Combined quality score (0-100).
    Weights: sharpness (60%) + brightness balance (40%)
    """
    blur    = blur_score(img_bgr)
    bright  = brightness_score(img_bgr)

    # Normalise blur: cap at 500
    blur_norm = min(blur / 500.0, 1.0) * 60

    # Penalise extreme brightness
    bright_norm = (1.0 - abs(bright - 128) / 128) * 40

    return round(blur_norm + bright_norm, 2)


def is_good_quality(img_bgr: np.ndarray, cfg: dict) -> tuple[bool, str]:
    """Returns (pass, reason)."""
    w, h = face_size(img_bgr)
    if w < cfg["min_face_size"] or h < cfg["min_face_size"]:
        return False, f"Too small ({w}x{h}px)"

    blur = blur_score(img_bgr)
    if blur < cfg["blur_threshold"]:
        return False, f"Too blurry (score={blur:.1f})"

    bright = brightness_score(img_bgr)
    if bright < 30:
        return False, f"Too dark (brightness={bright:.1f})"
    if bright > 230:
        return False, f"Too bright (brightness={bright:.1f})"

    return True, "OK"


# ═══════════════════════════════════════════════════════════════════════════════
# PERCEPTUAL HASHING
# ═══════════════════════════════════════════════════════════════════════════════

def phash(img_bgr: np.ndarray, hash_size: int = 8) -> np.ndarray:
    """
    Perceptual hash (pHash) of an image.
    Returns a binary array of hash_size^2 bits.
    Similar images produce similar hashes regardless of minor variations.
    """
    grey    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(grey, (hash_size * 4, hash_size * 4))
    # DCT-based hash
    dct     = cv2.dct(np.float32(resized))
    dct_low = dct[:hash_size, :hash_size]
    mean    = np.mean(dct_low)
    return (dct_low > mean).flatten()


def hamming_distance(h1: np.ndarray, h2: np.ndarray) -> int:
    """Number of differing bits between two hashes. 0 = identical."""
    return int(np.sum(h1 != h2))


def find_duplicates(image_paths: list[str], threshold: int) -> list[list[str]]:
    """
    Groups near-duplicate images together.
    Returns list of groups — each group is a list of similar image paths.
    """
    hashes = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            hashes.append(None)
            continue
        hashes.append(phash(img))

    groups = []
    used   = set()

    for i, h1 in enumerate(hashes):
        if i in used or h1 is None:
            continue
        group = [image_paths[i]]
        for j, h2 in enumerate(hashes):
            if j <= i or j in used or h2 is None:
                continue
            if hamming_distance(h1, h2) <= threshold:
                group.append(image_paths[j])
                used.add(j)
        if len(group) > 1:
            groups.append(group)
        used.add(i)

    return groups


def deduplicate_folder(folder: str, threshold: int, dry_run: bool = False) -> int:
    """
    Remove near-duplicate images from a folder.
    Keeps the highest quality image from each duplicate group.
    Returns number of images removed.
    """
    images = sorted([
        str(p) for p in Path(folder).glob("*.jpg")
    ] + [
        str(p) for p in Path(folder).glob("*.png")
    ])

    if len(images) < 2:
        return 0

    groups = find_duplicates(images, threshold)
    removed = 0

    for group in groups:
        # Score each image in the group
        scored = []
        for path in group:
            img = cv2.imread(path)
            if img is not None:
                scored.append((quality_score(img), path))

        if not scored:
            continue

        # Keep highest quality, remove the rest
        scored.sort(reverse=True)
        keep   = scored[0][1]
        remove = [s[1] for s in scored[1:]]

        for path in remove:
            log.info(f"  {'[DRY RUN] ' if dry_run else ''}Removing duplicate: {Path(path).name}")
            if not dry_run:
                os.remove(path)
            removed += 1

    return removed


# ═══════════════════════════════════════════════════════════════════════════════
# MOVE LOGGED FRAMES TO KNOWN FACES
# ═══════════════════════════════════════════════════════════════════════════════

def extract_name_from_filename(filename: str) -> str | None:
    """
    Extract person name from logged frame filename.
    Expected format: face_NAME_TIMESTAMP.jpg
    e.g. face_Brad_20260328_123456_789.jpg → 'Brad'
    """
    stem = Path(filename).stem  # remove extension
    parts = stem.split("_")
    if len(parts) >= 2 and parts[0] == "face":
        # Name is everything between 'face_' and the timestamp
        # Timestamps are pure digits, names are not
        name_parts = []
        for part in parts[1:]:
            if part.isdigit():
                break
            name_parts.append(part)
        name = "_".join(name_parts)
        if name and name != "Unknown":
            return name
    return None


def has_face(img_bgr: np.ndarray) -> bool:
    """Quick check if image contains a detectable face."""
    rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    locs = face_recognition.face_locations(rgb, model="hog")
    return len(locs) > 0


def move_logged_frames(cfg: dict, auto: bool = False,
                       max_per_person: int = 10) -> dict[str, int]:
    """
    Scans logged_frames/ for known-person images.
    Moves quality images to known_faces/Name/.
    Returns dict of {name: count_moved}.
    """
    log_dir   = Path(cfg["logged_frames_dir"])
    faces_dir = Path(cfg["known_faces_dir"])
    moved     = {}

    # Find all logged face frames
    logged_images = list(log_dir.glob("face_*.jpg"))
    log.info(f"Found {len(logged_images)} logged face frames to scan.")

    per_person_count = {}

    for img_path in sorted(logged_images, key=os.path.getmtime, reverse=True):
        name = extract_name_from_filename(img_path.name)
        if not name:
            continue

        # Skip Unknown
        if name.lower() == "unknown":
            continue

        # Limit per person per run
        count = per_person_count.get(name, 0)
        if count >= max_per_person:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Quality check
        ok, reason = is_good_quality(img, cfg)
        if not ok:
            log.debug(f"  Skipping {img_path.name}: {reason}")
            continue

        # Check for face
        if not has_face(img):
            log.debug(f"  No face detected in {img_path.name} — skipping")
            continue

        qs = quality_score(img)

        if not auto:
            print(f"\n  Person: {name}")
            print(f"  File:   {img_path.name}")
            print(f"  Quality score: {qs:.1f}/100  Blur: {blur_score(img):.1f}")
            ans = input("  Move to known_faces? [y/n/q]: ").strip().lower()
            if ans == 'q':
                break
            if ans != 'y':
                continue

        # Move to known_faces/Name/
        dest_dir = faces_dir / name
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{name}_log_{img_path.stem[-12:]}.jpg"

        shutil.copy2(str(img_path), str(dest))
        img_path.unlink()  # Remove from logged_frames

        log.info(f"  Moved: {img_path.name} → known_faces/{name}/{dest.name}  (quality={qs:.1f})")
        moved[name] = moved.get(name, 0) + 1
        per_person_count[name] = count + 1

    return moved


# ═══════════════════════════════════════════════════════════════════════════════
# RE-ENROL FROM KNOWN_FACES FOLDER
# ═══════════════════════════════════════════════════════════════════════════════

def reenrol_all(cfg: dict):
    """
    Clears all face encodings and re-encodes from known_faces/ folder.
    Run this after curating to update the database with new images.
    """
    db_path = cfg["db_path"]
    conn    = sqlite3.connect(db_path)
    conn.execute("DELETE FROM known_faces")
    conn.commit()

    enrolled = 0
    faces_dir = Path(cfg["known_faces_dir"])

    for person_dir in sorted(faces_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        name = person_dir.name
        images = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png"))

        person_enrolled = 0
        for img_path in images:
            try:
                img = face_recognition.load_image_file(str(img_path))
                encs = face_recognition.face_encodings(img)
                if encs:
                    conn.execute(
                        "INSERT INTO known_faces (person_name, encoding, source_img) VALUES (?,?,?)",
                        (name, encs[0].tobytes(), str(img_path))
                    )
                    person_enrolled += 1
            except Exception as e:
                log.warning(f"  Encoding failed {img_path.name}: {e}")

        conn.commit()
        log.info(f"  Re-enrolled {name}: {person_enrolled}/{len(images)} images")
        enrolled += person_enrolled

    conn.close()
    log.info(f"Re-enrolment complete — {enrolled} total encodings stored.")
    return enrolled


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary(cfg: dict):
    """Print a summary of the current known_faces dataset."""
    faces_dir = Path(cfg["known_faces_dir"])
    print("\n" + "="*50)
    print("KNOWN FACES DATASET SUMMARY")
    print("="*50)
    total_images = 0
    for person_dir in sorted(faces_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        images = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png"))
        scores = []
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is not None:
                scores.append(quality_score(img))
        avg_q = np.mean(scores) if scores else 0
        print(f"  {person_dir.name:<20} {len(images):>4} images  avg quality={avg_q:.1f}")
        total_images += len(images)
    print(f"\n  Total: {total_images} images across {sum(1 for p in faces_dir.iterdir() if p.is_dir())} people")
    print("="*50 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIS3425 Dataset Curation Tool")
    parser.add_argument("--auto",        action="store_true", help="Auto-move without prompts")
    parser.add_argument("--dedupe-only", action="store_true", help="Only remove duplicates")
    parser.add_argument("--move-only",   action="store_true", help="Only move logged frames")
    parser.add_argument("--threshold",   type=int, default=CONFIG["phash_threshold"],
                        help="pHash similarity threshold (default 10)")
    parser.add_argument("--dry-run",     action="store_true", help="Preview without making changes")
    args = parser.parse_args()

    CONFIG["phash_threshold"] = args.threshold

    print("\n" + "="*50)
    print("CIS3425 Dataset Curation Tool")
    print("="*50)

    print_summary(CONFIG)

    # ── Step 1: Move logged frames ─────────────────────────────────────────────
    if not args.dedupe_only:
        print("\n📁 STEP 1: Moving quality logged frames to known_faces/")
        print("-"*50)
        moved = move_logged_frames(
            CONFIG,
            auto=args.auto,
            max_per_person=CONFIG["max_auto_move"]
        )
        if moved:
            print(f"\n  Moved: {dict(moved)}")
        else:
            print("  No frames moved.")

    # ── Step 2: Remove duplicates ──────────────────────────────────────────────
    if not args.move_only:
        print("\n🔍 STEP 2: Removing near-duplicate images")
        print(f"  pHash threshold: {CONFIG['phash_threshold']} (lower = stricter)")
        print("-"*50)
        total_removed = 0
        faces_dir = Path(CONFIG["known_faces_dir"])
        for person_dir in sorted(faces_dir.iterdir()):
            if not person_dir.is_dir():
                continue
            removed = deduplicate_folder(
                str(person_dir),
                CONFIG["phash_threshold"],
                dry_run=args.dry_run
            )
            if removed > 0:
                log.info(f"  {person_dir.name}: removed {removed} duplicates")
            total_removed += removed
        print(f"\n  Total duplicates removed: {total_removed}")

    # ── Step 3: Re-enrol ───────────────────────────────────────────────────────
    if not args.dry_run:
        print("\n🔄 STEP 3: Re-enrolling from updated known_faces/")
        print("-"*50)
        count = reenrol_all(CONFIG)
        print(f"  {count} face encodings stored in database.")

    # ── Final summary ──────────────────────────────────────────────────────────
    print_summary(CONFIG)
    print("✅ Curation complete. Restart main_server.py to use updated encodings.")
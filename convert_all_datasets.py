

import os
import shutil
import pandas as pd
from PIL import Image

# ------------------------
# CONFIG
# ------------------------
BASE = "datasets"

PEOPLE_DIR = os.path.join(BASE, "People_Detection")
WIDER_TRAIN_DIR = os.path.join(BASE, "WIDER_train", "WIDER_train")
WIDER_VAL_DIR = os.path.join(BASE, "WIDER_val", "WIDER_val")
WIDER_SPLIT_DIR = os.path.join(BASE, "wider_face_split")

OUT_DIR = os.path.join(BASE, "combined")

PERSON_CLASS_ID = 0
FACE_CLASS_ID = 1


PEOPLE_ONLY_CLASSNAME = None  # e.g. "person"

# ------------------------
# HELPERS
# ------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def safe_copy(src: str, dst: str) -> None:
    ensure_dir(os.path.dirname(dst))
    shutil.copy2(src, dst)

def find_existing_path(*candidates: str) -> str:
    """Return the first candidate path that exists."""
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of these paths exist: {candidates}")

def find_col(df: pd.DataFrame, options: list[str]) -> str:
    for c in options:
        if c in df.columns:
            return c
    raise ValueError(f"Missing expected column. Have columns: {list(df.columns)}")

def parse_wider(txt_path: str):
    """
    Resynchronising WIDER FACE parser.
    It only accepts lines containing '.jpg' as image paths.
    If the next line isn't an integer face count, it skips and resyncs.
    """
    data = {}

    def next_nonempty(f):
        while True:
            line = f.readline()
            if not line:
                return None
            line = line.strip()
            if line != "":
                return line

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        while True:
            line = next_nonempty(f)
            if line is None:
                break

            # resync: only treat jpg lines as image keys
            if ".jpg" not in line.lower():
                continue

            img = line
            n_line = next_nonempty(f)
            if n_line is None:
                break

            try:
                n = int(n_line)
            except ValueError:
                # file got out of sync; resync to next jpg
                continue

            boxes = []
            for _ in range(n):
                b = next_nonempty(f)
                if b is None:
                    break
                parts = b.split()
                if len(parts) < 4:
                    continue

                x, y, w, h = map(float, parts[:4])

                # WIDER has extra attributes after w,h; we ignore them.
                # Skip degenerate boxes
                if w > 0 and h > 0:
                    boxes.append((x, y, w, h))

            data[img] = boxes

    return data

# ------------------------
# OUTPUT FOLDERS
# ------------------------
ensure_dir(os.path.join(OUT_DIR, "images", "train"))
ensure_dir(os.path.join(OUT_DIR, "images", "val"))
ensure_dir(os.path.join(OUT_DIR, "labels", "train"))
ensure_dir(os.path.join(OUT_DIR, "labels", "val"))

# =========================
# 1) CONVERT PEOPLE DATASET
# =========================
print("Converting People dataset...")


people_val_folder = "valid" if os.path.exists(os.path.join(PEOPLE_DIR, "valid")) else "val"

for split, out_split in [("train", "train"), (people_val_folder, "val")]:
    split_path = os.path.join(PEOPLE_DIR, split)
    csv_path = os.path.join(split_path, "_annotations.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"People CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # robust column selection (Roboflow exports vary)
    f_col = find_col(df, ["filename", "file", "image", "name"])
    c_col = find_col(df, ["class", "label"]) if any(c in df.columns for c in ["class", "label"]) else None
    xmin = find_col(df, ["xmin", "x_min", "x1", "left"])
    ymin = find_col(df, ["ymin", "y_min", "y1", "top"])
    xmax = find_col(df, ["xmax", "x_max", "x2", "right"])
    ymax = find_col(df, ["ymax", "y_max", "y2", "bottom"])
    wcol = find_col(df, ["width", "image_width", "w"])
    hcol = find_col(df, ["height", "image_height", "h"])

    # group by image
    for filename, g in df.groupby(f_col):
        src_img = os.path.join(split_path, filename)
        if not os.path.exists(src_img):
            continue

        new_img_name = f"people_{filename}"
        dst_img = os.path.join(OUT_DIR, "images", out_split, new_img_name)
        safe_copy(src_img, dst_img)

        img_w = float(g[wcol].iloc[0])
        img_h = float(g[hcol].iloc[0])

        yolo_lines = []
        for _, row in g.iterrows():
            if PEOPLE_ONLY_CLASSNAME and c_col:
                cls_name = str(row[c_col]).strip().lower()
                if cls_name != PEOPLE_ONLY_CLASSNAME:
                    continue

            x1, y1 = float(row[xmin]), float(row[ymin])
            x2, y2 = float(row[xmax]), float(row[ymax])

            cx = ((x1 + x2) / 2.0) / img_w
            cy = ((y1 + y2) / 2.0) / img_h
            bw = (x2 - x1) / img_w
            bh = (y2 - y1) / img_h

            # clamp to [0,1]
            cx = min(max(cx, 0.0), 1.0)
            cy = min(max(cy, 0.0), 1.0)
            bw = min(max(bw, 0.0), 1.0)
            bh = min(max(bh, 0.0), 1.0)

            yolo_lines.append(f"{PERSON_CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        dst_label = os.path.join(
            OUT_DIR, "labels", out_split, os.path.splitext(new_img_name)[0] + ".txt"
        )
        with open(dst_label, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))

print("People conversion complete.")

# =========================
# 2) CONVERT WIDER FACE
# =========================
print("Converting WIDER FACE...")

train_gt = find_existing_path(
    os.path.join(WIDER_SPLIT_DIR, "wider_face_train_bbx_gt.txt"),
    os.path.join(WIDER_SPLIT_DIR, "wider_face_split", "wider_face_train_bbx_gt.txt"),
)
val_gt = find_existing_path(
    os.path.join(WIDER_SPLIT_DIR, "wider_face_val_bbx_gt.txt"),
    os.path.join(WIDER_SPLIT_DIR, "wider_face_split", "wider_face_val_bbx_gt.txt"),
)

train_data = parse_wider(train_gt)
val_data = parse_wider(val_gt)

def convert_wider(mapping: dict[str, list[tuple[float, float, float, float]]], split_name: str) -> None:
    img_root = os.path.join(WIDER_TRAIN_DIR if split_name == "train" else WIDER_VAL_DIR, "images")
    out_split = "train" if split_name == "train" else "val"

    for rel_path, boxes in mapping.items():
        src_img = os.path.join(img_root, rel_path)
        if not os.path.exists(src_img):
            continue

        new_img_name = "wider_" + rel_path.replace("/", "_").replace("\\", "_")
        dst_img = os.path.join(OUT_DIR, "images", out_split, new_img_name)
        safe_copy(src_img, dst_img)

        with Image.open(src_img) as im:
            W, H = im.size

        yolo_lines = []
        for x, y, w, h in boxes:
            cx = (x + w / 2.0) / W
            cy = (y + h / 2.0) / H
            bw = w / W
            bh = h / H

            cx = min(max(cx, 0.0), 1.0)
            cy = min(max(cy, 0.0), 1.0)
            bw = min(max(bw, 0.0), 1.0)
            bh = min(max(bh, 0.0), 1.0)

            yolo_lines.append(f"{FACE_CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        dst_label = os.path.join(OUT_DIR, "labels", out_split, os.path.splitext(new_img_name)[0] + ".txt")
        with open(dst_label, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))

convert_wider(train_data, "train")
convert_wider(val_data, "val")

print("WIDER conversion complete.")
print(" Dataset ready at datasets/combined")

"""
CIS3425 Smart Home Security System
Number Plate Detection & OCR Evaluation Script
Author: [Your Name]

Evaluates the number plate pipeline against the UK Number Plate dataset.
Generates:
  - Detection metrics (precision, recall, F1, mAP)
  - OCR accuracy and confidence distribution
  - Confusion matrix (plate detected / not detected)
  - Sample visualisations of detections

Usage:
  python evaluate_plates.py
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from pathlib import Path
from ultralytics import YOLO
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_fscore_support
)
import easyocr
import re
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
CONFIG = {
    "dataset_path":     r"C:\Users\bmel0\OneDrive - Edge Hill University\Documents\GitHub\Uni\UK Number Plate Recognision.v2i.yolov8",
    "split":            "test",              # Evaluate on test split
    "yolo_model":       "yolov8n.pt",        # Swap for custom plate model if available
    "yolo_confidence":  0.45,
    "iou_threshold":    0.5,                 # IoU threshold for true positive
    "ocr_languages":    ["en"],
    "ocr_confidence":   0.25,
    "output_dir":       "evaluation_results",
    "sample_images":    8,                   # Number of sample images to visualise
}


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def load_yolo_labels(label_path: str, img_w: int, img_h: int) -> list[dict]:
    """Load YOLO format labels and convert to pixel coordinates."""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])
            x1 = int((cx - bw / 2) * img_w)
            y1 = int((cy - bh / 2) * img_h)
            x2 = int((cx + bw / 2) * img_w)
            y2 = int((cy + bh / 2) * img_h)
            boxes.append({"cls": cls, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return boxes


def compute_iou(box_a: dict, box_b: dict) -> float:
    """Compute Intersection over Union between two boxes."""
    xa = max(box_a["x1"], box_b["x1"])
    ya = max(box_a["y1"], box_b["y1"])
    xb = min(box_a["x2"], box_b["x2"])
    yb = min(box_a["y2"], box_b["y2"])

    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (box_a["x2"] - box_a["x1"]) * (box_a["y2"] - box_a["y1"])
    area_b = (box_b["x2"] - box_b["x1"]) * (box_b["y2"] - box_b["y1"])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def clean_ocr_text(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", text).upper().strip()


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def run_evaluation(config: dict):
    os.makedirs(config["output_dir"], exist_ok=True)

    # ── Load models ───────────────────────────────────────────────────────────
    log.info("Loading YOLOv8...")
    yolo = YOLO(config["yolo_model"])

    log.info("Loading EasyOCR...")
    reader = easyocr.Reader(config["ocr_languages"], gpu=True)

    # ── Dataset paths ─────────────────────────────────────────────────────────
    split       = config["split"]
    images_dir  = Path(config["dataset_path"]) / split / "images"
    labels_dir  = Path(config["dataset_path"]) / split / "labels"

    image_paths = sorted(list(images_dir.glob("*.jpg")) +
                         list(images_dir.glob("*.png")))
    log.info(f"Evaluating on {len(image_paths)} images from '{split}' split...")

    # ── Metrics accumulators ──────────────────────────────────────────────────
    true_labels      = []   # 1 = plate present in ground truth
    pred_labels      = []   # 1 = plate detected by model
    ocr_confidences  = []
    iou_scores       = []
    sample_frames    = []

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        # ── Ground truth ──────────────────────────────────────────────────────
        label_path = labels_dir / (img_path.stem + ".txt")
        gt_boxes   = load_yolo_labels(str(label_path), w, h)
        has_gt     = len(gt_boxes) > 0
        true_labels.append(1 if has_gt else 0)

        # ── YOLOv8 inference ──────────────────────────────────────────────────
        results   = yolo(img, verbose=False)[0]
        det_boxes = []

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < config["yolo_confidence"]:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            det_boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                              "conf": conf})

        has_det = len(det_boxes) > 0
        pred_labels.append(1 if has_det else 0)

        # ── IoU scoring ───────────────────────────────────────────────────────
        if has_gt and has_det:
            best_iou = 0.0
            for gt in gt_boxes:
                for det in det_boxes:
                    iou = compute_iou(gt, det)
                    best_iou = max(best_iou, iou)
            iou_scores.append(best_iou)

        # ── OCR on detected regions ───────────────────────────────────────────
        annotated = img.copy()
        for det in det_boxes:
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w, x2); y2 = min(h, y2)
            crop = img[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            # Preprocess crop
            if crop.shape[1] < 100:
                scale = 100 / crop.shape[1]
                crop = cv2.resize(crop,
                                  (int(crop.shape[1]*scale),
                                   int(crop.shape[0]*scale)),
                                  interpolation=cv2.INTER_CUBIC)

            grey  = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            grey  = clahe.apply(grey)
            _, binary = cv2.threshold(grey, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            try:
                ocr_results = reader.readtext(binary)
                for (_, text, conf) in ocr_results:
                    clean = clean_ocr_text(text)
                    if len(clean) >= 4 and conf >= config["ocr_confidence"]:
                        ocr_confidences.append(conf)
                        cv2.rectangle(annotated, (x1, y1), (x2, y2),
                                      (0, 200, 0), 2)
                        cv2.putText(annotated, f"{clean} {conf:.2f}",
                                    (x1, y1 - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (0, 200, 0), 2)
            except Exception:
                pass

        # Draw ground truth boxes in blue
        for gt in gt_boxes:
            cv2.rectangle(annotated,
                          (gt["x1"], gt["y1"]), (gt["x2"], gt["y2"]),
                          (255, 100, 0), 2)

        if len(sample_frames) < config["sample_images"]:
            sample_frames.append((img_path.name, annotated))

    # ═══════════════════════════════════════════════════════════════════════════
    # RESULTS & PLOTS
    # ═══════════════════════════════════════════════════════════════════════════

    log.info("\n" + "="*60)
    log.info("EVALUATION RESULTS")
    log.info("="*60)

    # ── Detection metrics ─────────────────────────────────────────────────────
    print("\nDetection Classification Report:")
    print(classification_report(
        true_labels, pred_labels,
        target_names=["No Plate", "Plate Present"],
        zero_division=0
    ))

    tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(true_labels, pred_labels) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(true_labels, pred_labels) if t == 0 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) \
                if (precision + recall) > 0 else 0
    avg_iou   = np.mean(iou_scores) if iou_scores else 0
    avg_ocr   = np.mean(ocr_confidences) if ocr_confidences else 0

    print(f"Precision:        {precision:.4f}")
    print(f"Recall:           {recall:.4f}")
    print(f"F1 Score:         {f1:.4f}")
    print(f"Mean IoU:         {avg_iou:.4f}")
    print(f"Mean OCR Conf:    {avg_ocr:.4f}")
    print(f"OCR reads:        {len(ocr_confidences)}")

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Plate", "Plate"],
                yticklabels=["No Plate", "Plate"])
    plt.title("Confusion Matrix — Number Plate Detection\n"
              f"YOLOv8 | Confidence threshold={config['yolo_confidence']}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    path = os.path.join(config["output_dir"], "confusion_matrix_plates.png")
    plt.savefig(path, dpi=150)
    log.info(f"Saved: {path}")
    plt.show()

    # ── OCR Confidence Distribution ───────────────────────────────────────────
    if ocr_confidences:
        plt.figure(figsize=(8, 4))
        plt.hist(ocr_confidences, bins=20, color="#4C72B0",
                 edgecolor="white", alpha=0.85)
        plt.axvline(x=np.mean(ocr_confidences), color="red",
                    linestyle="--", label=f"Mean = {np.mean(ocr_confidences):.3f}")
        plt.axvline(x=config["ocr_confidence"], color="orange",
                    linestyle="--",
                    label=f"Threshold = {config['ocr_confidence']}")
        plt.xlabel("OCR Confidence Score")
        plt.ylabel("Frequency")
        plt.title("EasyOCR Confidence Distribution — Number Plate Reads")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        path = os.path.join(config["output_dir"], "ocr_confidence_dist.png")
        plt.savefig(path, dpi=150)
        log.info(f"Saved: {path}")
        plt.show()

    # ── IoU Distribution ──────────────────────────────────────────────────────
    if iou_scores:
        plt.figure(figsize=(8, 4))
        plt.hist(iou_scores, bins=15, color="#55A868",
                 edgecolor="white", alpha=0.85)
        plt.axvline(x=config["iou_threshold"], color="red",
                    linestyle="--",
                    label=f"IoU threshold = {config['iou_threshold']}")
        plt.axvline(x=np.mean(iou_scores), color="orange",
                    linestyle="--",
                    label=f"Mean IoU = {np.mean(iou_scores):.3f}")
        plt.xlabel("IoU Score")
        plt.ylabel("Frequency")
        plt.title("IoU Distribution — Predicted vs Ground Truth Boxes")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        path = os.path.join(config["output_dir"], "iou_distribution.png")
        plt.savefig(path, dpi=150)
        log.info(f"Saved: {path}")
        plt.show()

    # ── Sample Detections Grid ────────────────────────────────────────────────
    if sample_frames:
        n     = len(sample_frames)
        cols  = 4
        rows  = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols,
                                 figsize=(cols * 4, rows * 3))
        axes = axes.flatten() if n > 1 else [axes]

        for i, (name, frame) in enumerate(sample_frames):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            axes[i].imshow(rgb)
            axes[i].set_title(name[:20], fontsize=7)
            axes[i].axis("off")

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.suptitle("Sample Detections  |  Green=Predicted  Blue=Ground Truth",
                     fontsize=10)
        plt.tight_layout()
        path = os.path.join(config["output_dir"], "sample_detections.png")
        plt.savefig(path, dpi=150)
        log.info(f"Saved: {path}")
        plt.show()

    log.info("\nAll evaluation plots saved to: " + config["output_dir"])
    log.info("These can be used directly in your CIS3425 final report.")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_evaluation(CONFIG)
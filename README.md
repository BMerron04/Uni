# VertexGuard — Smart Home Security System

A local, privacy-first smart home security system using computer vision and machine learning for real-time face recognition and number plate detection.

## Features

- Real-time face detection and recognition (known vs unknown)
- Number plate detection and OCR
- Live web dashboard with sidebar navigation
- Unknown face/plate enrolment via browser
- Multi-camera support
- Continuous video recording
- SQLite event logging
- Login system with admin/viewer roles
- Settings panel (cameras, thresholds, users)
- Developer performance panel

## Tech Stack

- **YOLOv8** — object detection
- **dlib / face_recognition** — face encoding and recognition
- **EasyOCR** — number plate text extraction
- **Flask** — web dashboard
- **OpenCV** — video capture and processing
- **SQLite** — local event logging

## Setup

### 1. Create conda environment

```bash
conda create -n security_project python=3.11 -y
conda activate security_project
```

### 2. Install dependencies

```bash
conda install -c conda-forge dlib -y
pip install face_recognition ultralytics easyocr flask opencv-python numpy scikit-learn matplotlib seaborn
```

### 3. Add your detection models

Place your model weights in the project root:
- `yolo26n.pt` — face/person detection model
- `plate_detector.pt` — number plate detection model

### 4. Add known faces

```
known_faces/
    PersonName/
        photo1.jpg
        photo2.jpg
```

### 5. Run

```bash
python main_server.py
```

Open `http://localhost:5000` in your browser.

Default login: `admin` / `admin123` — **change this immediately in Settings.**

## Project Structure

```
├── main_server.py          # Main server (Flask + detection threads)
├── facial_recognition_pipeline.py   # Standalone face recognition
├── number_plate_pipeline.py         # Standalone plate detection
├── evaluate_plates.py      # Plate detection evaluation
├── curate_dataset.py       # Dataset curation and deduplication
├── convert_all_datasets.py # Dataset format conversion
└── README.md
```

## Runtime folders (auto-created, not committed)

```
known_faces/     # Enrolled face photos
logged_frames/   # Detection event snapshots
recordings/      # Continuous video recordings
unknowns/        # Unknown face/plate crops pending enrolment
security.db      # SQLite database
config.json      # System configuration
```

## Notes

- All processing is local — no cloud dependency
- Recordings split into 10-minute chunks automatically
- Unknown faces appear in dashboard for manual enrolment
- Run `python curate_dataset.py --auto` periodically to improve recognition accuracy

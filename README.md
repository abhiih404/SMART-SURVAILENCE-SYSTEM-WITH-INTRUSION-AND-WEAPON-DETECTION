# 🔍 Smart Surveillance System
### YOLOv8 · LBPH Face Recognition · SVM Classifier · Telegram Alerts

A GPU-accelerated AI surveillance system with human detection, face recognition,
loitering detection, weapon detection, and real-time threat classification.

---

## 📁 Project Structure

```
smart_surveillance/
├── main.py                        ← Entry point (run this)
├── surveillance_engine.py         ← Core pipeline
├── capture_faces.py               ← Step 1: capture known persons
├── train_model.py                 ← Step 2: train LBPH + SVM
├── setup_telegram.py              ← Step 3: configure Telegram alerts
├── download_weapon_model.py       ← Step 4: get weapon model
├── train_custom_weapon_model.py   ← Optional: fine-tune weapon model
├── requirements.txt
├── .env                           ← Your configuration (edit this)
│
├── config/
│   └── settings.py                ← Loads .env into Python constants
│
├── utils/
│   ├── preprocessing.py           ← Gaussian blur + NMS
│   ├── telegram_alert.py          ← Async Telegram sender
│   └── logger.py                  ← Rotating file + console logger
│
├── known_faces/                   ← Created automatically
│   └── <PersonName>/              ← One folder per known person
│       ├── Person_0000.jpg
│       └── ...
│
├── models/                        ← Saved ML models
│   ├── yolov8n.pt                 ← Auto-downloaded
│   ├── lbph_face_model.xml        ← After training
│   ├── svm_classifier.pkl         ← After training
│   └── label_map.pkl              ← After training
│
├── weapons_model/
│   └── weapon_yolov8.pt           ← Download or train
│
├── alerts/                        ← Alert images + recordings saved here
└── logs/
    └── surveillance.log
```

---

## ⚙️ System Requirements

| Component | Requirement |
|-----------|-------------|
| GPU | NVIDIA GeForce RTX 2050 (4 GB VRAM) ✅ |
| CUDA | 11.8 or 12.x |
| cuDNN | 8.x |
| Python | 3.10+ |
| OS | Windows 10/11 or Ubuntu 20.04+ |
| RAM | 8 GB+ recommended |

---

## 🚀 Installation

### 1. Clone / download the project
```bash
cd smart_surveillance
```

### 2. Create and activate a virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python -m venv venv
source venv/bin/activate
```

### 3. Install PyTorch with CUDA support (RTX 2050 → CUDA 11.8)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 4. Install remaining dependencies
```bash
pip install -r requirements.txt
```

### 5. Verify GPU is detected
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
# Expected: CUDA: True | NVIDIA GeForce RTX 2050
```

---

## 📋 Step-by-Step Setup

### Step 1 – Capture Known Persons

Capture face images for everyone you want the system to recognise
(family members, neighbours, etc.):

```bash
# Capture 30 images of "John Doe" from webcam 0
python capture_faces.py --name "John Doe" --samples 30 --source 0

# Capture from a video file
python capture_faces.py --name "Jane Smith" --samples 40 --source /path/to/video.mp4
```

Controls inside the capture window:
- **SPACE** → capture current frame
- **Q** → quit

> 💡 **Tips for good captures:**
> - Face the camera directly under good lighting
> - Capture various expressions and slight angles
> - Aim for at least 30 samples per person
> - Avoid motion blur

Repeat for each known person. Images are saved to `known_faces/<PersonName>/`.

---

### Step 2 – Train the Models

```bash
python train_model.py
```

This trains:
1. **LBPH Face Recogniser** on your captured images
2. **SVM Threat Classifier** (feature-based: known/unknown, loitering, weapon, confidence)

Output:
```
models/lbph_face_model.xml
models/svm_classifier.pkl
models/label_map.pkl
```

Retrain whenever you add new persons.

---

### Step 3 – Configure Telegram Alerts

#### 3a. Create a Telegram Bot
1. Open Telegram → search for **@BotFather**
2. Send `/newbot` and follow the prompts
3. Copy the **API Token** (looks like `123456789:ABCdef...`)

#### 3b. Run the setup helper
```bash
python setup_telegram.py
```

This interactively saves your token + chat ID to `.env` and sends a test message.

#### Manual alternative – edit `.env` directly:
```env
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrSTUvwxYZ
TELEGRAM_CHAT_ID=987654321
```

---

### Step 4 – Weapon Detection Model

#### Option A – Download pre-trained model
```bash
python download_weapon_model.py
```

#### Option B – Train your own (recommended for best accuracy)
```bash
pip install roboflow
python train_custom_weapon_model.py --api-key YOUR_ROBOFLOW_API_KEY
```
Get a free API key at https://app.roboflow.com

#### Option C – Manual download
1. Go to https://universe.roboflow.com
2. Search for "weapon detection YOLOv8"
3. Download YOLOv8 format weights
4. Place the `.pt` file at `weapons_model/weapon_yolov8.pt`

> **Note:** The system runs without weapon detection (disabled automatically if model is missing).

---

### Step 5 – Configure Settings (`.env`)

```env
# Telegram
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id

# Loitering: alert after person stands still for this many seconds
LOITERING_TIME_THRESHOLD=10

# YOLO detection confidence (0.0–1.0)
CONFIDENCE_THRESHOLD=0.55

# LBPH: lower = stricter matching (0–100, default 70)
FACE_RECOGNITION_THRESHOLD=70

# Video source: 0 for webcam, or file path
VIDEO_SOURCE=0

# Use GPU
USE_GPU=True

# Weapon detection confidence
WEAPON_CONFIDENCE_THRESHOLD=0.50
```

---

## ▶️ Running the System

```bash
# Use settings from .env
python main.py

# Override video source
python main.py --source 0              # webcam
python main.py --source /path/to/video.mp4

# Also record the annotated output
python main.py --source 0 --record
```

### Keyboard controls during runtime

| Key | Action |
|-----|--------|
| Q | Quit |
| P | Pause / Resume |
| S | Take screenshot (saved to `alerts/`) |

---

## 🔄 Pipeline Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Input Frame                         │
└──────────────────────────┬──────────────────────────────┘
                           │
                    Gaussian Blur (3×3)
                           │
              ┌────────────┴────────────┐
              │                         │
    YOLOv8 Person Detection      YOLOv8 Weapon Detection
    + ByteTrack Tracking         (full-frame scan)
    + Custom NMS                        │
              │                  weapon_flag (bool)
    ┌─────────┴──────────┐
    │   Per-person ROI   │
    └─────────┬──────────┘
              │
    Haar Face Detection (in ROI)
              │
    LBPH Face Recognition
    → name, confidence
    → is_unknown (bool)
              │
    Loitering Tracker
    (centroid + time threshold)
    → loitering (bool)
              │
    ┌─────────┴──────────────────────────────┐
    │  SVM Classifier                        │
    │  Features: [is_unknown, loitering,     │
    │             weapon_flag, face_conf]    │
    │  Output: NORMAL / SUSPICIOUS / HIGH    │
    └─────────┬──────────────────────────────┘
              │
    ┌─────────┴──────────┐
    │  NORMAL            │ → no alert
    │  SUSPICIOUS        │ → Telegram alert + photo
    │  HIGH              │ → Telegram alert + photo
    └────────────────────┘
```

---

## 🎯 Threat Classification Logic

| Condition | Classification |
|-----------|---------------|
| Known person, no weapon | NORMAL |
| Unknown + loitering > threshold | SUSPICIOUS |
| Unknown, short stay | SUSPICIOUS |
| Any person + weapon detected | HIGH |

> Alerts are sent with a **60-second cooldown** per person to prevent spam.

---

## 🛠️ Tuning Guide

### Face recognition too strict (rejecting known persons)?
```env
FACE_RECOGNITION_THRESHOLD=85   # increase (more lenient)
```

### Too many false positives (strangers marked as known)?
```env
FACE_RECOGNITION_THRESHOLD=55   # decrease (more strict)
```

### Loitering triggers too fast?
```env
LOITERING_TIME_THRESHOLD=20     # increase to 20 seconds
```

### Missing persons in crowd?
```env
CONFIDENCE_THRESHOLD=0.40       # lower detection threshold
```

---

## 🔧 Troubleshooting

**`CUDA out of memory`**
→ RTX 2050 has 4 GB VRAM. Reduce `--imgsz 416` in weapon training, or use `batch=8`.

**`lbph_face_model.xml not found`**
→ Run `python train_model.py` after capturing at least one person.

**`Face recognition always returns Unknown`**
→ Check lighting, ensure faces are at least 60×60 px. Lower `FACE_RECOGNITION_THRESHOLD`.

**Telegram alerts not arriving**
→ Run `python setup_telegram.py` again, verify token and chat ID in `.env`.

**Weapon model not found**
→ Run `python download_weapon_model.py` or place `weapon_yolov8.pt` in `weapons_model/`.

---

## 📦 Key Libraries

| Library | Purpose |
|---------|---------|
| `ultralytics` | YOLOv8 person + weapon detection |
| `opencv-contrib-python` | LBPH face recognition, Haar cascade |
| `scikit-learn` | SVM threat classifier |
| `python-telegram-bot` | Async Telegram alerts |
| `torch + CUDA` | GPU acceleration |

---


## 🔒 Privacy Notice

This system captures and processes faces. Ensure you comply with local
privacy laws before deploying in any space. Only use on property you own
or have explicit permission to monitor.


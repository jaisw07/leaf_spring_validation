# 🍃 Leaf Spring Assembly Verification System

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg?style=flat-square&logo=python&logoColor=white)](#)
[![FastAPI](https://img.shields.io/badge/FastAPI-v0.100+-009688.svg?style=flat-square&logo=fastapi&logoColor=white)](#)
[![YOLOv8](https://img.shields.io/badge/YOLO-26n%20Model-red.svg?style=flat-square&logo=ultralytics&logoColor=white)](#)
[![SQLite](https://img.shields.io/badge/SQLite-Database-003B57.svg?style=flat-square&logo=sqlite&logoColor=white)](#)
[![OpenCV](https://img.shields.io/badge/OpenCV-v4.8+-green.svg?style=flat-square&logo=opencv&logoColor=white)](#)

An automated, real-time vision-based verification system designed to validate the correct installation of leaf spring models on a moving vehicle assembly line. 

Using **YOLO26n** object detection, homography-based perspective rectification, and a custom geometric reference tracker, the system automatically checks for left-to-right part matching and records historical run results.

---

## 🛠️ Tech Stack & Key Technologies

The system is built on a modular, asynchronous architecture combining deep learning models, local area network interfaces, database persistence, and a real-time reactive dashboard.

### 👁️ Computer Vision & AI
* **🤖 YOLO26n (End-to-End NMS-Free Nano)**: Custom trained (SGD optimizer, 100 epochs, resolution `576x1088`) for high-precision bounding box detection of reference rods and spring components.
* **📸 OpenCV**: Handles video capturing (RTSP IP camera and local MP4 streams), homography matrix operations, and live drawing overlays.
* **🔥 PyTorch**: Leveraged for high-frequency neural network inference (average execution speed: ~15.5 ms/image on RTX 4050).

### ⚙️ Backend & Database
* **⚡ FastAPI**: Asynchronous Python web framework hosting the REST APIs and managing WebSocket channels.
* **🚀 Uvicorn**: High-performance ASGI server for hosting the application.
* **🛡️ Pydantic**: Validates REST request payloads and structural database models.
* **🗄️ SQLite**: Database engine serving as the Single Source of Truth for side pickup queues and vehicle run records.

### 🖥️ Web Dashboard UI
* **🎨 Glassmorphism Dark Theme**: Modern, responsive grid-based styles featuring subtle animations, micro-interactions, and status indicators.
* **🕸️ WebSockets**: Stream live perspective-rectified canvas updates and queue state payloads from server to browser.
* **📦 Vanilla JS & HTML5**: Renders dynamic canvas feeds, handles stream logic, and formats real-time dashboard events.

### 🧪 Testing & Validation
* **✅ pytest**: Validates database queries, REST routes, client WebSocket bindings, and tracker client logic.
* **🌐 httpx**: Enables asynchronous API routing tests during testing.

---

## 🌟 Key System Features

* 🎥 **Real-Time Vision Pipeline**: Processes RTSP IP camera streams or local video files at target frame rates.
* 📐 **Perspective Normalization**: Rectifies camera-angle distortions utilizing predefined homography bounds mapping raw video frames into stabilized grid spaces.
* 🎯 **Unified Reference Point Tracker**: Dynamically tracks torque rods and springs as spatial reference points. Employs offset projection mathematics to estimate front slot locations under occlusion conditions.
* 📥 **FIFO Queue Verification**: Stores barcode/RFID pickup events from side/rear cameras in an SQLite database and pulls them in a First-In-First-Out (FIFO) manner as slots fill up.
* ⚖️ **Left-to-Right Part Matching**: Instantly flags matching issues between Front-Left/Front-Right (FL/FR) and Rear-Left/Rear-Right (RL/RR) spring pairs.
* 📊 **Web Dashboard**: An interactive, responsive glassmorphism dark-themed UI featuring live camera frame overlays, real-time queue states, hardware pickup event simulators, and detailed verification history log tables.

---

## 📊 Architecture & Pipeline Flow

The system coordinates side-camera events and top-camera vision analysis via a centralized FastAPI server:

```
Side Cameras (Pickup Events)        Top Camera (Video Stream Feed)
           │                                      │
           ▼ (POST /api/event)                    ▼ (OpenCV Stream Ingestion)
   ┌───────────────┐                    ┌─────────────────────────┐
   │ FastAPI Server│                    │ Perspective Warp        │
   │ (SQLite DB)   │                    │ (Normalizes Lane Space) │
   └───────┬───────┘                    └────────────┬────────────┘
           │                                         │
           │ (REST /api/queue/pop)                   ▼ (Inference)
           │ ◄───────────────────────── ┌─────────────────────────┐
           │                            │ YOLO26n Object Detector │
           │                            │ (Finds Springs/Rods)    │
           │                            └────────────┬────────────┘
           │                                         │
           │                                         ▼ (State Machine)
           │                            ┌─────────────────────────┐
           │                            │ Geometry Tracker        │
           │                            │ (Project Slots & Match) │
           │                            └────────────┬────────────┘
           │                                         │
           └───────────────────┬─────────────────────┘
                               │ (WebSocket updates /ws/client)
                               ▼
                    ┌─────────────────────┐
                    │ Web Dashboard UI    │
                    │ (Live status & logs)│
                    └─────────────────────┘
```

---

## 📂 Repository Directory Structure

* 📂 [context/](context)
  * 📄 [context.md](context/context.md) — Comprehensive technical reference, coordinates, and YOLO validation metrics.
  * 📄 [usage_instructions.md](context/usage_instructions.md) — Step-by-step runner configuration and camera integration.
  * 📄 [api_instructions.md](context/api_instructions.md) — Endpoint specifications, JSON models, and networking guidelines.
* 📂 [src/](src)
  * ⚙️ [server.py](src/server.py) — FastAPI server, endpoints, and WebSocket broadcasting hubs.
  * 🗄️ [database.py](src/database.py) — SQLite data-access layer for queuing and run storage.
  * 🔍 [tracker.py](src/tracker.py) — Core geometric tracking and left-right matching logic.
  * 🚀 [run_tracker.py](src/run_tracker.py) — Async production vision-capture loop with reconnection recovery.
  * 📐 [calibrate_offsets.py](src/calibrate_offsets.py) — Utility to calculate static slot project vectors.
  * 🖥️ [static/index.html](src/static/index.html) — Glassmorphism Dashboard UI HTML/CSS/JS.
* 📂 [tests/](tests)
  * Unit tests validating database APIs, WebSocket states, and model parsing behaviors.
* 📂 [mydata/](mydata)
  * Configuration metadata (homography settings, SQLite `system.db`, and pre-recorded MP4 validation videos).

---

## 🔧 Prerequisites & Installation

Ensure you have a working Python and Conda distribution. Activate the targeted environment containing PyTorch, OpenCV, and FastAPI dependencies:

```powershell
# Activate the Conda environment
conda activate dump
```

---

## ⚡ Quick Start & Execution

To deploy the verification system on your local system or LAN, open two terminals:

### Step 1: Start the FastAPI Server
Launch the server. It automatically runs migrations to initialize `mydata/system.db`.
```powershell
# Bind to 0.0.0.0 to enable access from other IP addresses on the LAN
conda run -n dump uvicorn src.server:app --host 0.0.0.0 --port 8000
```

### Step 2: Run the Vision Pipeline
Specify the video source (RTSP link or local file) and run the tracker module:
```powershell
# Set video target (defaults to mydata/videos/chassis1.mp4 if empty)
$env:VIDEO_SOURCE = "mydata/videos/chassis1.mp4"

# Run async tracker client
conda run -n dump python -u -m src.run_tracker
```

### Step 3: Access the Web Dashboard
* **Local Machine**: Open [http://127.0.0.1:8000](http://127.0.0.1:8000)
* **Remote Machine (LAN)**: Open `http://<server-ip-address>:8000` (e.g. `http://192.168.1.50:8000`)

> [!IMPORTANT]
> Ensure incoming TCP connections on port `8000` are permitted by the host's firewall rules when hosting the server on a LAN.

---

## 🧪 Running Automated Tests

The repository contains a full suite of automated unit and integration tests. Run tests via `pytest`:

```powershell
# Run the test suite
conda run -n dump pytest
```

---

## 📖 Documentation & Integration Guides

For deep-dive topics, consult the files in the `context/` directory:
* 📐 **System Math & Tracker Logic**: Consult [context.md](context/context.md) to understand homography warp matrices, standard deviation offsets for slot coordinates, and YOLO26n validation performance benchmarks.
* 🎥 **Camera Streaming & RTSP Reconnections**: Consult [usage_instructions.md](context/usage_instructions.md) for custom RTSP URL configuration, H.264/H.265 codec tuning, and performance troubleshooting.
* 🔌 **External API Integration (Sensors, RFID, Cameras)**: Consult [api_instructions.md](context/api_instructions.md) for structured endpoint payloads, Windows Defender Firewall rule setup, and programming code snippets in cURL, Python, Node.js, and PowerShell.

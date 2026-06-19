# Leaf Spring Verification System - Usage & RTSP Integration Guide

This guide provides step-by-step instructions to configure and run the Leaf Spring Assembly Verification system, connect it to a live RTSP IP camera feed, and view the web-based visualization dashboard.

---

## 1. Prerequisites & Environment Setup

Ensure you are working in the active Conda environment containing all required deep learning and computer vision dependencies:

```powershell
# Activate the Conda environment
conda activate dump
```

The system requires two concurrent services:
1. **FastAPI Web Backend & DB Server**: Hosts rest endpoints, stores queues and verification history, and manages WebSocket broadcasts.
2. **Vision Pipeline & Tracker**: Captures the camera feed (RTSP or MP4 file), executes YOLO26n inferences, rectifies perspective, tracks slots, and streams results.

---

## 2. Configuration & Video Source Selection

The vision tracker client is configured dynamically using environment variables. You can specify a local demo video file to run or connect a live RTSP IP camera feed.

### Selecting a Demo Video File
The project includes three pre-recorded chassis assembly video files in the `mydata/videos/` directory:
1. `mydata/videos/chassis1.mp4` (Default)
2. `mydata/videos/chassis2.mp4`
3. `mydata/videos/chassis3.mp4`

By default, if the `VIDEO_SOURCE` environment variable is not defined, the tracker will stream `chassis1.mp4`. To run the system with a different demo video:

#### In Windows PowerShell:
```powershell
$env:VIDEO_SOURCE = "mydata/videos/chassis2.mp4"
```

#### In Windows Command Prompt (CMD):
```cmd
set VIDEO_SOURCE=mydata/videos/chassis2.mp4
```

---

### Connecting a Live RTSP IP Camera Feed
If you want to run the system against a live camera instead of the demo files:

### Step 1: Obtain the RTSP URL
Identify your IP camera's RTSP stream path. The standard format for security cameras is:
```text
rtsp://[username]:[password]@[camera-ip-address]:[port]/[stream-profile-path]
```
*Common examples*:
* **Hikvision**: `rtsp://admin:12345@192.168.1.64:554/Streaming/Channels/101`
* **Dahua**: `rtsp://admin:admin123@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0`
* **Generic ONVIF**: `rtsp://192.168.1.100:554/onvif1`

### Step 2: Configure Environment Variables
Set the `VIDEO_SOURCE` environment variable in your terminal before running the script.

#### In Windows PowerShell (Recommended):
```powershell
# Set RTSP stream source
$env:VIDEO_SOURCE = "rtsp://admin:password@192.168.1.100:554/stream1"

# (Optional) If FastAPI server is on a different machine, configure URLs:
$env:API_URL = "http://192.168.1.50:8000"
$env:WS_URL = "ws://192.168.1.50:8000/api/ws/tracker"
```

#### In Windows Command Prompt (CMD):
```cmd
:: Set RTSP stream source
set VIDEO_SOURCE=rtsp://admin:password@192.168.1.100:554/stream1

:: (Optional) Configure remote backend endpoints:
set API_URL=http://192.168.1.50:8000
set WS_URL=ws://192.168.1.50:8000/api/ws/tracker
```

---

## 3. Running the System

Follow this startup sequence to launch the services:

### Step 1: Start the FastAPI Server
Run the uvicorn server in a terminal window. This initializes the SQLite database (`mydata/system.db`) and opens connection sockets:

```powershell
conda run -n dump uvicorn src.server:app --host 127.0.0.1 --port 8000
```
*If clients on your Local Area Network (LAN) need to view the dashboard, bind the host to `0.0.0.0` instead*:
```powershell
conda run -n dump uvicorn src.server:app --host 0.0.0.0 --port 8000
```

### Step 2: Start the Vision Pipeline
In a separate terminal window, set your `VIDEO_SOURCE` (RTSP stream or MP4 test video) and run the pipeline runner:

```powershell
# In PowerShell:
$env:VIDEO_SOURCE = "rtsp://admin:password@192.168.1.100:554/stream1"
conda run -n dump python -u -m src.run_tracker
```

---

## 4. Accessing and Using the Web Dashboard

Once both services are running:

1. Open your web browser (Chrome, Edge, or Firefox).
2. Navigate to: **[http://127.0.0.1:8000](http://127.0.0.1:8000)** (or `http://[server-ip-address]:8000` if hosting on your LAN).

### Core Features & Interactive Testing:
* **Live Camera View**: Shows the perspective-rectified lane boundaries frame overlayed with active ROIs, detected reference points, slot occupancy status labels, and verification results in real-time.
* **Side Camera Pickup Queues**: Shows left-to-right FIFO queues.
* **Hardware Event Simulator**: Allows you to simulate physical side camera barcode/RFID pickup events. Choose the side (Left/Right) and Model (e.g. `GREEN_A`, `RED_A`) and click **PICKUP EVENT**. These queue up on the server database. When a slot is filled in the stream, the tracker dequeues the models and validates them.
* **Reset Stream Button**: Click **RESET STREAM** at the bottom of the simulator card. This instantly:
  1. Clears both left/right camera queues on the server database.
  2. Tells the vision tracker client to reset its Geometry Tracker state machine (clears slot assignments, reference points, and velocities).
  3. Rewinds the stream video back to frame `0` **only if playing a local MP4 file**.
  * **RTSP Note**: On live RTSP streams, video seek/rewind is not possible. The reset clears queues and tracker state, but the stream continues from the current live frame.
* **Queue Preservation**: When a demo video file reaches the end and auto-loops, only the tracker state is reset — queued pickup events are preserved. Queues are only cleared by explicit user actions (Reset Stream or Clear Queues buttons).
* **Verification History Log**: Shows past completed runs, listing timestamps (IST), models detected in FL/FR/RL/RR slots, and overall pass/fail status.

---

## 5. Troubleshooting RTSP Connections

* **Video Frame Capture Fails**:
  * Verify the camera IP is reachable via ping.
  * Ensure port `554` (default RTSP port) is open and not blocked by local firewalls.
  * Open the RTSP URL in a media player like VLC to verify stream validity.
* **Codec Compatibility**:
  * OpenCV handles H.264 encoding natively. If your IP camera uses H.265 (HEVC), you may need to configure the camera's sub-stream/main-stream encoding to **H.264** in its system settings dashboard for compatibility with default OpenCV builds.
* **Network Latency & Buffering**:
  * If the video stream drifts or suffers from high latency, you can adjust the `TARGET_FPS` environment variable (defaults to `5` frames processed per second) to throttle processing load:
    ```powershell
    $env:TARGET_FPS = "3"
    ```

---

## 6. Architecture Notes

* **Queue Source of Truth**: The server SQLite database is the single source of truth for side-camera pickup queues. The web dashboard renders queue state exclusively from server-broadcast WebSocket messages. The vision tracker pops queue items via the server REST API during slot occupancy transitions.
* **Async Pipeline**: The vision pipeline (`run_tracker.py`) uses an async event loop. Tracker state updates and HTTP calls to the server API run synchronously within the main coroutine since the tracker object is not thread-safe.
* **RTSP Reconnection**: If an RTSP stream disconnects, the pipeline releases the old `VideoCapture` handle and creates a new one after a 2-second backoff, preventing socket/handle leaks.

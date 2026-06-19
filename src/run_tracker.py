import os
import sys
import json
import time
import base64
import asyncio
import cv2
import numpy as np
import websockets
from ultralytics import YOLO
from src.tracker import GeometryTracker

# Configuration defaults
API_URL = os.getenv("API_URL", "http://localhost:8000")
WS_URL = os.getenv("WS_URL", "ws://localhost:8000/api/ws/tracker")
VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", "mydata/videos/chassis1.mp4")
HOMOGRAPHY_PATH = os.getenv("HOMOGRAPHY_PATH", "mydata/metadata/homography.npy")
CONFIG_PATH = os.getenv("WARP_CONFIG_PATH", "mydata/metadata/warp_config.json")
MODEL_PATH = os.getenv("MODEL_PATH", "runs/detect/yolo/runs/train/weights/best.pt")
TARGET_FPS = int(os.getenv("TARGET_FPS", "5"))

async def listen_for_commands(ws, cap, tracker):
    try:
        async for message in ws:
            try:
                data = json.loads(message)
                if data.get("command") == "reset":
                    print("\n[VISION SYSTEM] Received reset command from server. Rewinding video capture...")
                    if not VIDEO_SOURCE.startswith("rtsp://"):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    tracker.reset()
            except Exception as e:
                print(f"Error parsing message in listener: {e}")
    except Exception as e:
        print(f"WebSocket command listener error: {e}")

async def stream_tracker():
    print("--- STARTING VISION TRACKER PIPELINE ---")
    print(f"Video Source: {VIDEO_SOURCE}")
    print(f"API URL: {API_URL}")
    print(f"WS URL: {WS_URL}")

    # Check dependencies
    if not os.path.exists(HOMOGRAPHY_PATH) or not os.path.exists(CONFIG_PATH):
        print("Error: Calibration metadata missing. Run warp_test.py and calibrate_offsets.py first.")
        sys.exit(1)
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model weights not found at {MODEL_PATH}.")
        sys.exit(1)

    # Load calibration & model
    H = np.load(HOMOGRAPHY_PATH)
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    lane_width = config["lane_width"]
    lane_height = config["lane_height"]

    model = YOLO(MODEL_PATH)
    tracker = GeometryTracker(api_url=API_URL)

    # Validate video source (if file, verify path)
    if not VIDEO_SOURCE.startswith("rtsp://") and not os.path.exists(VIDEO_SOURCE):
        print(f"Error: Video file not found at {VIDEO_SOURCE}")
        sys.exit(1)

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Error: Failed to open video source {VIDEO_SOURCE}")
        sys.exit(1)

    # Frame skipping calculation to match target FPS
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps <= 0 or np.isnan(native_fps):
        native_fps = 25.0
    frame_step = max(1, int(native_fps / TARGET_FPS))
    frame_delay = 1.0 / TARGET_FPS

    print(f"Native FPS: {native_fps} | Streaming Target FPS: {TARGET_FPS} (Process 1 in every {frame_step} frames)")

    frame_idx = 0

    # Main control loop with websocket automatic reconnection
    while True:
        try:
            print(f"Connecting to WebSocket: {WS_URL}...")
            async with websockets.connect(WS_URL) as ws:
                print("WebSocket connected successfully!")
                
                # Fetch initial queues once connected
                tracker.sync_queues()

                # Start command listener task
                listener_task = asyncio.create_task(listen_for_commands(ws, cap, tracker))

                while True:
                    start_time = time.time()
                    ret, frame = cap.read()
                    if not ret:
                        # If video file, loop it for demo purposes. If RTSP, wait and retry.
                        if not VIDEO_SOURCE.startswith("rtsp://"):
                            print("Video ended. Rewinding to start.")
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            tracker.reset(clear_queues=False)
                            continue
                        else:
                            print("RTSP stream disconnected. Retrying capture...")
                            await asyncio.sleep(2.0)
                            cap.release()
                            cap = cv2.VideoCapture(VIDEO_SOURCE)
                            continue

                    frame_idx += 1
                    if (frame_idx - 1) % frame_step != 0:
                        continue

                    # Warp perspective
                    warped = cv2.warpPerspective(frame, H, (lane_width, lane_height))

                    # Run YOLO prediction
                    preds = model.predict(source=warped, conf=0.5, verbose=False)
                    boxes = preds[0].boxes

                    detections = []
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            cls = int(box.cls[0].item())
                            conf = box.conf[0].item()
                            detections.append({
                                "class_id": cls,
                                "bbox": [x1, y1, x2, y2],
                                "conf": conf
                            })

                    # Update tracker
                    state = tracker.update(detections, lane_width, lane_height)

                    # Draw overlays
                    annotated = warped.copy()
                    if state["active"]:
                        # Draw tracked references
                        if state["ref_L"] is not None:
                            cv2.circle(annotated, tuple(map(int, state["ref_L"])), 8, (0, 0, 255), -1)
                        if state["ref_R"] is not None:
                            cv2.circle(annotated, tuple(map(int, state["ref_R"])), 8, (255, 0, 255), -1)

                        # Draw slots and ROIs
                        for label, slot in state["slots"].items():
                            if slot["center"] is not None:
                                cx, cy = map(int, slot["center"])
                                rx1, ry1 = cx - tracker.roi_w, cy - tracker.roi_h
                                rx2, ry2 = cx + tracker.roi_w, cy + tracker.roi_h

                                if not slot["occupied"]:
                                    color = (255, 255, 0) # Cyan
                                    text = f"{label}: EMPTY"
                                else:
                                    match_res = state["front_match"] if label.startswith("F") else state["rear_match"]
                                    if match_res == "PASS":
                                        color = (0, 255, 0)
                                    elif match_res == "FAIL":
                                        color = (0, 0, 255)
                                    else:
                                        color = (0, 255, 255)
                                    text = f"{label}: {slot['model']}"

                                cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), color, 2)
                                cv2.circle(annotated, (cx, cy), 4, color, -1)
                                cv2.putText(annotated, text, (rx1, ry1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

                        # Status banner
                        status = state["status"]
                        banner_color = (0, 255, 0) if status == "PASS" else ((0, 0, 255) if status == "FAIL" else (0, 255, 255))
                        cv2.rectangle(annotated, (0, 0), (lane_width, 60), banner_color, -1)
                        cv2.putText(annotated, f"VEHICLE DETECTED | STATUS: {status}", (20, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                    else:
                        cv2.rectangle(annotated, (0, 0), (lane_width, 60), (128, 128, 128), -1)
                        cv2.putText(annotated, "WAITING FOR VEHICLE...", (20, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

                    # Compress frame to JPEG and Base64 encode
                    _, buffer = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                    frame_data = f"data:image/jpeg;base64,{jpg_as_text}"

                    # Send payload to server
                    payload = {
                        "state": state,
                        "frame": frame_data
                    }
                    await ws.send(json.dumps(payload))

                    # Sleep to maintain target FPS
                    elapsed = time.time() - start_time
                    sleep_time = max(0.01, frame_delay - elapsed)
                    await asyncio.sleep(sleep_time)
        except (websockets.exceptions.ConnectionClosed, OSError) as e:
            print(f"WebSocket error: {e}. Reconnecting in 3 seconds...")
            if 'listener_task' in locals() and listener_task:
                listener_task.cancel()
            await asyncio.sleep(3.0)
        except Exception as e:
            print(f"Unexpected error: {e}. Reconnecting in 3 seconds...")
            if 'listener_task' in locals() and listener_task:
                listener_task.cancel()
            await asyncio.sleep(3.0)

if __name__ == "__main__":
    try:
        asyncio.run(stream_tracker())
    except KeyboardInterrupt:
        print("\nShutdown vision tracker stream.")

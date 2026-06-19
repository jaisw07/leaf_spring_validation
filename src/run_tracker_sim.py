import os
import json
import argparse
import cv2
import numpy as np
from ultralytics import YOLO
from tracker import GeometryTracker

def main():
    parser = argparse.ArgumentParser(description="Simulate Geometry Tracker on Chassis Video Feed")
    parser.add_argument("--video", type=str, default="mydata/videos/chassis1.mp4", help="Path to input MP4 video")
    parser.add_argument("--output", type=str, default="mydata/processed/tracker_sim_output.mp4", help="Path to output annotated MP4")
    parser.add_argument("--mismatch", action="store_true", help="Simulate a model mismatch scenario")
    parser.add_argument("--fps", type=int, default=1, help="Simulation processing FPS (default: 1)")
    args = parser.parse_args()

    # Verify input video exists
    if not os.path.exists(args.video):
        print(f"Error: Video file not found at {args.video}.")
        print("Please save the chassis video files in: mydata/videos/")
        return

    # Load Homography
    homography_path = "mydata/metadata/homography.npy"
    config_path = "mydata/metadata/warp_config.json"
    model_path = r"runs/detect/yolo/runs/train/weights/best.pt"

    if not os.path.exists(homography_path) or not os.path.exists(config_path):
        print("Error: Calibration metadata missing. Run warp_test.py and calibrate_offsets.py first.")
        return
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}.")
        return

    H = np.load(homography_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    lane_width = config["lane_width"]
    lane_height = config["lane_height"]

    # Load YOLO Model
    model = YOLO(model_path)

    # Initialize Geometry Tracker
    tracker = GeometryTracker()

    # Pre-populate FIFO queues based on simulation mode
    if args.mismatch:
        print("\n--- SIMULATING MISMATCH SCENARIO ---")
        # FL gets "GREEN_A", FR gets "GREEN_A" (MATCH)
        # RL gets "RED_B", RR gets "BLUE_C" (MISMATCH)
        tracker.add_left_event("GREEN_A")
        tracker.add_left_event("RED_B")
        
        tracker.add_right_event("GREEN_A")
        tracker.add_right_event("BLUE_C")
    else:
        print("\n--- SIMULATING NORMAL MATCH SCENARIO ---")
        # FL gets "GREEN_A", FR gets "GREEN_A" (MATCH)
        # RL gets "RED_B", RR gets "RED_B" (MATCH)
        tracker.add_left_event("GREEN_A")
        tracker.add_left_event("RED_B")
        
        tracker.add_right_event("GREEN_A")
        tracker.add_right_event("RED_B")

    print(f"Initial Left Queue: {tracker.left_queue}")
    print(f"Initial Right Queue: {tracker.right_queue}")

    # Open Video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Failed to open video {args.video}")
        return

    # Video specs
    native_fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate skip frames step
    frame_step = max(1, int(native_fps / args.fps))
    print(f"Loaded video: {args.video} | Native FPS: {native_fps} | Total Frames: {total_frames}")
    print(f"Simulation configured at {args.fps} FPS (processing every {frame_step} frames)")

    # Set up VideoWriter (write output in same size as warped lane)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, args.fps, (lane_width, lane_height))

    frame_idx = 0
    processed_count = 0
    last_status = "PENDING"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if (frame_idx - 1) % frame_step != 0:
            continue

        processed_count += 1

        # Warp perspective
        warped = cv2.warpPerspective(frame, H, (lane_width, lane_height))

        # Run YOLO prediction on warped frame
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

        # Draw overlays on warped frame
        annotated = warped.copy()

        if state["active"]:
            # Draw tracked references
            if state["ref_L"] is not None:
                cv2.circle(annotated, tuple(map(int, state["ref_L"])), 8, (0, 0, 255), -1) # Red RL Ref
            if state["ref_R"] is not None:
                cv2.circle(annotated, tuple(map(int, state["ref_R"])), 8, (255, 0, 255), -1) # Magenta RR Ref

            # Draw slots and ROIs
            for label, slot in state["slots"].items():
                if slot["center"] is not None:
                    cx, cy = map(int, slot["center"])
                    rx1, ry1 = cx - tracker.roi_w, cy - tracker.roi_h
                    rx2, ry2 = cx + tracker.roi_w, cy + tracker.roi_h

                    # ROI color:
                    # Blue if empty/pending, Green if occupied and matching, Red if mismatch/UNKNOWN
                    is_left = label.endswith("L")
                    match_res = state["front_match"] if label.startswith("F") else state["rear_match"]
                    
                    if not slot["occupied"]:
                        color = (255, 255, 0) # Cyan for empty
                        text = f"{label}: EMPTY"
                    else:
                        if match_res == "PASS":
                            color = (0, 255, 0) # Green for match
                        elif match_res == "FAIL":
                            color = (0, 0, 255) # Red for mismatch
                        else:
                            color = (0, 255, 255) # Yellow for pending single slot occupied
                        text = f"{label}: {slot['model']}"

                    cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), color, 2)
                    cv2.circle(annotated, (cx, cy), 4, color, -1)
                    cv2.putText(annotated, text, (rx1, ry1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

            # Draw status banner
            status = state["status"]
            if status == "PASS":
                banner_color = (0, 255, 0) # Green
            elif status == "FAIL":
                banner_color = (0, 0, 255) # Red
            else:
                banner_color = (0, 255, 255) # Yellow/Cyan
            
            # Print state transition in console
            if status != last_status:
                print(f"Frame {frame_idx:04d}: Match Status changed {last_status} -> {status}")
                print(f"  Slots: FL={state['slots']['FL']['model']}, FR={state['slots']['FR']['model']} | RL={state['slots']['RL']['model']}, RR={state['slots']['RR']['model']}")
                last_status = status

            cv2.rectangle(annotated, (0, 0), (lane_width, 60), banner_color, -1)
            cv2.putText(annotated, f"VEHICLE DETECTED | STATUS: {status}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(annotated, f"L Que: {state['left_queue']} | R Que: {state['right_queue']}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        else:
            # Idle/Waiting state
            cv2.rectangle(annotated, (0, 0), (lane_width, 60), (128, 128, 128), -1)
            cv2.putText(annotated, "WAITING FOR VEHICLE...", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Write frame to output video
        out.write(annotated)

        print(f"Processed frame {frame_idx}/{total_frames} (sim frame {processed_count})")

    cap.release()
    out.release()
    print(f"\nSimulation complete. Outputs saved to: {args.output}")

if __name__ == "__main__":
    main()

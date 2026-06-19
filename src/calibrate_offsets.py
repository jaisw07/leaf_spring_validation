import os
import json
import cv2
import numpy as np
from ultralytics import YOLO

def main():
    # Paths
    homography_path = "mydata/metadata/homography.npy"
    config_path = "mydata/metadata/warp_config.json"
    model_path = r"runs/detect/yolo/runs/train/weights/best.pt"
    samples_dir = "mydata/calibration_samples"
    output_dir = "mydata/processed/calibration"
    offsets_output_path = "mydata/metadata/offsets.json"

    # Verify paths
    if not os.path.exists(homography_path) or not os.path.exists(config_path):
        print("Error: Perspective warp calibration files missing. Run warp_test.py first.")
        return
    if not os.path.exists(model_path):
        print(f"Error: YOLO model weights not found at {model_path}.")
        return
    if not os.path.exists(samples_dir):
        print(f"Error: Calibration samples directory {samples_dir} does not exist.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Load calibration & model
    H = np.load(homography_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    lane_width = config["lane_width"]
    lane_height = config["lane_height"]

    model = YOLO(model_path)

    # Scan images
    valid_exts = (".png", ".jpg", ".jpeg")
    images = [f for f in os.listdir(samples_dir) if f.lower().endswith(valid_exts)]
    if not images:
        print(f"No calibration images found in {samples_dir}.")
        return

    print(f"Found {len(images)} calibration samples.")
    results_list = []

    for img_name in sorted(images):
        img_path = os.path.join(samples_dir, img_name)
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Warning: Failed to read {img_name}. Skipping.")
            continue

        # Warp perspective
        warped = cv2.warpPerspective(frame, H, (lane_width, lane_height))

        # Predict springs
        preds = model.predict(source=warped, conf=0.5, verbose=False)
        boxes = preds[0].boxes

        if boxes is None or len(boxes) < 4:
            print(f"Warning: Image {img_name} detected {len(boxes) if boxes is not None else 0} objects. Need at least 4 springs. Skipping.")
            continue

        # Filter for class 1 (spring)
        spring_dets = []
        for box in boxes:
            cls = int(box.cls[0].item())
            if cls == 1: # spring
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                conf = box.conf[0].item()
                spring_dets.append({
                    "bbox": [x1, y1, x2, y2],
                    "center": (cx, cy),
                    "conf": conf
                })

        if len(spring_dets) < 4:
            print(f"Warning: Image {img_name} detected only {len(spring_dets)} springs. Skipping.")
            continue

        # Take top 4 by confidence if there are more
        spring_dets = sorted(spring_dets, key=lambda x: x["conf"], reverse=True)[:4]

        # Sort by Y coordinate to separate Front and Rear
        spring_dets = sorted(spring_dets, key=lambda x: x["center"][1])
        front_springs = spring_dets[:2]
        rear_springs = spring_dets[2:]

        # Sort Front by X (left to right)
        front_springs = sorted(front_springs, key=lambda x: x["center"][0])
        FL = front_springs[0]
        FR = front_springs[1]

        # Sort Rear by X (left to right)
        rear_springs = sorted(rear_springs, key=lambda x: x["center"][0])
        RL = rear_springs[0]
        RR = rear_springs[1]

        # Calculate offsets (Front - Rear)
        dx_L = FL["center"][0] - RL["center"][0]
        dy_L = FL["center"][1] - RL["center"][1]

        dx_R = FR["center"][0] - RR["center"][0]
        dy_R = FR["center"][1] - RR["center"][1]

        results_list.append({
            "image": img_name,
            "FL": FL["center"],
            "FR": FR["center"],
            "RL": RL["center"],
            "RR": RR["center"],
            "dx_L": dx_L,
            "dy_L": dy_L,
            "dx_R": dx_R,
            "dy_R": dy_R
        })

        print(f"Image {img_name}:")
        print(f"  Left Side (FL -> RL): dx = {dx_L:+.2f}, dy = {dy_L:+.2f}")
        print(f"  Right Side (FR -> RR): dx = {dx_R:+.2f}, dy = {dy_R:+.2f}")

        # Annotate warped image for visual inspection
        annotated = warped.copy()
        colors = {
            "FL": (0, 255, 0),   # Green
            "FR": (255, 255, 0), # Cyan/Yellow
            "RL": (0, 0, 255),   # Red
            "RR": (255, 0, 255)  # Magenta
        }

        for label, spring in [("FL", FL), ("FR", FR), ("RL", RL), ("RR", RR)]:
            x1, y1, x2, y2 = map(int, spring["bbox"])
            cx, cy = map(int, spring["center"])
            color = colors[label]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.circle(annotated, (cx, cy), 5, color, -1)
            cv2.putText(annotated, f"{label} ({cx},{cy})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # Draw connecting line vectors
        cv2.line(annotated, (int(RL["center"][0]), int(RL["center"][1])), 
                 (int(FL["center"][0]), int(FL["center"][1])), (0, 255, 255), 1)
        cv2.line(annotated, (int(RR["center"][0]), int(RR["center"][1])), 
                 (int(FR["center"][0]), int(FR["center"][1])), (0, 255, 255), 1)

        out_name = f"warped_{os.path.splitext(img_name)[0]}.jpg"
        cv2.imwrite(os.path.join(output_dir, out_name), annotated)

    if not results_list:
        print("No valid calibration samples processed.")
        return

    # Calculate statistics
    dx_L_vals = [r["dx_L"] for r in results_list]
    dy_L_vals = [r["dy_L"] for r in results_list]
    dx_R_vals = [r["dx_R"] for r in results_list]
    dy_R_vals = [r["dy_R"] for r in results_list]

    stats = {
        "left": {
            "dx_mean": float(np.mean(dx_L_vals)),
            "dx_std": float(np.std(dx_L_vals)),
            "dy_mean": float(np.mean(dy_L_vals)),
            "dy_std": float(np.std(dy_L_vals))
        },
        "right": {
            "dx_mean": float(np.mean(dx_R_vals)),
            "dx_std": float(np.std(dx_R_vals)),
            "dy_mean": float(np.mean(dy_R_vals)),
            "dy_std": float(np.std(dy_R_vals))
        },
        "samples_count": len(results_list)
    }

    # Save to file
    with open(offsets_output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)

    print("\nCalibration Results:")
    print("====================")
    print(f"Processed {len(results_list)} samples.")
    print("Left Side Offset (FL - RL):")
    print(f"  dx: mean = {stats['left']['dx_mean']:+.4f}, std = {stats['left']['dx_std']:.4f}")
    print(f"  dy: mean = {stats['left']['dy_mean']:+.4f}, std = {stats['left']['dy_std']:.4f}")
    print("Right Side Offset (FR - RR):")
    print(f"  dx: mean = {stats['right']['dx_mean']:+.4f}, std = {stats['right']['dx_std']:.4f}")
    print(f"  dy: mean = {stats['right']['dy_mean']:+.4f}, std = {stats['right']['dy_std']:.4f}")
    print(f"\nCalibration metrics saved to: {offsets_output_path}")
    print(f"Annotated visualization outputs saved to: {output_dir}")

if __name__ == "__main__":
    main()

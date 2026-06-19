import os
import json
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

def warp_single_image(args):
    img_path, out_path, H, lane_width, lane_height = args
    try:
        frame = cv2.imread(img_path)
        if frame is None:
            return img_path, False, "Failed to read image"
        
        warped = cv2.warpPerspective(
            frame,
            H,
            (lane_width, lane_height)
        )
        
        # Save as JPEG with 95% quality
        success = cv2.imwrite(out_path, warped, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not success:
            return img_path, False, "Failed to write image"
            
        return img_path, True, None
    except Exception as e:
        return img_path, False, str(e)

def main():
    # Load homography and config
    homography_path = "mydata/metadata/homography.npy"
    config_path = "mydata/metadata/warp_config.json"
    
    if not os.path.exists(homography_path) or not os.path.exists(config_path):
        print("Error: Calibration metadata missing. Run warp_test.py first.")
        return

    H = np.load(homography_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
        
    lane_width = config["lane_width"]
    lane_height = config["lane_height"]
    
    tasks = []
    
    # Map chassis raw directories to processed warp directories
    dir_mapping = {
        "mydata/raw/chassis1": "mydata/processed/warp1",
        "mydata/raw/chassis2": "mydata/processed/warp2",
        "mydata/raw/chassis3": "mydata/processed/warp3"
    }
    
    for src_dir, dst_dir in dir_mapping.items():
        if not os.path.exists(src_dir):
            print(f"Warning: Source directory {src_dir} does not exist. Skipping.")
            continue
            
        os.makedirs(dst_dir, exist_ok=True)
        
        # Scan for PNG images
        files = [f for f in os.listdir(src_dir) if f.lower().endswith(".png")]
        print(f"Found {len(files)} images in {src_dir}")
        
        for file_name in files:
            img_path = os.path.join(src_dir, file_name)
            # Change extension to .jpg for output
            base_name, _ = os.path.splitext(file_name)
            out_file_name = f"{base_name}.jpg"
            out_path = os.path.join(dst_dir, out_file_name)
            
            tasks.append((img_path, out_path, H, lane_width, lane_height))
            
    total_tasks = len(tasks)
    if total_tasks == 0:
        print("No images found to process.")
        return
        
    print(f"Starting parallel warp on {total_tasks} images using ProcessPoolExecutor...")
    
    success_count = 0
    failure_count = 0
    
    # Use ProcessPoolExecutor to run tasks in parallel
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(warp_single_image, task): task for task in tasks}
        
        for i, future in enumerate(as_completed(futures), 1):
            img_path, success, error_msg = future.result()
            if success:
                success_count += 1
            else:
                failure_count += 1
                print(f"Error processing {img_path}: {error_msg}")
                
            if i % 50 == 0 or i == total_tasks:
                print(f"Progress: {i}/{total_tasks} completed ({success_count} success, {failure_count} failure)")
                
    print(f"Finished. Success: {success_count}, Failures: {failure_count}")

if __name__ == "__main__":
    main()

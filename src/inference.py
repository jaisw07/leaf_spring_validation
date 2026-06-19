import os
import torch
from ultralytics import YOLO

def main():
    model_path = r"C:\Users\SHREY\Desktop\2026-06-15_IPCAMRecording\edited\runs\detect\yolo\runs\train\weights\best.pt"
    test_images_dir = r"C:\Users\SHREY\Desktop\2026-06-15_IPCAMRecording\edited\yolo\images\test"
    project_dir = r"C:\Users\SHREY\Desktop\2026-06-15_IPCAMRecording\edited\runs\detect\yolo\runs"
    
    # Load the best trained model weights
    model = YOLO(model_path)
    
    # Run prediction on the test set
    results = model.predict(
        source=test_images_dir,
        save=True,          # Save annotated bounding box images
        save_txt=True,      # Save detection coordinates to text files
        save_conf=True,     # Save confidence scores along with coordinates
        project=project_dir,
        name="inference",   # Subdirectory for this run
        device=0 if torch.cuda.is_available() else "cpu"
    )
    
    # Calculate detection summary
    total_rods = 0
    total_springs = 0
    
    for r in results:
        boxes = r.boxes
        if boxes is not None:
            classes = boxes.cls.tolist()
            total_rods += classes.count(0)
            total_springs += classes.count(1)
            
    print(f"\nInference completed on {len(results)} test images.")
    print(f"Total objects detected:")
    print(f"  Rods (class 0): {total_rods}")
    print(f"  Springs (class 1): {total_springs}")
    print(f"Outputs saved to: {os.path.join(project_dir, 'inference')}")

if __name__ == "__main__":
    main()

import torch
from ultralytics import YOLO

def main():
    model_path = r"C:\Users\SHREY\Desktop\2026-06-15_IPCAMRecording\edited\runs\detect\yolo\runs\train\weights\best.pt"
    data_yaml = r"C:\Users\SHREY\Desktop\2026-06-15_IPCAMRecording\edited\yolo\dataset.yaml"
    
    # Load model
    model = YOLO(model_path)
    
    # Run validation on the test set
    print("Evaluating model on test dataset...")
    metrics = model.val(
        data=data_yaml,
        split="test",       # Use the test split
        device=0 if torch.cuda.is_available() else "cpu",
        rect=True           # Keep aspect ratio matching test set
    )
    
    # Print summary
    print("\nTest Evaluation Summary:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    
    # Print per-class metrics
    names = model.names
    for cid, name in names.items():
        p = metrics.box.p[cid] if cid < len(metrics.box.p) else 0.0
        r = metrics.box.r[cid] if cid < len(metrics.box.r) else 0.0
        ap50 = metrics.box.ap50[cid] if cid < len(metrics.box.ap50) else 0.0
        ap = metrics.box.ap[cid] if cid < len(metrics.box.ap) else 0.0
        
        print(f"  Class {cid} ({name}):")
        print(f"    Precision: {p:.4f}")
        print(f"    Recall:    {r:.4f}")
        print(f"    mAP50:     {ap50:.4f}")
        print(f"    mAP50-95:  {ap:.4f}")

if __name__ == "__main__":
    main()

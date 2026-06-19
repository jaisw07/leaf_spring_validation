import os
import csv
import torch
from ultralytics import YOLO

def on_fit_epoch_end(trainer):
    """
    Custom callback triggered at the end of each training & validation epoch.
    Extracts and logs per-class validation metrics (Precision, Recall, mAP50, mAP50-95)
    to a CSV file inside the training directory.
    """
    epoch = trainer.epoch + 1  # 1-indexed epoch
    metrics_file = os.path.join(trainer.save_dir, "per_class_metrics.csv")
    
    # Access the validator and its metrics safely
    validator = getattr(trainer, "validator", None)
    if validator is None:
        return
        
    metrics = getattr(validator, "metrics", None)
    if metrics is None:
        return
        
    # Get class names defined in model
    names = trainer.model.names
    
    # Extract metric arrays
    p_class = getattr(metrics.box, "p", [])
    r_class = getattr(metrics.box, "r", [])
    ap50_class = getattr(metrics.box, "ap50", [])
    ap_class = getattr(metrics.box, "ap", [])
    
    # Check if files already exist to write header
    file_exists = os.path.exists(metrics_file)
    with open(metrics_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["epoch", "class_id", "class_name", "precision", "recall", "ap50", "ap50_95"])
            
        # Log values for each class
        for cid, name in names.items():
            # Safely fetch values matching indices
            p = p_class[cid] if cid < len(p_class) else 0.0
            r = r_class[cid] if cid < len(r_class) else 0.0
            ap50 = ap50_class[cid] if cid < len(ap50_class) else 0.0
            ap = ap_class[cid] if cid < len(ap_class) else 0.0
            
            writer.writerow([epoch, cid, name, f"{p:.4f}", f"{r:.4f}", f"{ap50:.4f}", f"{ap:.4f}"])

def main():
    # Detect GPU / CUDA device
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} (CUDA available: {torch.cuda.is_available()})")
    
    # Load pretrained YOLO26 nano model
    model = YOLO("yolo26n.pt")
    
    # Register custom epoch logging callback
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    
    # Define dataset configuration path
    data_yaml = r"C:\Users\SHREY\Desktop\2026-06-15_IPCAMRecording\edited\yolo\dataset.yaml"
    
    # Start training with hyperparameters aligned with best practices
    model.train(
        data=data_yaml,
        epochs=100,
        batch=-1,             # Auto batch size based on GPU memory limit (70% VRAM)
        imgsz=[576, 1088],    # Native resolution to capture fine detail of rods & springs
        device=device,        # Run on RTX 4050 GPU if available
        optimizer="SGD",      # SGD with momentum for robust generalization
        lr0=0.01,             # Standard initial learning rate
        momentum=0.937,       # SGD momentum
        weight_decay=0.0005,  # L2 regularization
        cos_lr=True,          # Cosine learning rate decay scheduler
        patience=20,          # Early stopping patience: stop if val mAP does not improve for 20 epochs
        val=True,             # Validate every epoch
        rect=True,            # Rectangular training (efficient for non-square aspect ratio)
        close_mosaic=10,      # Disable mosaic augmentation during final 10 epochs for clean convergence
        workers=4,            # Data loading workers
        seed=42,              # Reproducible seed
        save=True,            # Explicitly save model checkpoints and best model weights
        project="yolo/runs",  # Save results to yolo/runs/
        name="train"          # Save current run in train/ folder
    )

if __name__ == "__main__":
    main()

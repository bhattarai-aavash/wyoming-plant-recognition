from ultralytics import YOLO

# --- CONFIGURATION ---
# IMPORTANT: You MUST use a model ending in '-seg.pt'
# Options: yolov8n-seg.pt (Nano), yolov8s-seg.pt (Small), yolov8m-seg.pt (Medium)
# (Or yolo11s-seg.pt if you upgraded)
MODEL_NAME = 'yolov8s-seg.pt' 
DATA_PATH = './combined_seg_dataset/data.yaml'

def train_robust_segmentation():
    # 1. Load the Segmentation Model
    print(f"Loading {MODEL_NAME} for Segmentation...")
    model = YOLO(MODEL_NAME)

    # 2. Train with Heavy Augmentation
    print("Starting robust training run...")
    
    results = model.train(
        data=DATA_PATH,
        epochs=150,           
        imgsz=640,
        batch=16,             # Reduce to 8 or 4 if you get "CUDA Out of Memory"
        patience=50,          # Stop if no improvement for 50 epochs
        
        # --- ROBUSTNESS / AUGMENTATION SETTINGS ---
        
        # Color & Lighting (Crucial for greenhouse/outdoor lighting changes)
        hsv_h=0.04,           # Hue shift (slight color variance)
        hsv_s=0.6,            # Saturation shift (dull vs vibrant days)
        hsv_v=0.6,            # Value/Brightness shift (bright sun vs shadows)
        
        # Geometric (Positioning)
        degrees=15.0,         # Rotation (+/- 15 degrees)
        translate=0.2,        # Translation (sliding image)
        scale=0.6,            # Scaling (zoom in/out heavily)
        shear=2.0,            # Shear angle
        perspective=0.0005,   # Perspective distortion
        
        # Flipping
        fliplr=0.5,           # 50% chance to flip Left-Right
        flipud=0.2,           # 20% chance to flip Up-Down (useful for drooping plants)
        
        # Advanced Compositions (The "Secret Sauce")
        mosaic=1.0,           # 100% Mosaic (Stitches 4 images into one)
        mixup=0.15,           # 15% Mixup (Blends two images transparently)
        copy_paste=0.3,       # 30% Copy-Paste (Specific to Segmentation: Pastes objects onto other images)
        
        # Segmentation Specific
        overlap_mask=True,    # Handle overlapping masks gracefully
        mask_ratio=4,         # Downsample ratio for masks (standard is 4)
        
        # System
        optimizer='auto',     
        verbose=True,
        device=0              # Force GPU 0
    )

    print("Training Complete.")
    # The results object contains the path to the best model
    print(f"Best model saved in: {results.save_dir}/weights/best.pt")

if __name__ == '__main__':
    train_robust_segmentation()
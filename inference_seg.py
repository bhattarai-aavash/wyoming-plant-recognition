import os
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

# --- CONFIGURATION ---
SOURCE_VIDEO_PATH = "/home/abhattar/auto_label/data/Dataset/Testing-20260128T172053Z-3-001/Testing/IMG_1403.mp4"       # Update this
TARGET_VIDEO_PATH = "my_plant_video_seg_3.mp4"
MODEL_PATH = "runs/segment/train/weights/best.pt" # Points to your SEGMENTATION model
CONFIDENCE_THRESHOLD = 0.5
MASK_OPACITY = 0.4  # How transparent the mask is (0.0 to 1.0)

# Define colors for your classes (B, G, R)
COLORS = [
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (0, 0, 255)     # Red
]

def process_video_segmentation():
    print(f"Loading segmentation model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)

    def callback(scene: cv2.Mat, index: int) -> cv2.Mat:
        # A. Run Inference
        results = model(scene)[0]
        
        # Check if we have any detections
        if results.masks is None:
            return scene

        # B. Prepare the "Overlay" for the masks
        # We draw all masks onto this copy first, then blend it later
        mask_overlay = scene.copy()
        
        # We need the polygons (xy coordinates) and the box info
        # results.masks.xy gives us the polygon points for every detection
        masks_polygons = results.masks.xy 
        boxes = results.boxes

        # Iterate through every object found
        for i, box in enumerate(boxes):
            # 1. Check Confidence
            conf = float(box.conf)
            if conf < CONFIDENCE_THRESHOLD:
                continue

            # 2. Get Class ID and Color
            cls_id = int(box.cls)
            class_name = model.names[cls_id]
            color = COLORS[cls_id % len(COLORS)]

            # --- DRAWING THE MASK (Polygon) ---
            # Get the polygon points for this specific object
            polygon = masks_polygons[i]
            
            # cv2.fillPoly requires integer points
            polygon = np.array(polygon, dtype=np.int32)
            
            # Draw the filled polygon onto our overlay layer
            cv2.fillPoly(mask_overlay, [polygon], color)

        # C. Blend the Mask Overlay with the Original Scene
        # This creates the transparency effect
        cv2.addWeighted(
            mask_overlay, MASK_OPACITY,  # Source 1 (The colored masks)
            scene, 1 - MASK_OPACITY,     # Source 2 (The original video)
            0, 
            scene                        # Destination
        )

        # D. Draw Boxes and Labels ON TOP (So they are sharp, not transparent)
        for box in boxes:
            conf = float(box.conf)
            if conf < CONFIDENCE_THRESHOLD: continue
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls)
            class_name = model.names[cls_id]
            color = COLORS[cls_id % len(COLORS)]

            # Draw Rectangle
            cv2.rectangle(scene, (x1, y1), (x2, y2), color, thickness=3)

            # Draw Label Background & Text
            label = f"{class_name} {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(scene, (x1, y1 - 25), (x1 + w, y1), color, -1)
            cv2.putText(scene, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return scene

    # Process Video
    print(f"Processing {SOURCE_VIDEO_PATH}...")
    sv.process_video(
        source_path=SOURCE_VIDEO_PATH,
        target_path=TARGET_VIDEO_PATH,
        callback=callback
    )
    print(f"Done! Saved to {TARGET_VIDEO_PATH}")

if __name__ == "__main__":
    process_video_segmentation()
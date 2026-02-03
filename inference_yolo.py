import argparse
import os
import cv2
import supervision as sv
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO inference on a video and save an annotated output video")
    parser.add_argument("--source", "-s", default="/home/abhattar/auto_label/data/Dataset/Testing-20260128T172053Z-3-001/Testing/IMG_1403.mp4",
                        help="Path to source video file")
    parser.add_argument("--target", "-t", default=None,
                        help="Path to output annotated file (video or image). If omitted a sensible name will be used.")
    parser.add_argument("--model", "-m", default="runs/detect/train/weights/best.pt",
                        help="Path to YOLO model weights or checkpoint")
    parser.add_argument("--confidence", "-c", type=float, default=0.5,
                        help="Confidence threshold for displaying detections")
    parser.add_argument("--device", default=None, help="Device to run inference on (e.g. 'cpu' or '0')")
    return parser.parse_args()


def process_video_inference(source_path: str, target_path: str, model_path: str, confidence_threshold: float, device=None):
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)

    # Callback function using raw OpenCV
    def callback(scene: cv2.Mat, index: int) -> cv2.Mat:
        # A. Run Inference
        results = model(scene)[0]

        # B. Get the boxes and classes directly
        for box in results.boxes:
            # 1. Check Confidence
            conf = float(box.conf)
            if conf < confidence_threshold:
                continue

            # 2. Get Coordinates (x1, y1, x2, y2)
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # 3. Get Class ID and Name
            cls_id = int(box.cls)
            class_name = model.names[cls_id]

            # Select color based on class ID (wrap around if more classes than colors)
            COLORS = [
                (0, 255, 0),    # Green
                (255, 0, 0),    # Blue
                (0, 0, 255)     # Red
            ]
            color = COLORS[cls_id % len(COLORS)]

            # 4. Draw Rectangle (Thick and Visible)
            cv2.rectangle(scene, (x1, y1), (x2, y2), color, thickness=3)

            # 5. Draw Label
            label = f"{class_name} {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(scene, (x1, y1 - 25), (x1 + w, y1), color, -1)
            cv2.putText(scene, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return scene

    # Decide whether source is image or video
    _, ext = os.path.splitext(source_path)
    ext = ext.lower()
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

    if ext in image_exts:
        # Image inference: read, annotate, write
        print(f"Running image inference on {source_path}...")
        img = cv2.imread(source_path)
        if img is None:
            raise SystemExit(f"Failed to read image: {source_path}")

        annotated = callback(img, 0)

        # Derive target path if not provided
        if not target_path:
            base, _ = os.path.splitext(source_path)
            target_path = f"{base}_labeled{ext}"

        cv2.imwrite(target_path, annotated)
        print(f"Done! Saved annotated image to {target_path}")
    else:
        # Video inference
        print(f"Processing video {source_path}...")
        # Derive target path if not provided
        if not target_path:
            base = os.path.splitext(os.path.basename(source_path))[0]
            target_path = f"{base}_labeled.mp4"

        sv.process_video(
            source_path=source_path,
            target_path=target_path,
            callback=callback
        )
        print(f"Done! Saved annotated video to {target_path}")


if __name__ == "__main__":
    args = parse_args()
    process_video_inference(args.source, args.target, args.model, args.confidence, args.device)
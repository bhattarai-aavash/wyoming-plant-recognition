import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train a YOLOv8 model with robust augmentation settings")
    parser.add_argument("--model", default='yolov8s.pt', help="Base model checkpoint to start from (e.g. yolov8s.pt)")
    parser.add_argument("--data", default='/home/abhattar/auto_label/combined_dataset/data.yaml', help="Path to data.yaml for training")
    parser.add_argument("--device", default='0', help="Device id for training (e.g. 0 or 'cpu')")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    return parser.parse_args()


def train_robust_model(model_size: str, data_path: str, device, epochs: int, imgsz: int, batch: int):
    # 1. Load the model
    print(f"Loading {model_size}...")
    model = YOLO(model_size)

    # 2. Train with Heavy Augmentation
    print("Starting robust training run...")

    results = model.train(
        data=data_path,
        device=device,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=50,

        # --- ROBUSTNESS / AUGMENTATION SETTINGS ---
        hsv_h=0.04,
        hsv_s=0.6,
        hsv_v=0.6,

        degrees=15.0,
        translate=0.2,
        scale=0.6,
        shear=2.0,
        perspective=0.0005,

        fliplr=0.5,
        flipud=0.2,

        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.3,

        optimizer='auto',
        verbose=True
    )

    print("Training Complete. Best model saved in 'runs/detect/train/weights/best.pt'")


if __name__ == '__main__':
    args = parse_args()
    train_robust_model(args.model, args.data, args.device, args.epochs, args.imgsz, args.batch)
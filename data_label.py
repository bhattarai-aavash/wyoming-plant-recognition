import argparse
import os
import cv2
import yaml
import supervision as sv
from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology


def parse_args():
    parser = argparse.ArgumentParser(
        description="Label images with GroundedSAM and produce visual previews and a YOLO dataset"
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--input", "-i", default=None,
                        help="Path to input images folder (single-folder mode)")
    group.add_argument("--data-root", "-r", default=None,
                        help="Path to a root directory containing class subfolders (batch mode)")

    parser.add_argument("--dataset", "-d", default=None,
                        help="Path where single-folder YOLO dataset will be written (single mode)")
    parser.add_argument("--dataset-root", default='.',
                        help="Root folder under which per-class dataset folders will be created when using --data-root (default: current dir)")
    parser.add_argument("--visual", "-v", default=None,
                        help="Path where single-folder visual preview images will be written (single mode)")
    parser.add_argument("--visual-root", default='.',
                        help="Root folder under which per-class visual preview folders will be created when using --data-root (default: current dir)")
    parser.add_argument("--confidence", "-c", type=float, default=0.5,
                        help="Confidence threshold for filtering detections")
    return parser.parse_args()


def main(input_dir: str, dataset_dir: str, visual_dir: str, confidence_threshold: float):
    # 1. Setup Class Naming and Ontology
    dataset_class_name = os.path.basename(os.path.normpath(input_dir))
    ontology = CaptionOntology({"plant": dataset_class_name})

    # Initialize the Model
    base_model = GroundedSAM(ontology=ontology)

    # --- PASS 1: Visualization (Draw Boxes) ---
    print(f"--- Pass 1: Generating Visual Previews in {visual_dir} ---")
    os.makedirs(visual_dir, exist_ok=True)

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    for image_name in os.listdir(input_dir):
        image_path = os.path.join(input_dir, image_name)

        if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        image = cv2.imread(image_path)
        if image is None:
            continue

        # Predict
        results = base_model.predict(image_path)

        # --- SKIP LOGIC ---
        # 1. If nothing found initially, skip this image
        if len(results) == 0:
            continue

        # 2. Try to filter, but if it crashes due to the shape bug, skip this image
        try:
            results = results[results.confidence > confidence_threshold]
        except ValueError:
            continue

        # 3. If filtering removed everything, skip this image
        if len(results) == 0:
            continue
        # ------------------

        # Annotate
        annotated_image = box_annotator.annotate(scene=image.copy(), detections=results)
        labels = [f"{dataset_class_name} {confidence:.2f}" for confidence in results.confidence]
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=results, labels=labels)

        cv2.imwrite(os.path.join(visual_dir, image_name), annotated_image)

    print("Visualization complete.")


    # --- PASS 2: Dataset Generation (YOLO Format) ---
    print(f"\n--- Pass 2: Generating YOLO Dataset in {dataset_dir} ---")

    base_model.label(
        input_folder=input_dir,
        output_folder=dataset_dir
    )

    # --- PASS 3: Fix data.yaml for Portability ---
    print(f"Fixing data.yaml paths...")
    yaml_path = os.path.join(dataset_dir, "data.yaml")

    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)

        # Force relative paths
        data_config['train'] = "./images/train"
        data_config['val'] = "./images/val"
        if 'test' in data_config:
            data_config['test'] = "./images/test"

        if 'path' in data_config:
            del data_config['path']

        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f, sort_keys=False)
    else:
        print(f"Warning: {yaml_path} not found. Skipping data.yaml fix.")

    print(f"Done! \n1. Visuals: {visual_dir} \n2. Training Data: {dataset_dir}")


def process_one(input_dir, dataset_dir, visual_dir, confidence_threshold):
    # small wrapper that ensures input exists and then calls main
    if not os.path.exists(input_dir) or not os.path.isdir(input_dir):
        print(f"Skipping {input_dir}: not found or not a directory")
        return
    # Create output directories if needed
    os.makedirs(os.path.abspath(os.path.dirname(dataset_dir)), exist_ok=True)
    os.makedirs(os.path.abspath(visual_dir), exist_ok=True)
    main(input_dir, dataset_dir, visual_dir, confidence_threshold)


if __name__ == "__main__":
    args = parse_args()

    # If user passed --data-root, run in batch mode over subfolders
    if args.data_root:
        if not os.path.exists(args.data_root) or not os.path.isdir(args.data_root):
            raise SystemExit(f"data-root {args.data_root} does not exist or is not a directory")

        for entry in sorted(os.listdir(args.data_root)):
            if entry.startswith('.'):
                continue
            cls_dir = os.path.join(args.data_root, entry)
            if not os.path.isdir(cls_dir):
                continue

            dataset_dir = os.path.join(args.dataset_root, f"{entry}_dataset")
            visual_dir = os.path.join(args.visual_root, f"{entry}_labeled_previews")

            print(f"\n=== Processing class folder: {entry} ===")
            process_one(cls_dir, dataset_dir, visual_dir, args.confidence)
    else:
        # Single-folder mode
        if not args.input:
            raise SystemExit("Either --input (single) or --data-root (batch) must be provided")

        dataset_dir = args.dataset if args.dataset else os.path.join('.', f"{os.path.basename(os.path.normpath(args.input))}_dataset")
        visual_dir = args.visual if args.visual else os.path.join('.', f"{os.path.basename(os.path.normpath(args.input))}_labeled_previews")

        process_one(args.input, dataset_dir, visual_dir, args.confidence)
import os
import cv2
import yaml
import shutil
import numpy as np
import supervision as sv
from PIL import Image, ImageOps  # Required for EXIF fix
from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology

# --- CONFIGURATION ---
INPUT_DIR = "/home/abhattar/auto_label/data/Sepherdia"
DATASET_DIR = "./sepherdia_seg_dataset"
VISUAL_DIR = "./sepherdia_seg_previews"
CONFIDENCE_THRESHOLD = 0.5

# 1. Setup Class Naming and Ontology
dataset_class_name = os.path.basename(INPUT_DIR)
ontology = CaptionOntology({"plant": dataset_class_name})

# Initialize the Model
base_model = GroundedSAM(ontology=ontology)

# --- HELPER: Load Image with EXIF Rotation Fixed ---
def load_image_correctly(path):
    try:
        # 1. Open with PIL (reads metadata)
        image_pil = Image.open(path)
        # 2. Apply EXIF transposition (fixes rotation)
        image_pil = ImageOps.exif_transpose(image_pil)
        # 3. Convert to OpenCV format (RGB -> BGR)
        return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

# --- PASS 1: Visualization (Draw MASKS) ---
print(f"--- Pass 1: Generating Segmentation Previews in {VISUAL_DIR} ---")
os.makedirs(VISUAL_DIR, exist_ok=True)

mask_annotator = sv.MaskAnnotator(opacity=0.5)
box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_scale=0.3)

for image_name in os.listdir(INPUT_DIR):
    image_path = os.path.join(INPUT_DIR, image_name)
    if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')): continue

    # 1. Load the image manually
    image = load_image_correctly(image_path)
    if image is None: continue

    # 2. Predict on the IMAGE ARRAY (Not the file path)
    # This forces the model to use the exact dimensions we just loaded.
    results = base_model.predict(image)
    
    # 3. Skip Logic
    if len(results) == 0: continue
    try:
        results = results[results.confidence > CONFIDENCE_THRESHOLD]
    except ValueError: continue
    if len(results) == 0: continue

    # 4. Annotate
    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=results)
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=results)
    labels = [f"{dataset_class_name} {conf:.2f}" for conf in results.confidence]
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=results, labels=labels)

    cv2.imwrite(os.path.join(VISUAL_DIR, image_name), annotated_image)

print("Visualization complete.")

# --- PASS 2: Dataset Generation (YOLO SEGMENTATION Format) ---
print(f"\n--- Pass 2: Generating YOLO Segmentation Dataset ---")

images_map = {}
annotations_map = {}

print("Collecting data...")
for image_name in os.listdir(INPUT_DIR):
    image_path = os.path.join(INPUT_DIR, image_name)
    if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')): continue

    # 1. Load manually again
    image = load_image_correctly(image_path)
    if image is None: continue

    # 2. Predict on the array
    results = base_model.predict(image)
    
    # 3. Filter
    if len(results) == 0: continue
    try:
        results = results[results.confidence > CONFIDENCE_THRESHOLD]
    except ValueError: continue
    if len(results) == 0: continue

    images_map[image_name] = imageimport os
import cv2
import yaml
import shutil
import numpy as np
import supervision as sv
from PIL import Image, ImageOps  # Required for EXIF fix
from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology

# --- CONFIGURATION ---
INPUT_DIR = "/home/abhattar/auto_label/data/Sepherdia"
DATASET_DIR = "./sepherdia_seg_dataset"
VISUAL_DIR = "./sepherdia_seg_previews"
CONFIDENCE_THRESHOLD = 0.5

# 1. Setup Class Naming and Ontology
dataset_class_name = os.path.basename(INPUT_DIR)
ontology = CaptionOntology({"plant": dataset_class_name})

# Initialize the Model
base_model = GroundedSAM(ontology=ontology)

# --- HELPER: Load Image with EXIF Rotation Fixed ---
def load_image_correctly(path):
    try:
        # 1. Open with PIL (reads metadata)
        image_pil = Image.open(path)
        # 2. Apply EXIF transposition (fixes rotation)
        image_pil = ImageOps.exif_transpose(image_pil)
        # 3. Convert to OpenCV format (RGB -> BGR)
        return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

# --- PASS 1: Visualization (Draw MASKS) ---
print(f"--- Pass 1: Generating Segmentation Previews in {VISUAL_DIR} ---")
os.makedirs(VISUAL_DIR, exist_ok=True)

mask_annotator = sv.MaskAnnotator(opacity=0.5)
box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_scale=0.3)

for image_name in os.listdir(INPUT_DIR):
    image_path = os.path.join(INPUT_DIR, image_name)
    if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')): continue

    # 1. Load the image manually
    image = load_image_correctly(image_path)
    if image is None: continue

    # 2. Predict on the IMAGE ARRAY (Not the file path)
    # This forces the model to use the exact dimensions we just loaded.
    results = base_model.predict(image)
    
    # 3. Skip Logic
    if len(results) == 0: continue
    try:
        results = results[results.confidence > CONFIDENCE_THRESHOLD]
    except ValueError: continue
    if len(results) == 0: continue

    # 4. Annotate
    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=results)
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=results)
    labels = [f"{dataset_class_name} {conf:.2f}" for conf in results.confidence]
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=results, labels=labels)

    cv2.imwrite(os.path.join(VISUAL_DIR, image_name), annotated_image)

print("Visualization complete.")

# --- PASS 2: Dataset Generation (YOLO SEGMENTATION Format) ---
print(f"\n--- Pass 2: Generating YOLO Segmentation Dataset ---")

images_map = {}
annotations_map = {}

print("Collecting data...")
for image_name in os.listdir(INPUT_DIR):
    image_path = os.path.join(INPUT_DIR, image_name)
    if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')): continue

    # 1. Load manually again
    image = load_image_correctly(image_path)
    if image is None: continue

    # 2. Predict on the array
    results = base_model.predict(image)
    
    # 3. Filter
    if len(results) == 0: continue
    try:
        results = results[results.confidence > CONFIDENCE_THRESHOLD]
    except ValueError: continue
    if len(results) == 0: continue

    images_map[image_name] = image
    annotations_map[image_name] = results

# Create Supervision Dataset
dataset = sv.DetectionDataset(
    classes=[dataset_class_name],
    images=images_map,
    annotations=annotations_map
)

# Split and Save
train_dataset, val_dataset = dataset.split(split_ratio=0.8)

def save_split(sv_dataset, split_name):
    split_dir = os.path.join(DATASET_DIR, split_name)
    sv_dataset.as_yolo(
        images_directory_path=os.path.join(split_dir, "images"),
        annotations_directory_path=os.path.join(split_dir, "labels"),
        data_yaml_path=os.path.join(split_dir, "data.yaml")
    )

if os.path.exists(DATASET_DIR): shutil.rmtree(DATASET_DIR)
save_split(train_dataset, "train")
save_split(val_dataset, "valid")

# --- PASS 3: Fix Folder Structure ---
print(f"Refining folder structure for YOLOv8-Seg...")

final_images_train = os.path.join(DATASET_DIR, "images", "train")
final_images_val = os.path.join(DATASET_DIR, "images", "val")
final_labels_train = os.path.join(DATASET_DIR, "labels", "train")
final_labels_val = os.path.join(DATASET_DIR, "labels", "val")

os.makedirs(final_images_train, exist_ok=True)
os.makedirs(final_images_val, exist_ok=True)
os.makedirs(final_labels_train, exist_ok=True)
os.makedirs(final_labels_val, exist_ok=True)

def move_files(src_dir, dst_dir):
    if not os.path.exists(src_dir): return
    for f in os.listdir(src_dir):
        shutil.move(os.path.join(src_dir, f), os.path.join(dst_dir, f))

move_files(os.path.join(DATASET_DIR, "train", "images"), final_images_train)
move_files(os.path.join(DATASET_DIR, "train", "labels"), final_labels_train)
move_files(os.path.join(DATASET_DIR, "valid", "images"), final_images_val)
move_files(os.path.join(DATASET_DIR, "valid", "labels"), final_labels_val)

shutil.rmtree(os.path.join(DATASET_DIR, "train"))
shutil.rmtree(os.path.join(DATASET_DIR, "valid"))

# Final YAML
yaml_data = {
    "path": os.path.abspath(DATASET_DIR),
    "train": "images/train",
    "val": "images/val",
    "names": {0: dataset_class_name}
}

with open(os.path.join(DATASET_DIR, "data.yaml"), "w") as f:
    yaml.dump(yaml_data, f, sort_keys=False)

print(f"Done! Dataset ready at {DATASET_DIR}")
    annotations_map[image_name] = results

# Create Supervision Dataset
dataset = sv.DetectionDataset(
    classes=[dataset_class_name],
    images=images_map,
    annotations=annotations_map
)

# Split and Save
train_dataset, val_dataset = dataset.split(split_ratio=0.8)

def save_split(sv_dataset, split_name):
    split_dir = os.path.join(DATASET_DIR, split_name)
    sv_dataset.as_yolo(
        images_directory_path=os.path.join(split_dir, "images"),
        annotations_directory_path=os.path.join(split_dir, "labels"),
        data_yaml_path=os.path.join(split_dir, "data.yaml")
    )

if os.path.exists(DATASET_DIR): shutil.rmtree(DATASET_DIR)
save_split(train_dataset, "train")
save_split(val_dataset, "valid")

# --- PASS 3: Fix Folder Structure ---
print(f"Refining folder structure for YOLOv8-Seg...")

final_images_train = os.path.join(DATASET_DIR, "images", "train")
final_images_val = os.path.join(DATASET_DIR, "images", "val")
final_labels_train = os.path.join(DATASET_DIR, "labels", "train")
final_labels_val = os.path.join(DATASET_DIR, "labels", "val")

os.makedirs(final_images_train, exist_ok=True)
os.makedirs(final_images_val, exist_ok=True)
os.makedirs(final_labels_train, exist_ok=True)
os.makedirs(final_labels_val, exist_ok=True)

def move_files(src_dir, dst_dir):
    if not os.path.exists(src_dir): return
    for f in os.listdir(src_dir):
        shutil.move(os.path.join(src_dir, f), os.path.join(dst_dir, f))

move_files(os.path.join(DATASET_DIR, "train", "images"), final_images_train)
move_files(os.path.join(DATASET_DIR, "train", "labels"), final_labels_train)
move_files(os.path.join(DATASET_DIR, "valid", "images"), final_images_val)
move_files(os.path.join(DATASET_DIR, "valid", "labels"), final_labels_val)

shutil.rmtree(os.path.join(DATASET_DIR, "train"))
shutil.rmtree(os.path.join(DATASET_DIR, "valid"))

# Final YAML
yaml_data = {
    "path": os.path.abspath(DATASET_DIR),
    "train": "images/train",
    "val": "images/val",
    "names": {0: dataset_class_name}
}

with open(os.path.join(DATASET_DIR, "data.yaml"), "w") as f:
    yaml.dump(yaml_data, f, sort_keys=False)

print(f"Done! Dataset ready at {DATASET_DIR}")
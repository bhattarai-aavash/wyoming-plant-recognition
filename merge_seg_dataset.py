import os
import shutil
import yaml
from tqdm import tqdm

# --- CONFIGURATION ---
# Update these paths to match your actual segmentation dataset folders
DATASETS = [
    ("./aureum_seg_dataset", "Aureum"),       # Class 0
    ("./eleagnus_seg_dataset", "Eleagnus"),   # Class 1
    ("./sepherdia_seg_dataset", "Sepherdia")  # Class 2
]

OUTPUT_DIR = "./combined_seg_dataset"

def merge_segmentation_datasets():
    # 1. Prepare Output Directories
    if os.path.exists(OUTPUT_DIR):
        print(f"Removing existing {OUTPUT_DIR}...")
        shutil.rmtree(OUTPUT_DIR)
    
    # Create standard YOLO structure: output/images/train, output/labels/train, etc.
    for split in ["train", "val"]:
        os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "labels", split), exist_ok=True)

    print(f"Merging {len(DATASETS)} datasets into {OUTPUT_DIR}...")

    # 2. Iterate through each dataset
    for new_class_id, (dataset_path, class_name) in enumerate(DATASETS):
        print(f"\nProcessing '{class_name}' as Class ID {new_class_id}...")
        
        # Check if the dataset exists
        if not os.path.exists(dataset_path):
            print(f"⚠️ WARNING: Could not find {dataset_path}. Skipping.")
            continue

        # We look for 'images/train' and 'images/val' structure
        # (This matches the script we wrote previously)
        for split in ["train", "val"]:
            src_images_dir = os.path.join(dataset_path, "images", split)
            src_labels_dir = os.path.join(dataset_path, "labels", split)

            if not os.path.exists(src_images_dir):
                print(f"  Skipping split '{split}' (folder not found)")
                continue

            # Process files
            image_files = [f for f in os.listdir(src_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for img_file in tqdm(image_files, desc=f"  Copying {split}"):
                # A. Copy Image
                # Prefix filename to prevent overwriting (e.g., Aureum_001.jpg)
                new_filename = f"{class_name}_{img_file}"
                
                shutil.copy2(
                    os.path.join(src_images_dir, img_file),
                    os.path.join(OUTPUT_DIR, "images", split, new_filename)
                )

                # B. Process Label (The important part for Segmentation)
                label_file = os.path.splitext(img_file)[0] + ".txt"
                src_label_path = os.path.join(src_labels_dir, label_file)
                dst_label_path = os.path.join(OUTPUT_DIR, "labels", split, os.path.splitext(new_filename)[0] + ".txt")
                
                if os.path.exists(src_label_path):
                    with open(src_label_path, "r") as f:
                        lines = f.readlines()
                    
                    new_lines = []
                    for line in lines:
                        parts = line.strip().split()
                        if not parts: continue
                        
                        # CHANGE CLASS ID
                        # We replace the first number (old ID) with 'new_class_id'
                        # We keep the rest of the line (the polygon coordinates) exactly same
                        parts[0] = str(new_class_id)
                        new_lines.append(" ".join(parts) + "\n")
                    
                    with open(dst_label_path, "w") as f:
                        f.writelines(new_lines)

    # 3. Create the Combined data.yaml
    print("\nCreating data.yaml...")
    yaml_data = {
        "path": os.path.abspath(OUTPUT_DIR),
        "train": "images/train",
        "val": "images/val",
        "nc": len(DATASETS),
        "names": {i: name for i, (_, name) in enumerate(DATASETS)}
    }
    
    with open(os.path.join(OUTPUT_DIR, "data.yaml"), "w") as f:
        yaml.dump(yaml_data, f, sort_keys=False)

    print(f"Done! Combined Segmentation Dataset ready at: {OUTPUT_DIR}")

if __name__ == "__main__":
    merge_segmentation_datasets()
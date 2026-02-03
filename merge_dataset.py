import argparse
import os
import shutil
import yaml
from tqdm import tqdm


# Standard YOLO folder structure
SPLITS = ["train", "valid", "test"]


def get_split_path(base_dir, split_name):
    # check for 'val' vs 'valid' naming differences
    options = [split_name]
    if split_name == 'valid':
        options.append('val')

    for opt in options:
        path = os.path.join(base_dir, opt)
        if os.path.exists(path):
            return path
    return None


def collect_datasets_from_root(root_path):
    """Given a root folder, return a list of (dataset_path, class_name) for each subdirectory.

    It will include subdirectories that look like datasets (exist and are directories).
    The class_name is the subdirectory name.
    """
    datasets = []
    for entry in sorted(os.listdir(root_path)):
        full = os.path.join(root_path, entry)
        if not os.path.isdir(full):
            continue
        # skip hidden or special dirs
        if entry.startswith('.'):
            continue
        datasets.append((full, entry))
    return datasets


def merge_datasets(datasets, output_dir):
    # 1. Prepare Output Directories
    if os.path.exists(output_dir):
        print(f"Removing existing {output_dir}...")
        shutil.rmtree(output_dir)

    for split in ["train", "val"]:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

    print(f"Merging {len(datasets)} datasets into {output_dir}...")

    # 2. Iterate through each dataset
    for class_id, (dataset_path, class_name) in enumerate(datasets):
        print(f"\nProcessing '{class_name}' as Class ID {class_id}...")

        # Determine if source uses 'valid' or 'val'
        # We map source 'valid'/'val' -> destination 'val'
        src_splits = {
            "train": get_split_path(dataset_path, "train"),
            "val": get_split_path(dataset_path, "valid")
        }

        for split_name, src_split_path in src_splits.items():
            if not src_split_path or not os.path.exists(src_split_path):
                continue

            src_images = os.path.join(src_split_path, "images")
            src_labels = os.path.join(src_split_path, "labels")

            if not os.path.exists(src_images):
                print(f"  Warning: images folder not found for {class_name} at {src_images}. Skipping split.")
                continue

            # Copy and Process Files
            image_files = [f for f in os.listdir(src_images) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            for img_file in tqdm(image_files, desc=f"  Copying {split_name}"):
                # A. Handle Image
                # We prefix the filename to avoid conflicts (e.g. aureum_img001.jpg)
                new_filename = f"{class_name}_{img_file}"

                shutil.copy2(
                    os.path.join(src_images, img_file),
                    os.path.join(output_dir, "images", split_name, new_filename)
                )

                # B. Handle Label
                label_file = os.path.splitext(img_file)[0] + ".txt"
                src_label_path = os.path.join(src_labels, label_file)
                dst_label_path = os.path.join(output_dir, "labels", split_name, f"{class_name}_{label_file}")

                if os.path.exists(src_label_path):
                    with open(src_label_path, "r") as f:
                        lines = f.readlines()

                    new_lines = []
                    for line in lines:
                        parts = line.strip().split()
                        if not parts:
                            continue

                        # CHANGE THE CLASS ID (The first number in the row)
                        # We ignore the old ID (likely 0) and use our new 'class_id'
                        parts[0] = str(class_id)
                        new_lines.append(" ".join(parts) + "\n")

                    with open(dst_label_path, "w") as f:
                        f.writelines(new_lines)

    # 3. Create the Combined data.yaml
    print("\nCreating data.yaml...")
    yaml_data = {
        "path": os.path.abspath(output_dir),  # Optional, sometimes helpful
        "train": "./images/train",
        "val": "./images/val",
        "nc": len(datasets),
        "names": [d[1] for d in datasets]
    }

    with open(os.path.join(output_dir, "data.yaml"), "w") as f:
        yaml.dump(yaml_data, f, sort_keys=False)

    print(f"Done! Combined dataset ready at: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Merge multiple YOLO-style datasets (one per class) into a single multi-class YOLO dataset")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--data-root", "-r", help="Path to a root directory containing subfolders for each class (each subfolder should be a dataset or raw images) ")
    group.add_argument("--datasets", "-d", nargs='+', help="List of dataset paths. Each path should be a dataset folder; class name will be derived from the folder name.")
    parser.add_argument("--output", "-o", default="./combined_dataset", help="Output folder for the merged dataset")
    parser.add_argument("--dataset-root", default='.', help="Root folder where per-class dataset folders live or will be created (default: current dir)")
    parser.add_argument("--visual-root", default='.', help="Root folder where per-class visual preview folders live or will be created (used only with --autolabel)")
    parser.add_argument("--autolabel", action='store_true', help="If provided and per-class dataset folders are missing, run the data_label.py script to generate them from raw images under --data-root")
    parser.add_argument("--label-script", default='./data_label.py', help="Path to the data_label.py script used when --autolabel is enabled")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.data_root:
        if not os.path.exists(args.data_root) or not os.path.isdir(args.data_root):
            raise SystemExit(f"data-root {args.data_root} does not exist or is not a directory")

        # Discover class folders under data_root
        class_entries = [e for e in sorted(os.listdir(args.data_root)) if os.path.isdir(os.path.join(args.data_root, e)) and not e.startswith('.')]

        # For each class we expect a per-class dataset folder under dataset_root named <ClassName>_dataset
        datasets = []
        missing = []
        for entry in class_entries:
            expected = os.path.join(args.dataset_root, f"{entry}_dataset")
            if os.path.exists(expected) and os.path.isdir(expected):
                datasets.append((expected, entry))
            else:
                missing.append(entry)

        # If any missing and autolabel requested, run the label script to create per-class datasets
        if missing and args.autolabel:
            import subprocess, sys
            print(f"Missing datasets for classes: {missing}. Running label script to generate per-class datasets...")
            try:
                subprocess.run([
                    sys.executable,
                    args.label_script,
                    "--data-root", args.data_root,
                    "--dataset-root", args.dataset_root,
                    "--visual-root", args.visual_root
                ], check=True)
            except subprocess.CalledProcessError as e:
                raise SystemExit(f"Auto-labeling failed: {e}")

            # Re-scan expected dataset folders
            datasets = []
            for entry in class_entries:
                expected = os.path.join(args.dataset_root, f"{entry}_dataset")
                if os.path.exists(expected) and os.path.isdir(expected):
                    datasets.append((expected, entry))
                else:
                    print(f"Warning: dataset folder for class {entry} still missing at {expected}. It will be skipped.")

        DATASETS = datasets
    elif args.datasets:
        DATASETS = []
        for p in args.datasets:
            if not os.path.exists(p) or not os.path.isdir(p):
                raise SystemExit(f"dataset path {p} does not exist or is not a directory")
            DATASETS.append((p, os.path.basename(os.path.normpath(p))))
    else:
        # Default behavior: look for three well-known dataset folders in cwd if they exist
        defaults = [
            ("./aureum_dataset", "Aureum"),
            ("./eleagnus_dataset", "Eleagnus"),
            ("./sepherdia_dataset", "Sepherdia"),
        ]
        DATASETS = [d for d in defaults if os.path.exists(d[0])]
        if not DATASETS:
            raise SystemExit("No datasets provided and no default dataset folders found. Use --data-root or --datasets to specify inputs.")

    merge_datasets(DATASETS, args.output)
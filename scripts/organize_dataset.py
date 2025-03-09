import os
import shutil
import random
import argparse

def organize_dataset(root_dir, train_ratio=0.8):
    """
    Organizes images and label files into YOLO training dataset structure.
    """
    # Define paths
    images_dir = os.path.join(root_dir, "images")
    dataset_dir = os.path.join(root_dir, "dataset")
    train_images_dir = os.path.join(dataset_dir, "images", "train")
    val_images_dir = os.path.join(dataset_dir, "images", "val")
    train_labels_dir = os.path.join(dataset_dir, "labels", "train")
    val_labels_dir = os.path.join(dataset_dir, "labels", "val")
    
    # Create dataset folders
    for folder in [train_images_dir, val_images_dir, train_labels_dir, val_labels_dir]:
        os.makedirs(folder, exist_ok=True)
    
    # Read class names from classes.txt
    classes_file = os.path.join(images_dir, "classes.txt")
    if not os.path.exists(classes_file):
        raise FileNotFoundError("classes.txt not found in images folder")
    
    with open(classes_file, "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    
    # Collect image-label pairs
    image_files = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]
    
    data_pairs = []
    for img_file in image_files:
        name, ext = os.path.splitext(img_file)
        label_file = name + ".txt"
        label_path = os.path.join(images_dir, label_file)
        if os.path.exists(label_path):
            data_pairs.append((img_file, label_file))
    
    # Shuffle and split data
    random.shuffle(data_pairs)
    split_idx = int(len(data_pairs) * train_ratio)
    train_pairs = data_pairs[:split_idx]
    val_pairs = data_pairs[split_idx:]
    
    # Move files to corresponding directories
    for img_file, label_file in train_pairs:
        shutil.copy(os.path.join(images_dir, img_file), os.path.join(train_images_dir, img_file))
        shutil.copy(os.path.join(images_dir, label_file), os.path.join(train_labels_dir, label_file))
    
    for img_file, label_file in val_pairs:
        shutil.copy(os.path.join(images_dir, img_file), os.path.join(val_images_dir, img_file))
        shutil.copy(os.path.join(images_dir, label_file), os.path.join(val_labels_dir, label_file))
    
    # Generate dataset.yaml
    dataset_yaml = os.path.join(dataset_dir, "dataset.yaml")
    with open(dataset_yaml, "w") as f:
        f.write(f"""
path: {dataset_dir}
train: images/train
val: images/val
names: {class_names}
""".strip())
    
    print(f"Dataset organized successfully in {dataset_dir}")
    print(f"Train images: {len(train_pairs)}, Validation images: {len(val_pairs)}")
    print(f"Dataset YAML file created at {dataset_yaml}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir", type=str, help="Path to the root directory of the dataset")
    args = parser.parse_args()
    
    organize_dataset(args.root_dir)

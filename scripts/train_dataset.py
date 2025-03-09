import argparse
from ultralytics import YOLO
import yaml
import os

def train_yolo(dataset_yaml, config_yaml):  
    # Convert dataset_yaml to absolute path
    dataset_yaml = os.path.abspath(dataset_yaml)
    
    # Check if YAML files exist
    if not os.path.exists(dataset_yaml):
        raise FileNotFoundError(f"Dataset YAML file not found: {dataset_yaml}")
    if not os.path.exists(config_yaml):
        raise FileNotFoundError(f"Config YAML file not found: {config_yaml}")
    
    # Load dataset YAML
    with open(dataset_yaml, "r") as f:
        dataset_config = yaml.safe_load(f)
    
    if "train" not in dataset_config or "val" not in dataset_config:
        raise KeyError("Dataset YAML must contain 'train' and 'val' paths")
    
    # Ensure paths are relative to the dataset root
    dataset_root =              os.path.dirname(dataset_yaml)
    dataset_config["train"] =   os.path.join(dataset_root, dataset_config["train"])
    dataset_config["val"] =     os.path.join(dataset_root, dataset_config["val"])
    
    # Save modified dataset.yaml with updated relative paths
    modified_dataset_yaml = os.path.join(dataset_root, "modified_dataset.yaml")
    with open(modified_dataset_yaml, "w") as f:
        yaml.dump(dataset_config, f)

    # Load training config YAML
    with open(config_yaml, "r") as f:
        train_config = yaml.safe_load(f)
    
    # Extract required parameters
    model_weights = train_config.get("model_weights")
    epochs =        train_config.get("epochs", 100)
    batch_size =    train_config.get("batch_size", 8)
    img_size =      train_config.get("img_size", 640)
    device = "cuda" if train_config.get("device_cuda", True) else "cpu"
    print(f"---- Configurations ----")
    print(f"model_weights : \t{model_weights}")
    print(f"epochs : \t{epochs}")
    print(f"batch_size : \t{batch_size}")
    print(f"img_size : \t{img_size}")
    print(f"Device : \t{device}")
    # Initialize YOLO model with pre-trained weights
    model = YOLO(model_weights)
    
    # Train the model
    model.train(
        data =      modified_dataset_yaml,
        epochs =    epochs,
        imgsz =     img_size,
        batch =     batch_size,
        device =    device
    )
    
    print("Training completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_yaml", type=str, help="Path to the dataset.yaml file")
    parser.add_argument("config_yaml", type=str, help="Path to the training config YAML file")
    
    args = parser.parse_args()
    
    train_yolo(args.dataset_yaml, args.config_yaml)

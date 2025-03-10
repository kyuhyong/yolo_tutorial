# YOLO TUTORIAL

This repository is prepared to help organize and train your data set for object detaction leveraging **YOLO** as backend. 

## Before begin

### Install ultralytics using pip method

You need to install **ultralytics** following this [Quickstart Guide](https://docs.ultralytics.com/quickstart/) and verify it by entering command below.
```
$ python3 -c "import ultralytics; print(ultralytics.__version__)"
8.3.80
```
Also check if torch is available with cuda enabled by entering
```
$ python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
2.4.1+cu121
True
```
If nothing works, check what returns ```nvidia-smi```, ```nvcc --version``` and check your device or cuda version is supported by the latest ultralytics. 

### Run ultralytics on Edge devices using docker



## Clone this package under workspace folder
```
$ mkdir ~/workspace2;cd ~/workspace2
$ git clone https://github.com/kyuhyong/yolo_tutorial.git
```

## Train model for custom dataset

### Prepare custom dataset

To train custom yolo weight, all images must be saved with labels under datasets.

1. Create a folder **"my_data/images"** under **"datasets"** folder and copy all your images under **"images"** folder.
    ```
    cd yolo_tutorial
    mkdir -p datasets/my_data/images
    ```
2. Label with image labeller like  https://github.com/HumanSignal/labelImg

3. After the labelling is done, the folder should be look like below.

    ```
    ./datasets/my_data/
    ├── images/
    |   ├── classes.txt
    │   ├── img_1.jpg
    │   ├── img_1.txt
    │   ├── img_2.jpg
    │   ├── img_2.txt ...
    ```

### Organize custom dataset

1. Run **organize_yolo_dataset.py** under scripts folder with a path to the root directory of your data.

    ```
    $ python3 scripts/organize_dataset.py datasets/my_data
    ```
2. The script will automatically organize the data as train/val data and generate a **dataset.yaml** file as below.
    ```
    ./datasets/my_data/
    ├── images/
    └── dataset/
        ├── dataset.yaml
        ├── images/
        │   ├── train/
        │   │   ├── img_1.jpg
        │   │   ├── img_2.jpg
        │   └── val/
        │       ├── img_3.jpg
        │       ├── img_4.jpg
        ├── labels/
        │   ├── train/
        │   │   ├── img_1.txt
        │   │   ├── img_2.txt
        │   └── val/
        │       ├── img_3.txt
        │       ├── img_4.txt
    ```

You can find detailed explanation in this [Dataset Guide](
https://docs.ultralytics.com/datasets/detect/#adding-your-own-dataset)

### Modify train.yaml

The **"train.yaml"** file contains configurations for training dataset. 
 
| Item | Explanation | Default |
|:------| :-- | --:| 
| model_weights | Pretrained weight for training.<br>[List of models](https://docs.ultralytics.com/models/) available for ultralytics  | weights/yolov5nu.pt
| epochs | One complete pass through the entire training dataset <br> More epochs can improve learning, but too many might lead to overfitting | 100 |
| batch_size | Subset of the dataset used in one training step. <br> Larger batch size can speed up training but requires more memory <br> Smaller batch size can help models generalize better but may be slower | 8 |
| img_size | Size of image | 640 |
| device_cuda | Wheather gpu is used or not | True


### Train custom dataset

**Run** train_images.py with path to the dataset.yaml and train.yaml as below example
```
$ python3 scripts/train_dataset.py datasets/my_data/dataset/dataset.yaml config/train.yaml
```
If everything was set properly, you can get similar result as below.  
This result is from a laptop with RTX 3070 Ti.
```
100 epochs completed in 0.015 hours.
Optimizer stripped from runs/detect/train3/weights/last.pt, 5.3MB
Optimizer stripped from runs/detect/train3/weights/best.pt, 5.3MB

Validating runs/detect/train3/weights/best.pt...
Ultralytics 8.3.80 🚀 Python-3.8.10 torch-2.4.1+cu121 CUDA:0 (NVIDIA GeForce RTX 3070 Ti Laptop GPU, 7966MiB)
YOLOv5n summary (fused): 84 layers, 2,503,334 parameters, 0 gradients, 7.1 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  m
                   all         12         60      0.996          1      0.995       0.92
                  pass         12         52      0.998          1      0.995      0.929
                  fail          8          8      0.995          1      0.995       0.91
Speed: 0.1ms preprocess, 0.9ms inference, 0.0ms loss, 0.4ms postprocess per image
Results saved to runs/detect/train3
Training completed successfully!
```
As indicated above, all the results will be saved under **"/runs"** with all the reports.

## Deploying inference model

### Export PyTorch model (.pt) optimized for edge devices

The `export.py` script for YOLO models is designed to convert a trained PyTorch model (.pt) into different deployment formats such as:

- **ONNX** (Open Neural Network Exchange)
- **TensorRT** (for NVIDIA devices)
- **CoreML** (for Apple devices)
- **TF.js** (TensorFlow.js for web apps)

https://docs.ultralytics.com/modes/export/

Key Parameters Explained
- weights: Path to your trained .pt model file.
- --img_size: The input image size. Keep it consistent with your training size (e.g., 640x640).
- --format: Specifies the export format:
    - onnx: Best for cross-platform deployment.
    - engine: Optimized for TensorRT inference (**recommended** for NVIDIA devices).
    - torchscript: Efficient format for PyTorch inference.
    - coreml: For Apple’s CoreML models.
- --dynamic: Enables dynamic batch size in the exported model (recommended for scalable inference).
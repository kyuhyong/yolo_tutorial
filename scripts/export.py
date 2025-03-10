import argparse
from ultralytics import YOLO

def export_model(weights, img_size=640, format='onnx', dynamic=False):
    """
    Exports a YOLO model to various deployment formats.

    Args:
        weights (str): Path to the trained .pt model.
        img_size (int): Image size for export (default 640).
        format (str): Export format — options: ['onnx', 'engine', 'torchscript', 'coreml'].
        dynamic (bool): Enable dynamic batch size for ONNX models (default: False).
    """

    # Initialize YOLO model
    model = YOLO(weights)

    # Export the model in the desired format
    model.export(format=format, imgsz=img_size, dynamic=dynamic)

    print(f"Model successfully exported to {format} format!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Model Export Script")
    parser.add_argument("weights", type=str, help="Path to the trained .pt model file")
    parser.add_argument("--img_size", type=int, default=640, help="Image size for export")
    parser.add_argument("--format", type=str, default='onnx', choices=['onnx', 'engine', 'torchscript', 'coreml'],
                        help="Export format (onnx, engine, torchscript, coreml)")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic batch size for ONNX export")

    args = parser.parse_args()

    export_model(args.weights, args.img_size, args.format, args.dynamic)
import argparse
import os
from ultralytics import YOLO
from config import MODEL_PATH, CONFIDENCE_THRESHOLD, DEVICE

# Load the YOLOv8 model.
def load_model(model_path=MODEL_PATH):
    model = YOLO(model_path)
    model.conf = CONFIDENCE_THRESHOLD  # set confidence threshold if needed
    return model

# Validate and return the input file path.
def load_source(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Source file '{path}' not found.")
    return path

# Run inference and return results.
def detect_objects(model, source):
    results = model(source, device=DEVICE)
    return results

# Parse and print detection results.
def print_results(results, class_names):
    for result in results:
        print("\nDetected Objects:")
        boxes = result.boxes
        if boxes is None:
            print("No objects detected.")
            continue
        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = class_names.get(cls_id, f"class_{cls_id}")
            print(f"Object {i+1}: Class = {name}, Confidence = {conf:.2f}")

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Object Detection Pipeline")
    parser.add_argument('--source', type=str, required=True, help='Path to image or video')
    parser.add_argument('--model', type=str, default=MODEL_PATH, help='Path to YOLOv8 model weights')
    args = parser.parse_args()

    model = load_model(args.model)
    source = load_source(args.source)
    results = detect_objects(model, source)
    print_results(results, model.names)

if __name__ == "__main__":
    main()

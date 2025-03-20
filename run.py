import argparse
from ultralytics import YOLO
import cv2
import torch

# Argument parser for command-line execution
def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLOv8 object detection on an image.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--model", type=str, default="best.pt", help="Path to the YOLOv8 model file.")
    parser.add_argument("--output", type=str, default="output.jpg", help="Path to save the output image.")
    return parser.parse_args()

# Function to run YOLOv8 detection
def run_yolo(image_path, model_path, output_path):
    # Load YOLO model
    model = YOLO(model_path)
    
    # Run inference
    results = model(image_path)

    # Save output image with detections
    for result in results:
        img = result.plot()  # Draw bounding boxes
        cv2.imwrite(output_path, img)
        print(f"✅ Detection completed! Output saved at: {output_path}")

if __name__ == "__main__":
    args = parse_args()
    run_yolo(args.image, args.model, args.output)

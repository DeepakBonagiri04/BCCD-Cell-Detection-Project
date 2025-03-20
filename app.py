import gradio as gr
from ultralytics import YOLO

# Load YOLOv8 Model
model = YOLO("best.pt")

def detect(image):
    results = model(image)
    img = results[0].plot()
    return img

# Create Gradio Interface
iface = gr.Interface(fn=detect, inputs="image", outputs="image",
                     title="BCCD Cell Detection",
                     description="Upload an image to detect cells using YOLOv8.")

iface.launch(share=True)


from transformers import pipeline
from PIL import Image, ImageDraw
import requests

# Load the image
image_path = '../data/past_papers/2025_specimen/paper_1/663662-2025-specimen-paper-1_pages-to-jpg-0003.jpg' # MAKE SURE THIS FILE EXISTS
try:
    image = Image.open(image_path).convert("RGB")
except FileNotFoundError:
    print(f"Error: Image file not found at {image_path}")
    exit()

# Initialize the object detection pipeline with LayoutLMv3
# microsoft/layoutlmv3-base-finetuned-publaynet is good for general document layout
# It identifies 'text', 'title', 'list', 'table', 'figure'
object_detector = pipeline("object-detection", model="microsoft/dit-base-finetuned-rvlcdip")

# Get predictions
predictions = object_detector(image)

print("Detected objects:", predictions)

# Filter for 'figure' labels and extract them
diagram_crops = []
for i, pred in enumerate(predictions):
    if pred['label'] == 'figure': # PubLayNet uses 'figure'
        box = pred['box']
        cropped_diagram = image.crop((box['xmin'], box['ymin'], box['xmax'], box['ymax']))
        diagram_crops.append(cropped_diagram)

        save_path = f"extracted_figure_layoutlmv3_{i}.png"
        cropped_diagram.save(save_path)
        print(f"Saved extracted figure to {save_path}")
        # cropped_diagram.show()

if not diagram_crops:
    print("No 'figure' elements detected by LayoutLMv3.")

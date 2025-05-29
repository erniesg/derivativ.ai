from transformers import DetrImageProcessor, DetrForObjectDetection # Common for DETR-like models
from transformers import AutoImageProcessor, AutoModelForObjectDetection # More generic
from PIL import Image
import torch # DiT models are PyTorch based

# Load the image
image_path = '../data/past_papers/2025_specimen/paper_1/663662-2025-specimen-paper-1_pages-to-jpg-0003.jpg' # MAKE SURE THIS FILE EXISTS
try:
    image = Image.open(image_path).convert("RGB")
except FileNotFoundError:
    print(f"Error: Image file not found at {image_path}")
    exit()

# Use the correct model for layout detection (object detection)
# microsoft/dit-base-finetuned-publaynet is for layout detection
model_name = "microsoft/layoutlmv3-large"

# Load the processor and model explicitly
try:
    # For DiT models fine-tuned on object detection tasks like PubLayNet,
    # the processor might still be based on DETR's image processor.
    # Let's try AutoImageProcessor first for flexibility.
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForObjectDetection.from_pretrained(model_name)
except Exception as e:
    print(f"Error loading model/processor automatically: {e}")
    print("Attempting with DetrImageProcessor/DetrForObjectDetection as a fallback for DETR-like architectures...")
    try:
        # If AutoImageProcessor fails or doesn't have the right methods,
        # try DetrImageProcessor as DiT's object detection often uses DETR-style heads
        image_processor = DetrImageProcessor.from_pretrained(model_name)
        model = DetrForObjectDetection.from_pretrained(model_name)
    except Exception as e2:
        print(f"Error loading with Detr specific classes: {e2}")
        print("Please ensure the model name is correct and supports object detection.")
        exit()


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)
print(f"Device set to use {device}")


# Prepare image for model
inputs = image_processor(images=image, return_tensors="pt").to(device)

# Get predictions
outputs = model(**inputs)

# Convert outputs to COCO API format (common for object detection)
# The post_process_object_detection method expects target_sizes
target_sizes = torch.tensor([image.size[::-1]], device=device) # (height, width)
results = image_processor.post_process_object_detection(outputs, threshold=0.7, target_sizes=target_sizes)[0] # Adjust threshold as needed

print("Detected objects:", results)

# Filter for 'figure' labels and extract them
# PubLayNet labels include 'text', 'title', 'list', 'table', 'figure'
diagram_crops = []
for i, (score, label, box) in enumerate(zip(results["scores"], results["labels"], results["boxes"])):
    # model.config.id2label gives mapping from label ID to label name
    label_name = model.config.id2label[label.item()]
    print(f"Detected: {label_name} with confidence {score.item()}")
    if label_name == 'figure':
        box = [round(coord) for coord in box.tolist()]
        cropped_diagram = image.crop(box)
        diagram_crops.append(cropped_diagram)

        save_path = f"extracted_figure_dit_publaynet_{i}.png"
        cropped_diagram.save(save_path)
        print(f"Saved extracted figure to {save_path}")
        # cropped_diagram.show()

if not diagram_crops:
    print("No 'figure' elements detected by the model.")

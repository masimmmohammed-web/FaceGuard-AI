from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import torch

model_name = "prithivMLmods/Deepfake-vs-Real-8000"

processor = AutoImageProcessor.from_pretrained(model_name)
model = SiglipForImageClassification.from_pretrained(model_name)

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1).squeeze().tolist()
    labels = {0: "Deepfake", 1: "Real one"}
    return {labels[i]: round(probs[i], 3) for i in range(len(probs))}

# Example usage:
print(predict("images/masi.jpeg"))

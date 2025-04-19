import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import os
from torchvision import transforms

class ImageDetector(nn.Module):
    def __init__(self):
        super(ImageDetector, self).__init__()
        self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        self.model = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224',
            num_labels=2,
            ignore_mismatched_sizes=True  # Handle mismatched classifier sizes
        )
        self.model.classifier = nn.Linear(self.model.config.hidden_size, 2)  # 2 classes: Real, Fake
        # Load pre-trained weights with key adjustment
        weights_path = 'weights/vit_image_model.pth'
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location='cpu')
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('model.'):
                    new_state_dict[k[len('model.'):]] = v
                else:
                    new_state_dict[k] = v
            self.model.load_state_dict(new_state_dict, strict=False)  # Allow partial loading
        self.model.eval()

    def forward(self, pixel_values=None, **kwargs):
        """Forward pass accepting pixel_values as a keyword argument."""
        if pixel_values is None:
            raise ValueError("pixel_values must be provided")
        inputs = {"pixel_values": pixel_values}
        return self.model(**inputs)

# Initialize model
model = ImageDetector()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define transform without normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()  # Convert to tensor [0, 1], normalization handled by processor
])

# Image preprocessing
def preprocess_image(img_path):
    """Preprocess image from file path."""
    try:
        img = Image.open(img_path).convert('RGB')
        return img  # Return PIL image for processor
    except Exception as e:
        raise Exception(f"Image preprocessing failed: {str(e)}")

def detect_image(filepath):
    """Detect if an image is real or fake."""
    try:
        img = preprocess_image(filepath)
        inputs = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')(images=img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            output = model(**inputs)  # Pass the full inputs dictionary
            probs = torch.softmax(output.logits, dim=1)  # Access logits attribute
            confidence, predicted = torch.max(probs, 1)
            result = 'Fake' if predicted.item() == 1 else 'Real'
            confidence = confidence.item()

        return result, confidence
    except Exception as e:
        raise Exception(f"Image detection failed: {str(e)}")

# Expose transform, model, and device for import
__all__ = ['transform', 'model', 'device', 'detect_image']

if __name__ == '__main__':
    test_path = 'path/to/test/image.jpg'
    try:
        result, confidence = detect_image(test_path)
        print(f"Result: {result}, Confidence: {confidence:.2%}")
    except Exception as e:
        print(f"Error: {str(e)}")
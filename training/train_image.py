
# used this code in google colab for faster training the datasets are provided in the project directory

# Mount Drive and set paths
from google.colab import drive
drive.mount('/content/drive')
data_dir = '/content/drive/MyDrive/datasets/ciplab'
weights_path = '/content/drive/MyDrive/weights/vit_image_model.pth'

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from transformers import ViTForImageClassification, ViTFeatureExtractor
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Ensure GPU usage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize model and feature extractor with ignore_mismatched_sizes
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224',
    num_labels=2,
    ignore_mismatched_sizes=True  # Skip mismatched classifier sizes
)
model.classifier = nn.Linear(model.config.hidden_size, 2)  # Reinitialize classifier for 2 classes
model.to(device)

# Check dataset balance
real_count = len([f for f in os.listdir(os.path.join(data_dir, 'real')) if f.endswith(('.jpg', '.png'))])
fake_count = len([f for f in os.listdir(os.path.join(data_dir, 'fake')) if f.endswith(('.jpg', '.png'))])
print(f"Dataset Balance - Real: {real_count}, Fake: {fake_count}")
total_samples = real_count + fake_count

# Calculate class weights
weight_for_0 = (1 / real_count) * (total_samples) / 2.0
weight_for_1 = (1 / fake_count) * (total_samples) / 2.0
class_weights = torch.FloatTensor([weight_for_0, weight_for_1]).to(device)
print(f"Class weights - Real (0): {weight_for_0:.2f}, Fake (1): {weight_for_1:.2f}")

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Custom Dataset
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        for label, subdir in enumerate(['real', 'fake']):
            subdir_path = os.path.join(data_dir, subdir)
            for fname in os.listdir(subdir_path):
                if fname.endswith(('.jpg', '.png')):
                    self.images.append(os.path.join(subdir_path, fname))
                    self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

dataset = ImageDataset(data_dir, transform=lambda x: feature_extractor(x, return_tensors="pt")['pixel_values'][0])
print(f"Dataset size: {len(dataset)} images")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, num_workers=2)

# Fine-tune all parameters
for param in model.parameters():
    param.requires_grad = True

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Create weights directory if it doesn't exist
os.makedirs('/content/drive/MyDrive/weights', exist_ok=True)

best_acc = 0.0
for epoch in range(15):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/15'):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    model.eval()
    correct = 0
    total = 0
    val_labels = []
    val_preds = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs).logits
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            val_preds.extend(predicted.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_acc = correct / total
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}, Val Acc: {val_acc:.4f}')

    cm = confusion_matrix(val_labels, val_preds)
    print("Confusion Matrix:\n", cm)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), weights_path)
        print(f'Saved best model with Val Acc: {best_acc:.4f}')

torch.save(model.state_dict(), weights_path)
print("Training completed. Weights saved to Google Drive.")
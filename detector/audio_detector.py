# detector/audio_detector.py
import torch
import torch.nn as nn
import librosa
import numpy as np
from torchvision import transforms
import os

class AudioDetector(nn.Module):
    def __init__(self):
        super(AudioDetector, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(32 * 16 * 32, 2)  # Corrected: 32*16*32 = 16384

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = AudioDetector()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

weights_path = 'weights/audio_model.pth'
if os.path.exists(weights_path):
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Loaded weights from {weights_path}")
    except Exception as e:
        print(f"Failed to load weights from {weights_path}: {str(e)}")
model.eval()

def detect_audio(filepath):
    try:
        y, sr = librosa.load(filepath, sr=16000)
        spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, hop_length=512)
        spec_db = librosa.power_to_db(spec, ref=np.max)
        
        spec_tensor = torch.tensor(spec_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        spec_tensor = torch.nn.functional.interpolate(spec_tensor, size=(64, 128), mode='bilinear')
        spec_tensor = spec_tensor.to(device)
        
        with torch.no_grad():
            output = model(spec_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probs, 1)
            result = 'Fake' if predicted.item() == 1 else 'Real'
            confidence = confidence.item()

        return result, confidence
    except Exception as e:
        raise Exception(f"Audio detection failed: {str(e)}")
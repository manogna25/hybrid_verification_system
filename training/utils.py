# training/utils.py
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import librosa
import numpy as np
from torchvision import transforms

class ImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """Custom dataset for image deepfake detection."""
        self.transform = transform
        self.images = []
        self.labels = []
        
        for label, subdir in enumerate(['real', 'fake']):
            subdir_path = os.path.join(data_dir, subdir)
            if os.path.exists(subdir_path):
                for fname in os.listdir(subdir_path):
                    if fname.lower().endswith(('.jpg', '.png')):
                        self.images.append(os.path.join(subdir_path, fname))
                        self.labels.append(label)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, label

class AudioDataset(Dataset):
    def __init__(self, data_dir):
        """Custom dataset for audio deepfake detection."""
        self.audios = []
        self.labels = []
        
        for label, subdir in enumerate(['real', 'fake']):
            subdir_path = os.path.join(data_dir, subdir)
            if os.path.exists(subdir_path):
                for fname in os.listdir(subdir_path):
                    if fname.lower().endswith(('.wav', '.mp3')):
                        self.audios.append(os.path.join(subdir_path, fname))
                        self.labels.append(label)
    
    def __len__(self):
        return len(self.audios)
    
    def __getitem__(self, idx):
        audio_path = self.audios[idx]
        label = self.labels[idx]
        
        y, sr = librosa.load(audio_path, sr=16000)
        spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, hop_length=512)
        spec_db = librosa.power_to_db(spec, ref=np.max)
        
        spec_tensor = torch.tensor(spec_db, dtype=torch.float32).unsqueeze(0)
        spec_tensor = torch.nn.functional.interpolate(spec_tensor.unsqueeze(0), size=(64, 128), mode='bilinear').squeeze(0)
        
        return spec_tensor, label
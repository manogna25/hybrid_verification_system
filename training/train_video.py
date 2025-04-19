# used this code in google colab for faster training the datasets are provided in the project directory

# Mount Drive and set paths
from google.colab import drive
drive.mount('/content/drive')
data_dir = '/content/drive/MyDrive/datasets/videos'
model_path = '/content/drive/MyDrive/weights/svm_video_model.pth'

# Import required libraries
import os
import cv2
import numpy as np
import torch  # Added missing import
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import joblib

# Ensure GPU usage (SVM doesn't use GPU, but feature extraction benefits from CPU optimization)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Feature extraction function
def extract_optical_flow_features(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_features = []
    frame_count = 0
    max_frames = 30

    ret, prev_frame = cap.read()
    if not ret:
        return None

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count > 0:
            gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            features = np.concatenate([magnitude.flatten(), angle.flatten()])
            frame_features.append(features)
        prev_frame = frame
        frame_count += 1

    cap.release()
    return np.mean(frame_features, axis=0) if frame_features else None

# Collect features and labels
X, y = [], []
for label, subdir in enumerate(['real', 'fake']):
    subdir_path = os.path.join(data_dir, subdir)
    for fname in tqdm(os.listdir(subdir_path), desc=f"Processing {subdir}"):
        if fname.endswith(('.mp4', '.avi')):
            features = extract_optical_flow_features(os.path.join(subdir_path, fname))
            if features is not None:
                X.append(features)
                y.append(label)

if not X or not y:
    raise ValueError("No video features extracted")

# Preprocess features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM
svm = SVC(kernel='rbf', probability=True)
svm.fit(X_train, y_train)

# Evaluate
accuracy = svm.score(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# Save model
os.makedirs('/content/drive/MyDrive/weights', exist_ok=True)
joblib.dump(svm, model_path)
print(f"Saved SVM model to {model_path}")
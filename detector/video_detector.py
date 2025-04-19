import cv2
import numpy as np
import torch
from sklearn.svm import SVC
import joblib
import os
from PIL import Image
from detector.image_detector import transform, model, device  # Import shared objects

class VideoDetector:
    def __init__(self):
        self.model = joblib.load('weights/svm_video_model.pth')
        self.scaler = joblib.load('weights/svm_scaler.pth') if os.path.exists('weights/svm_scaler.pth') else None
        self.expected_features = 25088  # Match the trained model's input dimension

    def extract_frames(self, video_path):
        """Extract and preprocess frames from video."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        try:
            while cap.isOpened() and len(frames) < 30:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                frames.append(transform(frame_pil))  # Apply 224x224 transform
        except Exception as e:
            print(f"Error extracting frames: {str(e)}")
            raise
        finally:
            cap.release()

        return torch.stack(frames) if frames else None

    def extract_optical_flow(self, frames):
        """Extract Optical Flow features from frames."""
        frame_features = []
        for i in range(1, len(frames)):
            # Convert tensor back to uint8 array with 224x224 resolution
            gray1 = cv2.cvtColor((frames[i-1].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor((frames[i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            features = np.concatenate([magnitude.flatten(), angle.flatten()])
            print(f"Frame pair {i} features shape: {features.shape}")  # Debug feature count
            frame_features.append(features[:self.expected_features // (len(frames)-1)])  # Truncate to match expected size
        return np.mean(frame_features, axis=0) if frame_features else None

    def detect_video(self, video_path):
        """Detect if a video is real or fake."""
        try:
            frames = self.extract_frames(video_path)
            if frames is None:
                raise Exception("No frames extracted")
            features = self.extract_optical_flow(frames)
            if features is None:
                raise Exception("No optical flow features extracted")
            if len(features) != self.expected_features:
                print(f"Warning: Reshaping features from {len(features)} to {self.expected_features}")
                features = np.resize(features, (1, self.expected_features))  # Force reshape
            elif self.scaler is not None:
                features = self.scaler.transform([features])
            else:
                features = features.reshape(1, -1)
            prediction = self.model.predict_proba(features)
            confidence = prediction[0][1]  # Probability of Fake
            result = 'Fake' if confidence > 0.5 else 'Real'
            confidence = confidence if result == 'Fake' else 1 - confidence
            return result, confidence
        except Exception as e:
            raise Exception(f"Video detection failed: {str(e)}")

if __name__ == '__main__':
    detector = VideoDetector()
    result, confidence = detector.detect_video('path/to/test/video.mp4')
    print(f"Result: {result}, Confidence: {confidence:.2%}")
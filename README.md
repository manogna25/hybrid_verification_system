# Hybrid Deepfake Detector

A multimodal deepfake detection system that analyzes images, videos, and audio to classify content as "Real" or "Fake." This project integrates advanced machine learning techniques and is designed for educational and research purposes.

## Overview
- **Image Detection**: Utilizes a Vision Transformer (ViT) fine-tuned for binary classification.
- for this image detection download the https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection
- **Video Detection**: Employs Optical Flow features with a Support Vector Machine (SVM) for temporal analysis.
- for this video detection download the https://www.kaggle.com/datasets/ashifurrahman34/dfdc-dataset
- **Audio Detection**: Uses a Convolutional Neural Network (CNN) on Mel Spectrograms to detect synthetic speech.
- for this video detection download the https://www.kaggle.com/datasets/birdy654/deep-voice-deepfake-voice-recognition?select=KAGGLE
- **Deployment**: Hosted via Flask, allowing file uploads and result visualization.

## Features
- Supports `.jpg`, `.jpeg`, `.png` (images), `.mp4`, `.avi` (videos), and `.mp3`, `.wav` (audio).
- Returns classification results with confidence scores.
- Maximum upload size: 100MB.
## code used for training image and video models 
   in train_image.py and train_video.py


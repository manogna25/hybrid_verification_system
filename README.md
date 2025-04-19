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

## To Excecute
### clone this repo
   git clone https://github.com/manogna25/hybrid_verification_system.git
   cd hybrid_verification_system
### Create and Activate Virtual Environment

   ### Create venv (Windows)
   python -m venv venv
   
   ### Activate venv (Windows)
   venv\Scripts\activate
### intall dependencies
   pip install -r requirements.txt
### train
   download audio dataset into folder datasets which contains real, fake folders
   train using python training/train_audio.py ->which creates weights folder with .pth file
   for image and video 
   download the datasets and place them in your google drive, use google collab to train, use runtime gpu for faster process
   add the dowloaded .pth files of image and video in weights folder
### run
   python app.py
   
   Running on http://127.0.0.1:5000/

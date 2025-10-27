# ML Models Inventory

## Overview
This repository contains Machine Learning models for **Gender and Age Detection** using Computer Vision.

## Models Present in Repository

### 1. Face Detection Model
**Location:** `ML MODELS/`
- **Model Files:**
  - `opencv_face_detector_uint8.pb` (2.7 MB) - Pre-trained TensorFlow model
  - `opencv_face_detector.pbtxt` (35 KB) - Model configuration file
- **Purpose:** Detects faces in images or video frames
- **Framework:** OpenCV DNN module with TensorFlow backend
- **Input:** Images of size 300x300
- **Architecture:** Deep Neural Network for face detection
- **Confidence Threshold:** 0.7 (default)

### 2. Age Prediction Model
**Location:** `ML MODELS/`
- **Model Files:**
  - `age_net.caffemodel` (Expected, referenced in code but not present in repo)
  - `age_deploy.prototxt` (2.3 KB) - Caffe model architecture definition
- **Purpose:** Predicts age range of detected faces
- **Framework:** Caffe (using OpenCV DNN module)
- **Architecture:** CaffeNet-based Convolutional Neural Network
  - Input dimensions: 1 x 3 x 227 x 227
  - Layers: Convolution, ReLU, Pooling, Fully Connected
  - First conv layer: 96 filters, 7x7 kernel, stride 4
- **Age Categories:** 8 ranges
  - (0-2), (4-6), (8-12), (15-20), (25-32), (38-43), (48-53), (60-100)
- **Mean Values:** RGB(78.43, 87.77, 114.90)

### 3. Gender Classification Model
**Location:** `ML MODELS/`
- **Model Files:**
  - `gender_net.caffemodel` (Expected, referenced in code but not present in repo)
  - `gender_deploy.prototxt` (2.3 KB) - Caffe model architecture definition
- **Purpose:** Classifies gender of detected faces
- **Framework:** Caffe (using OpenCV DNN module)
- **Architecture:** CaffeNet-based Convolutional Neural Network
  - Input dimensions: 10 x 3 x 227 x 227
  - Layers: Convolution, ReLU, Pooling, Fully Connected
  - First conv layer: 96 filters, 7x7 kernel, stride 4
- **Classes:** Binary classification
  - Male
  - Female
- **Mean Values:** RGB(78.43, 87.77, 114.90)

## Application: Gender and Age Detection (gad.py)

### Description
The `gad.py` script implements a real-time gender and age detection system that:
1. Captures video from webcam or processes an input image
2. Detects faces using OpenCV's face detector
3. Predicts gender and age for each detected face
4. Displays results with bounding boxes and text overlays

### Usage
```bash
# Using webcam
python gad.py

# Using an image file
python gad.py --image path/to/image.jpg
```

### Technical Details
- **Input Processing:** Blob creation with size 300x300 for face detection, 227x227 for age/gender
- **Face Padding:** 20 pixels around detected face for better context
- **Visualization:** Green bounding boxes with cyan text showing gender and age
- **Real-time Processing:** Processes video frames continuously until interrupted

## Model Architecture Summary

### CaffeNet Architecture (Age & Gender Models)
Both age and gender models use variants of the CaffeNet architecture:
- **Convolutional Layers:** Extract spatial features from face images
- **ReLU Activation:** Non-linear activation function
- **Max Pooling:** Reduces spatial dimensions and computational complexity
- **Fully Connected Layers:** Final classification layers
- **Input Size:** 227x227x3 (RGB images)

### Face Detection Network
- **Architecture:** OpenCV DNN-based face detector
- **Type:** Single Shot Detector (SSD) variant
- **Input Size:** 300x300
- **Output:** Face bounding boxes with confidence scores

## Missing Model Files

⚠️ **Note:** The following model weight files are referenced in the code but not present in the repository:
- `age_net.caffemodel` - Age prediction model weights
- `gender_net.caffemodel` - Gender classification model weights

These files would need to be downloaded separately to run the application. They are typically available from:
- OpenCV's official model zoo
- Pre-trained model repositories
- Original research papers/implementations

## Technologies Used
- **OpenCV (cv2):** Computer vision library for image processing and DNN inference
- **Caffe Models:** Deep learning framework for age and gender prediction
- **TensorFlow:** Backend for face detection model
- **Python 3:** Programming language

## Performance Considerations
- Face detection confidence threshold set to 0.7 for balanced precision/recall
- Real-time processing capability on standard hardware
- GPU acceleration possible with OpenCV DNN CUDA backend

## Potential Improvements
1. Add the missing .caffemodel files to the repository
2. Include sample images for testing
3. Add requirements.txt for dependency management
4. Implement batch processing for multiple images
5. Add model performance metrics and accuracy benchmarks
6. Consider upgrading to more recent model architectures (e.g., ONNX format)
7. Add error handling for missing model files

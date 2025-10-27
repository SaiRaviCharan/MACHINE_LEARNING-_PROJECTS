# Gender and Age Detection (GAD)

## Overview
This directory contains pre-trained models and application code for real-time gender and age detection using computer vision.

## Files in This Directory

### Application Code
- **gad.py** - Main Python script for gender and age detection

### Model Files

#### Face Detection
- `opencv_face_detector_uint8.pb` (2.7 MB) - TensorFlow model for face detection
- `opencv_face_detector.pbtxt` (35 KB) - Model configuration

#### Age Prediction
- `age_deploy.prototxt` (2.3 KB) - Caffe model architecture for age prediction
- `age_net.caffemodel` ⚠️ **Missing** - Model weights (not included in repo)

#### Gender Classification
- `gender_deploy.prototxt` (2.3 KB) - Caffe model architecture for gender classification
- `gender_net.caffemodel` ⚠️ **Missing** - Model weights (not included in repo)

## How to Use

### Prerequisites
```bash
pip install opencv-python opencv-contrib-python numpy
```

### Running the Application

**Option 1: Using Webcam**
```bash
python gad.py
```

**Option 2: Using an Image File**
```bash
python gad.py --image path/to/your/image.jpg
```

**Option 3: Using a Video File**
```bash
python gad.py --image path/to/your/video.mp4
```

### Controls
- Press any key to exit the application

## Model Details

### Face Detection Model
- **Input Size:** 300 x 300 pixels
- **Confidence Threshold:** 0.7
- **Output:** Bounding boxes of detected faces

### Age Prediction Model
- **Architecture:** CaffeNet-based CNN
- **Input Size:** 227 x 227 pixels
- **Age Ranges:** 8 categories
  - (0-2) years
  - (4-6) years
  - (8-12) years
  - (15-20) years
  - (25-32) years
  - (38-43) years
  - (48-53) years
  - (60-100) years

### Gender Classification Model
- **Architecture:** CaffeNet-based CNN
- **Input Size:** 227 x 227 pixels
- **Classes:** Male, Female

## Missing Model Files

To run the application, you need to download the following model files:
- `age_net.caffemodel` (~23 MB)
- `gender_net.caffemodel` (~44 MB)

These can typically be obtained from:
- [OpenCV's GitHub repository](https://github.com/opencv/opencv/tree/master/samples/dnn)
- [GilLevi/AgeGenderDeepLearning](https://github.com/GilLevi/AgeGenderDeepLearning)

## How It Works

1. **Face Detection:** The script first detects faces in the input frame using OpenCV's DNN face detector
2. **Face Extraction:** Each detected face is extracted with padding for better context
3. **Preprocessing:** Faces are resized to 227x227 and normalized with mean values
4. **Gender Prediction:** The gender model classifies each face as Male or Female
5. **Age Prediction:** The age model predicts the age range for each face
6. **Visualization:** Results are displayed on the video feed with bounding boxes and text labels

## Technical Specifications

### Image Preprocessing
- **Mean RGB Values:** (78.43, 87.77, 114.90)
- **Face Padding:** 20 pixels
- **Color Space:** BGR (OpenCV default)

### Model Inference
- **Framework:** OpenCV DNN module
- **Backend Options:** OpenCV, OpenCL, Halide, Inference Engine
- **Target Devices:** CPU, GPU (CUDA), OpenCL, VPU

## Example Output

The application will display:
- Green bounding boxes around detected faces
- Text overlay showing: `Male, 25-32` or `Female, 15-20`
- Console output with detected gender and age

## Limitations

1. **Missing Model Weights:** The .caffemodel files are not included in this repository
2. **Binary Gender Classification:** Only Male/Female categories
3. **Age Range Predictions:** Provides ranges rather than exact ages
4. **Lighting Conditions:** Performance may vary with poor lighting
5. **Face Angles:** Works best with frontal or near-frontal faces

## Future Enhancements

- [ ] Include the missing .caffemodel files
- [ ] Add requirements.txt for easy dependency installation
- [ ] Support for batch image processing
- [ ] Command-line options for confidence threshold adjustment
- [ ] Option to save output images/videos
- [ ] Add model performance metrics
- [ ] Implement face tracking for video streams

## References

- [Age and Gender Classification using Convolutional Neural Networks](https://talhassner.github.io/home/publication/2015_CVPR)
- [OpenCV DNN Module Documentation](https://docs.opencv.org/master/d2/d58/tutorial_table_of_content_dnn.html)

---

For more detailed information about the models, see [ML_MODELS_INVENTORY.md](../ML_MODELS_INVENTORY.md) in the root directory.



# Real-time Facial Emotion Recognition with 68 Facial Keypoints 

## Introduction

This repository contains the source code for a real-time facial emotion recognition application. The application captures video from the laptop camera and recognizes facial expressions as one of seven emotions (e.g., happy, sad, angry, disgust, surprise, fear, neutral) displayed on a person's face. The user can see the facial emotion prediction displayed on their screen in real-time, utilizing a model that leverages 68 facial keypoints for enhanced accuracy.

## Detailed Description

The application follows a structured approach, as illustrated in the diagram below:

### Code Structure Diagram
(Include your code structure diagram here)

### Implementation Steps
Creating this application involves the following steps:

1. **Deep Learning Model Development**: 
   - Build a deep learning model using Keras/TensorFlow in Python. 
   - Train the model using a dataset of facial expressions.
   - Save the model graph and weights as a `.pb` file for deployment.

2. **Real-time Video Capture**: 
   - Capture video using the laptop webcam in real-time with OpenCV.

3. **Face Detection**: 
   - Use the Haar Cascades face detector in OpenCV to detect each face and its location.

4. **Facial Keypoint Detection**: 
   - Extract 68 facial keypoints from the detected face, enhancing the accuracy of emotion recognition.

5. **Region of Interest Processing**: 
   - Draw a rectangular bounding box around the face (region of interest) and preprocess the image.
   - Resize the keypoint data to fit the input expected by the model.

6. **Model Inference**: 
   - Deploy the model using the OpenCV CNN module to perform inference on the image data.

7. **Display Results**: 
   - Display the video feed along with the bounding box and the model's prediction on the screen.


## Rubric Points Satisfied

The project meets the following rubric criteria:

- Demonstrates an understanding of C++ functions and control structures.
- Reads data from files and processes it, or writes data to files.
- Utilizes Object-Oriented Programming techniques.
- Classes use appropriate access specifiers for class members.
- Class constructors utilize member initialization lists.
- Classes abstract implementation details from their interfaces.
- Classes encapsulate behavior.
- Makes use of references in function declarations.

## Dependencies for Running Locally

This project was run and tested on macOS.

- `cmake >= 3.7`
- `make >= 4.1` (Linux, Mac), `3.81` (Windows)
- `gcc/g++ >= 5.4`
- `OpenCV == 4.3.0` (other versions may work but are untested)
- `tensorflow <= 1.15` (for Python notebooks only)

## Basic Build Instructions

1. Clone this repository.
2. Create a build directory in the top-level directory: 
   ```bash
   mkdir build && cd build
   ```
3. Compile the application:
   ```bash
   cmake .. && make
   ```

## Future Work

To enhance the accuracy of user emotion recognition, I plan to integrate more advanced Deep Neural Network (DNN) architectures into the application. This may include experimenting with various pre-trained models and fine-tuning them on the emotion recognition dataset to achieve better performance. Additionally, I aim to implement techniques such as data augmentation and transfer learning, which can further improve the model's robustness against diverse facial expressions and keypoint variations.

Another avenue for future work is to explore real-time optimization strategies, such as model quantization and pruning, to increase inference speed without sacrificing accuracy. This would help in deploying the application on resource-constrained devices while maintaining high performance in emotion recognition tasks.



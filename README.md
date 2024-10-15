# Microplastic Detection in Water Using EfficientNetB0
## Table of Contents
1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Solution Approach](#solution-approach)
4. [Methodology](#methodology)
   - [Dataset Preparation](#dataset-preparation)
   - [Model Architecture](#model-architecture)
   - [Training Process](#training-process)
5. [Results](#results)
6. [Real-World Application](#real-world-application)
7. [Conclusion](#conclusion)

## Project Overview

This project focuses on developing a deep learning model to detect microplastics in water samples. The goal is to create an efficient and accurate system that can distinguish between clean water and water contaminated with microplastics using image processing techniques.

## Problem Statement

Microplastics, which are plastic particles less than 5 mm in size, pose a significant threat to aquatic environments and public health. Their small size makes them difficult to detect using traditional methods. This project aims to address this challenge by leveraging deep learning technology for automated detection of microplastics in water samples.

## Solution Approach

The project utilizes EfficientNetB0, a convolutional neural network (CNN) architecture known for its balance between computational efficiency and accuracy. This model is adapted to classify images of water samples into two categories: clean water and water contaminated with microplastics.

## Methodology

### Dataset Preparation

1. The dataset consists of images divided into two classes: clean water and water with microplastics.
![alt text](https://github.com/AryanDahiya00/Microplastic_Detection/blob/main/Capture1.JPG)
2. Images are organized into three main folders: training, validation, and testing.
3. Data augmentation techniques are applied to the training set, including:
   - Rescaling
   - Rotation
   - Width and height shift
   - Shear
   - Zoom
   - Horizontal flip

### Model Architecture

1. Base model: EfficientNetB0 pre-trained on ImageNet
2. Custom layers added:
   - Flattening layer
   - Dense layer with 1024 units (ReLU activation)
   - Batch normalization
   - Dropout (0.5) for preventing overfitting
   - Final dense layer with sigmoid activation for binary classification
![alt text](https://github.com/AryanDahiya00/Microplastic_Detection/blob/main/Untitled%20Diagram.drawio.png)

### Training Process

1. Optimizer: RMSprop (learning rate: 0.0001, rho: 0.9)
2. Loss function: Binary cross-entropy
3. Metric: Accuracy
4. Training duration: 10 epochs
5. Early stopping implemented with a patience of 3 epochs
6. Best weights restored based on validation loss

## Results

The model achieved impressive results in detecting microplastics:

1. Overall accuracy on test data: 98%
2. Performance metrics:
   - Clean Water class:
     - Precision: 0.98
     - Recall: 0.99
     - F1-score: 0.99
   - Microplastics class:
     - Precision: 0.99
     - Recall: 0.98
     - F1-score: 0.98
3. Confusion Matrix:
   - 99 correct predictions for clean water
   - 98 correct predictions for microplastics
   - Only 1 false positive and 2 false negatives
![alt text](https://github.com/AryanDahiya00/Microplastic_Detection/blob/main/Capture2.JPG)

## Real-World Application

To demonstrate the model's practical use, a user-defined function was created to classify new, unseen images:

1. The function takes an image as input.
2. It pre-processes the image if necessary.
3. The image is then fed into the trained model.
4. The model predicts whether the water sample contains microplastics or not.

This feature allows for interactive testing and shows the model's potential for real-world water quality assessment.
![alt text](https://github.com/AryanDahiya00/Microplastic_Detection/blob/main/Capture3.JPG)

## Conclusion

The EfficientNetB0-based model for microplastic detection in water samples has shown remarkable accuracy and reliability. Key takeaways include:

1. High accuracy (98%) in distinguishing between clean water and water contaminated with microplastics.
2. Robust performance across different metrics (precision, recall, F1-score).
3. Potential for real-world application in environmental monitoring and water quality assessment.
4. Contribution to advancing microplastic detection techniques and supporting environmental conservation efforts.

Future work could focus on:
1. Expanding the dataset to include more diverse samples.
2. Exploring additional regularization techniques and hyperparameter tuning.
3. Implementing the model in continuous water quality monitoring systems.

This project demonstrates the potential of deep learning in addressing critical environmental challenges, offering a promising tool for ensuring water safety and supporting sustainable water management practices.

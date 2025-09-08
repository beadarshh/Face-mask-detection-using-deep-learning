# Real-Time Face Mask Detection using Deep Learning

This project provides an end-to-end solution for detecting face masks in real-time using a deep learning model built with TensorFlow and Keras. The entire pipeline, from data preprocessing and model training to real-time inference, is implemented within Google Colab notebooks.

![Example of 'With Mask' Detection](placeholder-with-mask-image)
![Example of 'Without Mask' Detection](placeholder-without-mask-image)

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Results & Performance](#results--performance)
- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Dataset](#dataset)
- [Setup Instructions](#setup-instructions)
- [How to Run](#how-to-run)
  - [Part 1: Model Training](#part-1-model-training)
  - [Part 2: Real-Time Detection](#part-2-real-time-detection)
- [License](#license)

## Project Overview

The objective of this endeavor is the establishment of an automated system with the capacity for ascertaining whether an individual is attired with a facial mask, accomplished through the analysis of a live video transmission. Said objective is realized through the utilization of a Convolutional Neural Network (CNN), predicated upon the MobileNetV2 architectural framework, which subsequently undergoes a process of fine-tuning with a bespoke dataset composed of facial images, both with and without masks. The resultant solution has been engineered for optimal efficiency, precision, and facile reproducibility within a cloud-based computational environment.

## Features

- **High Accuracy**: Achieves ~99.5% validation accuracy through transfer learning.
- **Real-Time Inference**: Optimized to process live webcam feeds with minimal latency.
- **Accessible & Reproducible**: The entire project is built using Google Colab, leveraging free GPU resources.
- **Visual Feedback**: Provides clear, real-time bounding boxes and confidence scores for each detection.

## Technology Stack

- **Language**: Python
- **Frameworks**: TensorFlow, Keras
- **Computer Vision**: OpenCV, Pillow
- **Data Science**: NumPy, Matplotlib
- **Development Environment**: Google Colab

## Results & Performance

The model was trained in two phases (feature extraction and fine-tuning), leading to excellent performance on the validation dataset. The training history plots below show a stable convergence of accuracy and loss, indicating a well-generalized model.

### Final Validation Metrics:
- **Accuracy**: ~99.5%
- **Loss**: ~0.02

## Getting Started

Follow these steps to set up and run the project in your own Google Colab environment.

## Prerequisites

- A Google Account to access Google Colab and Google Drive.
- A webcam for the real-time detection part.

## Dataset

The model was trained on a face mask dataset sourced from Kaggle. You will need to download and prepare this dataset.

1. **Download the Dataset**: Visit a face mask dataset on Kaggle (e.g., the "Face Mask Detection" dataset) and download it. It is usually provided as a `.zip` file.

2. **Prepare the Data**: The dataset should contain two folders:
   - `with_mask`
   - `without_mask`

3. **Upload to Google Drive**: Create a folder in your Google Drive (e.g., `FaceMaskData`) and upload the `data.zip` file there. This will allow the Colab notebook to access it easily.

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/beadarshh/Face-mask-detection-using-deep-learning.git
   ```

2. **Upload Notebooks**: Upload the two Jupyter notebooks from the repository to your Google Drive:
   - `Face_Mask_Detection_DeepLearning.ipynb`
   - `RealTimeDetection.ipynb`

## How to Run

The project is split into two main parts, each corresponding to a notebook.

### Part 1: Model Training (`Face_Mask_Detection_DeepLearning.ipynb`)

This notebook handles data loading, preprocessing, model creation, and training.

1. **Open in Google Colab**: Navigate to the notebook in your Google Drive and open it with Google Colab.

2. **Enable GPU**: Go to **Runtime** → **Change runtime type** and select **GPU** from the hardware accelerator dropdown. This will significantly speed up training.

3. **Update File Paths**: In the code cells, make sure the path to your dataset `.zip` file in Google Drive is correct.

4. **Run All Cells**: Execute the cells sequentially by clicking **Runtime** → **Run all**.

The notebook will mount your Google Drive, unzip the dataset, augment the data, build the MobileNetV2 model, and start the training process.

Upon completion, the trained model (`best_mask_model.h5`) will be saved to your Google Drive.

### Part 2: Real-Time Detection (`RealTimeDetection.ipynb`)

This notebook uses the trained model to perform real-time mask detection using your webcam.

1. **Open in Google Colab**: Open the `RealTimeDetection.ipynb` notebook.

2. **Update Model Path**: Ensure the file path in the first code cell points to the location where `best_mask_model.h5` was saved in your Google Drive.

3. **Run the Cells**: Execute the cells in order.

4. When you run the `run_colab_detection()` cell, you will be prompted to grant camera access to your browser.

5. A capture button will appear. Click it to take a photo.

6. The notebook will process the captured image, detect faces, and draw bounding boxes with the corresponding prediction ("WITH MASK" or "WITHOUT MASK") and confidence score.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

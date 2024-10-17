# Automated-Disease-Diagnosis-Using-X-Ray-Image-Analysis-v1
X-ray image analysis for automated disease diagnosis leverages advanced techniques in machine learning, computer vision, and deep learning to assist healthcare professionals in identifying and diagnosing medical conditions from X-ray scans.
# X-Ray Image Analysis for Automated Disease Diagnosis

## Overview

This project focuses on leveraging machine learning algorithms, specifically Support Vector Machines (SVM) and K Nearest Neighbors (KNN), to analyze chest X-ray images and detect pneumonia. Pneumonia is a significant global health concern, and timely, accurate diagnosis is crucial for proper treatment. By applying machine learning techniques, we aim to develop a system that can assist healthcare professionals in diagnosing pneumonia from chest X-rays automatically.

### Key Features:
- **Automated pneumonia detection** using SVM and KNN.
- **Data preprocessing** to ensure optimal model performance.
- **Model training and evaluation** using a real-world dataset.
- **Software integration** for use in hospital and lab settings to enable fast, accurate diagnosis.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Data Collection](#data-collection)
3. [Data Preprocessing](#data-preprocessing)
4. [Model Training](#model-training)
5. [Results](#results)
6. [Software Development](#software-development)
7. [Conclusion](#conclusion)
8. [How to Run](#how-to-run)

---

## Introduction

In medical diagnostics, machine learning has opened new avenues for enhancing diagnostic accuracy and efficiency. This project focuses on applying SVM and KNN algorithms to chest X-ray image datasets to detect pneumonia. The primary goal is to demonstrate the efficacy of these models in distinguishing pneumonia-positive cases from normal conditions, providing a reliable diagnostic tool.

## Data Collection

We obtained a chest X-ray image dataset from Kaggle, containing separate folders for pneumonia and normal cases. The dataset was imported into Google Colaboratory using the Kaggle API. This dataset was crucial in training and evaluating the machine learning models for the task.

## Data Preprocessing

- **Conversion to Pandas DataFrame**: The images were converted into a Pandas DataFrame and saved in CSV format.
- **Grayscale Conversion**: X-ray images were converted to grayscale, as color does not affect the diagnosis of pneumonia.
- **Image Resizing**: Each image was resized to 150x150 pixels, allowing efficient training of the machine learning models without compromising image clarity.
- **Exploratory Data Analysis**: During exploratory data analysis, we discovered that the dataset was imbalanced. We balanced the dataset using the Random Under Sampling technique to ensure a fair evaluation of our models.

## Model Training

We trained two machine learning models using Scikit-learn:
1. **Support Vector Machine (SVM)**: SVM finds the optimal hyperplane to separate different classes with the largest margin, making it a robust choice for image classification tasks.
2. **K Nearest Neighbors (KNN)**: KNN classifies new data points based on the 'k' nearest neighbors in the feature space, useful for recognizing patterns in image data.

## Results

- **Support Vector Machine (SVM)**: The SVM classifier showed superior performance, providing high recall rates, which is critical in ensuring that pneumonia cases are accurately detected.
- **K Nearest Neighbors (KNN)**: KNN also delivered reliable results, closely matching the performance of our custom and Scikit-learn implementations.

## Software Development

We developed software tailored for hospital and laboratory use, allowing users to:
- Register an account and log in.
- Upload X-ray images.
- Receive a pneumonia diagnosis prediction.

The software integrates the SVM model for backend processing due to its high accuracy and recall in detecting pneumonia.

## Conclusion

For disease detection, especially pneumonia, recall is the most crucial metric. Misclassifying an infected person as normal can have severe consequences. Based on our evaluation, the SVM model provided the highest recall, making it the most effective model for this use case. The KNN model also showed promising results, making it a robust alternative.

---

## How to Run

### Requirements:
- Python 3.7+
- Scikit-learn
- Pandas
- OpenCV
- Numpy

### Steps:
1. Clone the repository:
    ```
    git clone https://github.com/yourusername/X-Ray-Image-Analysis-Automated-Disease-Diagnosis.git
    ```
2. Install dependencies:
    ```
    pip install -r requirements.txt
    ```
3. Run the model training script:
    ```
    python train_model.py
    ```
4. Use the software to upload images and predict diagnoses.

---

## License

This project is licensed under the MIT License.

---

Feel free to contribute by opening issues or pull requests.

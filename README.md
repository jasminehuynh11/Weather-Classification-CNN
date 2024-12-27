# Weather Image Classification

This repository contains a comprehensive project that applies machine learning and deep learning techniques to classify images of weather conditions into one of four categories: **Cloudy**, **Rain**, **Shine**, and **Sunrise**. The project explores various classification techniques, from simple neural networks to advanced convolutional neural networks (CNNs) and pre-trained models, demonstrating the power of deep learning for image classification tasks.

---

## **Table of Contents**
1. [Overview](#overview)
2. [Project Workflow](#project-workflow)
3. [Implemented Models](#implemented-models)
4. [Setup and Usage](#setup-and-usage)
5. [Results and Analysis](#results-and-analysis)
6. [Future Enhancements](#future-enhancements)

---

## **Overview**

Weather image classification is a challenging task due to the subtle differences between certain weather conditions. This project employs a variety of deep learning models, ranging from custom neural networks to pre-trained MobileNetV2, to achieve state-of-the-art classification performance. The pipeline includes:

- Data preprocessing and augmentation.
- Building and training simple and complex classifiers.
- Using CNNs and pre-trained models for improved accuracy.
- Hyperparameter tuning to optimize model performance.

---

## **Project Workflow**

### 1. **Data Preparation**
- **Dataset Partitioning**: Split the dataset into training, validation, and test sets (70% training, 20% testing, 10% validation).
- **Preprocessing**: Resized all images to 230x230 pixels with normalization to [0, 1].
- **Labeling**: Images were labeled into one of four categories based on file names.

### 2. **Model Development**
- Built multiple models of increasing complexity:
  - Simple dense neural networks.
  - CNN-based architectures.
  - Transfer learning using MobileNetV2.
- Optimized hyperparameters using Keras Tuner.

### 3. **Evaluation**
- Evaluated models on unseen test data.
- Compared performance metrics, including accuracy, precision, and recall.
- Conducted error analysis to identify areas for improvement.

---

## **Implemented Models**

### 1. **Simple Neural Network (Baseline Model)**
- Architecture: A flatten layer followed by a dense layer with softmax activation.
- Accuracy: **81.07%** on the test set.

### 2. **Complex Neural Network with Hyperparameter Tuning**
- Architecture: Multiple dense layers with dropout.
- Hyperparameters tuned: Number of layers, units per layer, dropout rate, learning rate.
- Accuracy: **89.35%** on the test set.

### 3. **Convolutional Neural Network (CNN)**
- Architecture: Two convolutional layers with max pooling, followed by a dense layer.
- Features: Extracted spatial features using convolution and reduced dimensions using pooling.
- Accuracy: **99.41%** on the test set.

### 4. **Transfer Learning with MobileNetV2**
- Pre-trained Model: MobileNetV2 (trained on ImageNet).
- Modifications: Added a custom classification layer.
- Performance: Achieved **100% accuracy** on the test set.

---

## **Setup and Usage**

### Prerequisites
- Python 3.8+
- TensorFlow 2.8+
- Keras Tuner
- Pandas, NumPy, Matplotlib, Scikit-learn

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/weather-image-classification.git
   ```
2. Navigate to the project directory:
   ```bash
   cd weather-image-classification
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage
1. **Prepare Data**: Place your dataset in the `dataset/` directory.
2. **Train Models**: Run the provided notebooks or scripts to train models.
3. **Evaluate Models**: Use the evaluation scripts to compare models on the test set.

---

## **Results and Analysis**

### **Comparative Performance**
| Model                        | Test Accuracy |
|------------------------------|---------------|
| Simple Neural Network        | 81.07%        |
| Complex Neural Network       | 89.35%        |
| CNN                          | 99.41%        |
| Transfer Learning (MobileNetV2) | 100%          |

### **Key Observations**
- **Transfer Learning**: Leveraging pre-trained models like MobileNetV2 significantly improves accuracy and reduces training time.
- **CNNs**: Show excellent performance for spatial data like images.
- **Error Analysis**: Simple models struggled with subtle differences between similar categories (e.g., cloudy vs. shine).

---

## **Future Enhancements**
- Implement data augmentation to improve model generalization.
- Explore additional pre-trained models (e.g., ResNet, EfficientNet).
- Deploy the best-performing model as a web application using Streamlit or Flask.
- Integrate additional evaluation metrics like F1-score and confusion matrices.


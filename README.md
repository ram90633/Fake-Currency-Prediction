# 💵🔍 Fake Currency Prediction using Deep Learning 🧠📄

## 📌 Overview

Welcome to the **Fake Currency Prediction** project — a deep learning-based solution that detects counterfeit currency images using Convolutional Neural Networks (CNNs). The system is designed to distinguish between genuine and fake currency notes, offering practical applications in banking, retail, ATMs, and digital cash validation.

---

## 🎯 Objectives

- Build an efficient image classification model to distinguish between fake and real currency.
- Leverage Convolutional Neural Networks (CNNs) with data augmentation for improved accuracy.
- Use Keras and TensorFlow to implement and train the model.
- Evaluate the model using standard performance metrics and visualization.

---

## 📦 Dataset Summary

The dataset is a curated collection of real and fake Indian currency note images used to train and validate the CNN model.

> **Note**: The dataset appears to be a local collection (not publicly linked via Kaggle), comprising two folders: `fake` and `real`.

### 🧬 Description

This dataset consists of labeled images divided into two categories:  
- **Real Currency Notes**  
- **Fake Currency Notes**

The dataset is used to train a binary classification model, enabling real-time prediction on new, unseen images.

### 🔍 Key Technical Highlights

- 🧾 **Binary Classification Task**: Model classifies each image as either "Fake" or "Real".
- 🖼️ **Image Format**: Images are primarily in `.jpg` format with various resolutions.
- 🪄 **Preprocessing**: All images are resized to **224x224 pixels** and normalized.
- 🔄 **Augmentation**: The `ImageDataGenerator` is used to apply horizontal/vertical flips, rotations, shear, and zoom to generalize the model better.
- 🧪 **Split Ratio**: The data is split into training and validation sets using an 80:20 ratio.
- ⚖️ **Balanced Classes**: Both "fake" and "real" classes are balanced, ensuring unbiased training.

---

## 🧠 Model Summary

This project uses a **custom Convolutional Neural Network (CNN)** architecture tailored for binary classification of currency images. The model is implemented using **Keras Sequential API** with progressive convolutional and pooling layers followed by dense layers.

### 🏗️ Architecture Overview

- 📸 **Input Layer**: RGB images resized to **224x224x3**
- 🧱 **Convolutional Layers**: Three blocks of `Conv2D` layers with ReLU activation and `MaxPooling2D`
- 🧴 **Regularization**: Includes `Dropout` and `BatchNormalization` to reduce overfitting
- 🧮 **Fully Connected Layers**: Flattened outputs followed by dense layers
- 🔚 **Output Layer**: A single neuron with `sigmoid` activation for binary classification

---

## 🎯 Model Evaluation on Validation Set

- **Validation Loss:** ~0.135  
- **Validation Accuracy:** **98.4%**

---

*This model demonstrates outstanding accuracy in detecting counterfeit currency, offering a practical and efficient solution for secure currency verification systems in real-world scenarios.* 💵✅🧠

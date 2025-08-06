# Image-Based Fruit Type Prediction Using Convolutional Neural Networks

![image](https://github.com/alicelinh/fruit-classification/blob/main/fruit%20image.jpg?raw=true)

## Introduction
Fruit classification using computer vision has practical applications in agriculture, retail, and automated food processing, where accurate and efficient identification of fruit types is essential. This project explores the use of deep learning models to classify fruit images based on visual features. Convolutional neural networks (CNNs) were developed using TensorFlow and Keras, with architectures including MobileNetV2, ResNet50, VGG16, and DenseNet121. These models were trained and evaluated on a curated fruit image dataset to assess their performance in distinguishing between various fruit categories, aiming to support real-world deployment in smart farming, self-checkout systems, and supply chain automation.


## Table of Contents
- [Introduction](#introduction)
- [Data](#data)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Convolutional Neural Networks Models](#convolutional-neural-networks-models)
- [Results](#results)


## Data
The dataset consists of 16 fruit classes, with each class containing 100 training images and 20 testing images. The data is sourced from:
- [Kaggle Fruits and Vegetables Image Recognition Dataset](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition)
- Manually scraped images from the internet


## Exploratory Data Analysis
Sample images:

![image](https://media.githubusercontent.com/media/alicelinh/fruit-classification/main/sample%20fruits.png)


## Convolutional Neural Networks Models
This project applies transfer learning using several popular CNN architectures pre-trained on ImageNet:
- MobileNetV2: Lightweight architecture optimized for speed and mobile deployment
- ResNet50: Deep residual network capable of learning complex hierarchical features
- VGG16: Simple and uniform structure, widely used for benchmarking
- DenseNet121: Employs dense connections to improve feature propagation and reuse

For each model, the original top layers are removed (`include_top=False`) and replaced with a custom classification head consisting of:
- `GlobalAveragePooling2D`
- `Dense(128, activation='relu')`
- `Dropout(0.5)`
- `Dense(NUM_CLASSES, activation='softmax')`

Training is performed in two stages:
- Initial Training: The base model is frozen and only the custom head is trained.
- Fine-Tuning: The top layers of the base model are unfrozen and trained using a lower learning rate to further adapt to the fruit dataset.

All models are compiled with the `Adam` optimizer and `CategoricalCrossentropy` loss with label smoothing. Regularization and learning rate scheduling are handled via `EarlyStopping` and `ReduceLROnPlateau` callbacks.


## Results
















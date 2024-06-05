
# Image Classification Project
## Introduction
The goal of this project is to develop an image classification algorithm that can classify images into one of three categories: T-shirt/top, Sneaker, and Bag. By leveraging a subset of the Fashion MNIST dataset, we aim to build and evaluate a deep learning model using MobileNetV2 for this classification task.

## Data Source
The dataset used for this analysis is the Fashion MNIST dataset, which is publicly available and can be accessed through TensorFlow Datasets. Fashion MNIST consists of 70,000 grayscale images in 10 categories, with 7,000 images per category. For this analysis, we used a subset of the dataset that includes three categories: T-shirt/top, Sneaker, and Bag. 

## Steps Taken to Analyze the Data
1. Import Libraries
We started by importing necessary libraries, including TensorFlow, TensorFlow Datasets, Matplotlib, and others.

2. Load and Normalize Data
The Fashion MNIST dataset was loaded and filtered to include only the T-shirt/top, Sneaker, and Bag categories. The images were normalized to a [0, 1] range.

3. Preprocess Data
We converted grayscale images to RGB, resized them to 224x224 pixels to match the input size expected by MobileNetV2, and applied data augmentation techniques such as random horizontal flips and rotations to enhance the training process.

4. Prepare Datasets
The dataset was prepared for training by batching, prefetching, and caching the data to optimize the training process.

5. Build and Compile the Model
We used the MobileNetV2 architecture, pre-trained on ImageNet, as the base model. The model was customized to classify images into three categories by adding a new dense layer. Initially, only the top layer was trained while the pre-trained layers were kept frozen.

6. Train the Model
We trained the top layer of the model on the filtered Fashion MNIST dataset.

7. Fine-Tune the Model
After training the top layer, we fine-tuned the entire model with a very low learning rate. This step involved unfreezing the base model and training it end-to-end to adapt the pre-trained weights to our specific dataset.

## Results
The final model achieved a high training accuracy of 96.14%. However, the validation accuracy was lower at around 35.58%, indicating that the model might be overfitting to the training data.

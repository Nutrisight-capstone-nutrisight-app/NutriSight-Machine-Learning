# NutriSight-Machine-Learning

## Overview

NutriSight is a machine learning model designed to analyze and provide insights into nutritional data. This model has been developed using various preprocessing and modeling techniques, and it is capable of making accurate predictions based on input data. The project includes the following key components:

- **Modeling.ipynb**: This notebook contains the entire modeling process, including model training and evaluation.
- **Preprocessing.ipynb**: This notebook covers all the preprocessing steps, such as data cleaning, cropping, resizing, augmentation, and normalization, to prepare the data for modeling.
- **model.h5**: The trained model saved in H5 format, which can be loaded and used for predictions.

## How to Use

To view or download the model, please click the following link:

[Download NutriSight Model](https://drive.google.com/drive/folders/1RHi0qCqgNvhYrGAW56Xk4Q-jQu8HJf8D?usp=sharing)

## Installation

To get started with NutriSight, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/nutrisight.git
    ```

2. Navigate to the project directory:
    ```sh
    cd nutrisight
    ```

## Usage

### Preprocessing Data

Run the `Preprocessing.ipynb` notebook to preprocess your data. This step involves cleaning, cropping, resizing, augmentation, and normalization the raw data to make it suitable for model training.

### Training the Model

Use the `Modeling.ipynb` notebook to train the NutriSight model. This notebook includes steps for model training and evaluation.

### Making Predictions

You can load the trained model using the `model.h5` files and use it to make predictions on new data.

Example of loading the H5 model:

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2

# Load the model
model = load_model('model.h5')

# Function to preprocess the input image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img.astype('float32') / 255.0 
    img = np.expand_dims(img, axis=0) 
    return img

# Function to make a prediction
def predict(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    return prediction

# Example usage
image_path = 'path_to_your_image.jpg'
prediction = predict(image_path)
print(f'Prediction: {prediction}')

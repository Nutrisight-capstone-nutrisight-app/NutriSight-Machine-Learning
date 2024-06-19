# NutriSight-Machine-Learning

## Overview

NutriSight is a machine learning model designed to analyze and provide insights into nutritional data. This model has been developed using various preprocessing and modeling techniques, and it is capable of making accurate predictions based on input data. The project includes the following key components:

- **Preprocessing.ipynb**: This notebook covers all the preprocessing steps, such as data cleaning, cropping, resizing, augmentation, and normalization, to prepare the data for modeling.
- **Modeling.ipynb**: This notebook contains the entire modeling process, including building, training, and evaluating the model.
- **predict.py**: A script to make predictions using the trained model.
- **model.h5**: The trained model saved in H5 format, which can be loaded and used for predictions.
- **requirements.txt**: A file listing all the dependencies required to run the project.

## How to Use

To view or download the model, please click the following link:

[Download NutriSight Model](https://drive.google.com/file/d/1kB_BM1VFDS1TbeX_uS0cYamFUQBePWxN/view?usp=sharing)

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
3. Install the required dependencies using `pip`:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Preprocessing Data

Run the `Preprocessing.ipynb` notebook to preprocess your data. This step involves cleaning, cropping, resizing, augmentation, and normalization the raw data to make it suitable for model training.

### Training the Model

Use the `Modeling.ipynb` notebook to train the NutriSight model. This notebook includes steps for model training and evaluation.

### Making Predictions

To make predictions using the trained model, use the `predict.py` script. You can run this script from the command line. For example:

```python
# Example usage
python predict.py model.h5 "path_to_your_image.jpg"


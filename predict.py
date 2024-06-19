import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import argparse

# Dictionary to map class indices to product names
class_names = {
    0: "Chitato",
    1: "Coca-cola",
    2: "Fanta",
    3: "Fitbar",
    4: "Frisian Flag",
    5: "Garuda Kacang Kulit",
    6: "Good Day",
    7: "Hydro Coco",
    8: "Indomie Goreng",
    9: "Kratingdeng",
    10: "Milo Kaleng",
    11: "Mizone",
    12: "Nabati",
    13: "Netscafe",
    14: "Nutriboost",
    15: "Oreo",
    16: "Pocky",
    17: "Pop Mie",
    18: "Qtela",
    19: "Roma Kelapa",
    20: "Silver Queen",
    21: "SoyJoy",
    22: "Sprite",
    23: "Ultra Milk",
    24: "You C1000 140ml",
    25: "You C1000 Water Orange 500ml"
}

def load_trained_model(model_path):
    """
    Load the pre-trained Keras model from the specified file path.
    """
    return load_model(model_path)

def preprocess_image(image_path):
    """
    Preprocess the image to the format required by the model.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image file not found at the specified path: {image_path}")
    img = cv2.resize(img, (224, 224))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict(model, image_path):
    """
    Predict the class of the image using the loaded model.
    """
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    return prediction

def main(args):
    model = load_trained_model(args.model_path)
    prediction = predict(model, args.image_path)
    predicted_class = np.argmax(prediction)
    predicted_probability = prediction[0][predicted_class]
    product_name = class_names[predicted_class]
    print(f'Predicted class: {product_name}, Probability: {predicted_probability}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict the class of an image using a pre-trained model.")
    parser.add_argument('model_path', type=str, help="Path to the pre-trained model file.")
    parser.add_argument('image_path', type=str, help="Path to the image file to be predicted.")
    
    args = parser.parse_args()
    main(args)
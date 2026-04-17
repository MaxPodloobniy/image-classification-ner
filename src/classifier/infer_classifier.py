"""
Image classification inference script.

This script loads a trained Keras image classification model and predicts
the animal class from a given image, returning the predicted class
and confidence score.
"""
import argparse
import json
import logging
import os

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array, load_img

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments for training."""
    parser = argparse.ArgumentParser(description='Inference image classification model')

    parser.add_argument('--image_path', type=str, required=True, help='Path to image')
    parser.add_argument('--model_path', type=str, default='models/classifier/classifier_model.keras', help='Path to model')

    return parser.parse_args()


def preprocess_image(image_path, target_size):
    """Load and preprocess image for model inference."""
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array


def classify_image(image_path, model_path):
    """Checks if all files exist, load model and classes and make a prediction on the image"""
    # Check if image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Check if classes exists
    classes_path = model_path.replace('.keras', '_classes.json')
    if not os.path.exists(classes_path):
        raise FileNotFoundError(f"Class names file not found: {classes_path}")

    # Load class names
    with open(classes_path) as f:
        class_indices = json.load(f)

    # Load the model
    model = load_model(model_path)

    # Load the image
    input_shape = model.input_shape[1:3]  # (height, width)
    img_array = preprocess_image(image_path, input_shape)

    predictions = model.predict(img_array, verbose=0)
    idx_to_class = list(class_indices.keys())  # ['chimpanzee', 'coyote', ...]
    predicted_class = idx_to_class[np.argmax(predictions)]
    confidence = np.max(predictions)

    return predicted_class, confidence


def main():
    args = parse_args()
    predicted_class, confidence = classify_image(args.image_path, args.model_path)
    logger.info(f"Predicted class: {predicted_class} (confidence: {confidence:.2%})")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    main()

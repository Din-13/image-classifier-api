import numpy as np
import tensorflow as tf
from PIL import Image


def preprocess_image(image, target_size=(224, 224)):
    if image.size != target_size:
        image = image.resize(target_size)

    image_array = np.array(image)

    image_array = np.expand_dims(image_array, axis=0)

    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)

    return image_array


def format_predictions(predictions, top_k=5):
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=top_k)[0]


    results = [
        {"class_id": class_id, "class_name": class_name, "probability": float(score)}
        for class_id, class_name, score in decoded_predictions
    ]

    return results
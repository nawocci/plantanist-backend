import tensorflow as tf
import numpy as np
from google.cloud import storage
from typing import List, Tuple
import os

# Paths
MODEL_PATH = 'model/plant_disease_model.h5'
CLASSES_PATH = 'model/classes.txt'
BUCKET_NAME = 'plantanist-model'
MODEL_BLOB_NAME = 'plant_disease_model.h5'
CLASSES_BLOB_NAME = 'classes.txt'

def download_file_from_gcs(bucket_name: str, source_blob_name: str, destination_file_name: str):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"File {source_blob_name} downloaded to {destination_file_name}")

def load_model(model_path: str) -> tf.keras.Model:
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        raise RuntimeError(f"Error loading model from {model_path}: {e}")

def load_classes(file_path: str) -> List[str]:
    try:
        with open(file_path, 'r') as f:
            return [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        raise RuntimeError(f"Class file not found at {file_path}. Please provide a valid classes.txt file.")

def preprocess_image(img_path: str) -> np.ndarray:
    try:
        img = tf.keras.utils.load_img(img_path, target_size=(256, 256))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        raise ValueError(f"Error preprocessing image {img_path}: {e}")

def classify_image(img_path: str, model: tf.keras.Model, classes: List[str]) -> Tuple[str, float]:
    try:
        img_array = preprocess_image(img_path)
        predictions = model.predict(img_array)
        prediction_index = np.argmax(predictions[0])
        predicted_class = classes[prediction_index]
        confidence_score = np.max(predictions[0]) * 100
        return predicted_class, confidence_score
    except Exception as e:
        raise RuntimeError(f"Error during classification: {e}")

# Download model and classes file from GCS if not already done
def download_model_and_classes():
    if not os.path.exists(MODEL_PATH):
        download_file_from_gcs(BUCKET_NAME, MODEL_BLOB_NAME, MODEL_PATH)

    if not os.path.exists(CLASSES_PATH):
        download_file_from_gcs(BUCKET_NAME, CLASSES_BLOB_NAME, CLASSES_PATH)

# Call this function before starting the app
download_model_and_classes()

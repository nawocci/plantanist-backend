from flask import Flask, request, jsonify
from model import load_model, load_classes, classify_image, download_file_from_gcs
import os
import tempfile

app = Flask(__name__)

MODEL_PATH = '/tmp/plant_disease_model.h5'
CLASSES_PATH = '/tmp/classes.txt'
BUCKET_NAME = 'plantanist-model'
MODEL_BLOB_NAME = 'plant_disease_model.h5'
CLASSES_BLOB_NAME = 'classes.txt'

# Download the model and classes file from GCS if not already downloaded
def download_model_and_classes():
    if not os.path.exists(MODEL_PATH):
        download_file_from_gcs(BUCKET_NAME, MODEL_BLOB_NAME, MODEL_PATH)

    if not os.path.exists(CLASSES_PATH):
        download_file_from_gcs(BUCKET_NAME, CLASSES_BLOB_NAME, CLASSES_PATH)

# Download files at app startup
download_model_and_classes()

# Load model and classes after download
try:
    model = load_model(MODEL_PATH)
    classes = load_classes(CLASSES_PATH)
except RuntimeError as e:
    raise RuntimeError(f"Initialization error: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected image'}), 400

    if file:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            file.save(temp_file.name)

        try:
            predicted_class, confidence_score = classify_image(temp_file.name, model, classes)
            confidence_score = float(confidence_score)  # Ensure it's JSON serializable
        except RuntimeError as e:
            return jsonify({'error': f"Prediction error: {e}"}), 500
        finally:
            os.remove(temp_file.name)

        return jsonify({
            'predicted_class': predicted_class,
            'confidence_score': confidence_score
        })

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

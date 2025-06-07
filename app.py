from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import requests
import tensorflow as tf
from PIL import Image
import numpy as np
import io

app = Flask(__name__)
CORS(app)

MODEL_URL = "https://kitish-whatsapp-bot-media.s3.ap-south-1.amazonaws.com/documentMessage_1749283994496.bin"
MODEL_PATH = "model.h5"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        response = requests.get(MODEL_URL)
        response.raise_for_status()  # Ensure HTTP errors raise exceptions
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        print("Model downloaded.")

download_model()

# Load model once on startup
model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((150, 150))  # Replace with your model’s required size
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        image_bytes = file.read()
        input_tensor = preprocess_image(image_bytes)
        prediction = model.predict(input_tensor)
        result = "Pneumonia Detected" if prediction[0][0] > 0.5 else "Normal"
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Use PORT env var or default to 8000
    app.run(host="0.0.0.0", port=port, debug=True)

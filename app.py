import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename

# Create uploads folder if it doesn't exist
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
model = keras.models.load_model("mnist_model.h5")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Preprocess the image
    try:
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return jsonify({"error": "Invalid image format"}), 400
        img = cv2.resize(img, (28, 28))
        img = img / 255.0  # Normalize
        img = img.reshape(1, 28, 28, 1)

        # Make prediction
        prediction = model.predict(img)
        predicted_digit = int(np.argmax(prediction))
        confidence = float(np.max(prediction)) * 100

        return jsonify({
            "digit": predicted_digit,
            "confidence": round(confidence, 2),
            "image": filepath
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Serve uploaded images
@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
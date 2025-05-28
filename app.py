import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
model = keras.models.load_model("mnist_model.h5")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def home():
    return render_template("index.html")

def preprocess_image(filepath):
    # 1. Load in grayscale
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not read image. Try uploading a different file.")

    # 2. Apply GaussianBlur to reduce noise
    img = cv2.GaussianBlur(img, (5,5), 0)

    # 3. Apply adaptive thresholding (invert so digit is white, background is black)
    img = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    # 4. Find contours and get the biggest one (assume it's the digit)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No digit found in image.")

    # Find the largest contour by area
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    digit = img[y:y+h, x:x+w]

    # 5. Resize digit to fit in a 20x20 box, then pad to 28x28
    # Keep aspect ratio
    h, w = digit.shape
    if h > w:
        new_h = 20
        new_w = int(w * (20.0 / h))
    else:
        new_w = 20
        new_h = int(h * (20.0 / w))
    digit = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Pad the image to 28x28
    pad_top = (28 - new_h) // 2
    pad_bottom = 28 - new_h - pad_top
    pad_left = (28 - new_w) // 2
    pad_right = 28 - new_w - pad_left
    digit = np.pad(digit, ((pad_top, pad_bottom), (pad_left, pad_right)), 'constant', constant_values=0)

    # 6. Normalize and reshape for model
    digit = digit.astype("float32") / 255.0
    digit = digit.reshape(1, 28, 28, 1)
    return digit

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

    try:
        img = preprocess_image(filepath)
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

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
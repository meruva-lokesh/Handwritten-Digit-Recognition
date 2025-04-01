import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os
from flask import Flask, request, jsonify, render_template

# Load the trained model
model = keras.models.load_model("mnist_model.h5")  # Save model after training

app = Flask(__name__)

# Route for homepage
@app.route("/")
def home():
    return render_template("index.html")

# Route for predictions
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    filepath = os.path.join("static/uploads", file.filename)
    file.save(filepath)

    # Preprocess the image
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    img = img / 255.0  # Normalize
    img = img.reshape(1, 28, 28, 1)

    # Make prediction
    prediction = model.predict(img)
    predicted_digit = np.argmax(prediction)

    return jsonify({"digit": int(predicted_digit), "image": filepath})

if __name__ == "__main__":
    app.run(debug=True)

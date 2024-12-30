from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

app = Flask(__name__)

# Ensure the uploads folder exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Try loading the model and handle potential errors
try:
    # Load the trained model (ensure this path is correct)
    model = load_model('models/OCR_ResNet_best.h5')  # Use this to load .h5 models
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None  # Ensure the model is set to None if loading fails

# Define the categories (10 digits + 26 letters)
categories = [str(i) for i in range(10)] + [chr(i) for i in range(ord('A'), ord('Z') + 1)]

# Define the image size (for consistency with your model input)
IMG_SIZE = (28, 28)

# Preprocessing function to prepare the image for the model
def prepare_image(image):
    try:
        img = load_img(image, target_size=IMG_SIZE, color_mode='grayscale')
        img = img_to_array(img) / 255.0  # Normalize the image
        img = img.reshape(1, 28, 28, 1)  # Add batch dimension
        return img
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

@app.route('/')
def home():
    return render_template('index.html', prediction=None)  # Initial render without prediction

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Save the uploaded file temporarily
        img_path = os.path.join('uploads', file.filename)
        try:
            file.save(img_path)

            # Prepare the image
            img = prepare_image(img_path)

            if img is None:
                return jsonify({'error': 'Image preprocessing failed'}), 400

            # Make prediction (ensure the model is loaded)
            if model:
                prediction = model.predict(img)
                predicted_class = categories[np.argmax(prediction)]  # Get the class with the highest probability

                # Render the template and pass the prediction to it
                return render_template('index.html', prediction=predicted_class)
            else:
                return jsonify({'error': 'Model not loaded'}), 500
        except Exception as e:
            return jsonify({'error': str(e)}), 500

from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import base64
import numpy as np
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.preprocessing import image

# Initialize Flask application
app = Flask(__name__)

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the models directory exists
MODEL_FOLDER = 'static/models'
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Load the local Keras model
MODEL_PATH = 'static/models/skin_lesion_model.h5'
local_model = load_model(MODEL_PATH)


categories = [
    'actinic keratosis / intraepithelial carcinoma',  # 0
    'basal cell carcinoma',                          # 1
    'benign keratosis-like lesions',                 # 2
    'dermatofibroma',                                # 3
    'melanocytic nevi',                              # 4
    'melanoma',                                      # 5
    'vascular lesions'                               # 6
]

def classify_with_local_model(image_path):
    """Classify image using the local .h5 model trained on 28x28x3 RGB data."""
    try:
        img = image.load_img(image_path, target_size=(28, 28))  # Your model input size
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0  # Normalize to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 28, 28, 3)

        prediction = local_model.predict(img_array)
        predicted_index = np.argmax(prediction[0])

        return categories[predicted_index]
    except Exception as e:
        print(f"Local model classification error: {e}")
        return None

def classify_image_local_only(image_path):
    """Use only the local model for prediction. Groq is disabled."""
    local_prediction = classify_with_local_model(image_path)
    return local_prediction if local_prediction else "Prediction Failed"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['GET', 'POST'])
def classify():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Use only local model for classification
            predicted_category = classify_image_local_only(file_path)

            return render_template('classify.html', prediction=predicted_category, filename=filename)

    return render_template('classify.html')

@app.route("/results")
def results():
    return render_template('results.html')

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(MODEL_FOLDER, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

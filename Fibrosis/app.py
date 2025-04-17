from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import base64
import numpy as np
from werkzeug.utils import secure_filename
from groq import Groq

# Initialize Flask application
app = Flask(__name__)

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the models directory exists
MODEL_FOLDER = 'static/models'
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Groq API client
client = Groq(api_key="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")

# Predefined categories
categories = ['benign keratosis-like lesions', 'melanocytic nevi', 'dermatofibroma', 'melanoma', 'vascular lesions', 'basal cell carcinoma', 'actinic keratosis / intraepithelial carcinoma']

def encode_image(image_path):
    """Convert image to base64 encoding."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def classify_image(image_path):
    """Classify the image using the Groq API."""
    base64_image = encode_image(image_path)
    
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Classify this skin lesion into one of these categories: benign keratosis-like lesions, melanocytic nevi, dermatofibroma, melanoma, vascular lesions, basal cell carcinoma, actinic keratosis / intraepithelial carcinoma."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}   
                ],
            }
        ],
        model="meta-llama/llama-4-scout-17b-16e-instruct",
    )
    
    response_text = chat_completion.choices[0].message.content.lower()
    for category in categories:
        if category in response_text:
            return category
    return "Unknown"

@app.route('/')
def home():
    """Render the homepage."""
    return render_template('index.html')

@app.route('/classify', methods=['GET', 'POST'])
def classify():
    """Handle image upload and classification."""
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

            # Perform classification
            predicted_category = classify_image(file_path)

            return render_template('classify.html', prediction=predicted_category, filename=filename)

    return render_template('classify.html')

@app.route("/results")
def results():
    """Render the results page."""
    return render_template('results.html')

@app.route('/download/<filename>')
def download_file(filename):
    """Serve files from the models directory."""
    return send_from_directory(MODEL_FOLDER, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

from flask import Flask, render_template, request, redirect, url_for
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
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                ],
            }
        ],
        model="llama-3.2-11b-vision-preview",
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

if __name__ == '__main__':
    app.run(debug=True)

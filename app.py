import base64
import json
import os
from google.cloud import aiplatform
from flask import Flask, request, jsonify, render_template, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Set up your project and model info
PROJECT_ID = "second-457818"
MODEL_ID = "your-vertex-ai-model-id"
ENDPOINT_ID = "872102936337973248"
REGION = "us-central1"

# Initialize the Vertex AI API client
aiplatform.init(project=PROJECT_ID, location=REGION)

# Endpoint for bone fracture prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if the file is in the request
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        
        # If no file is selected
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        # Secure the filename and save the file temporarily
        filename = secure_filename(file.filename)
        file_path = os.path.join('/tmp', filename)
        file.save(file_path)
        
        # Read the file and encode it into base64
        with open(file_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
        
        # Create the request payload for the Vertex AI model
        instances = [{"b64": encoded_image}]
        
        # Call Vertex AI prediction endpoint
        endpoint = aiplatform.Endpoint(endpoint_name=ENDPOINT_ID)
        response = endpoint.predict(instances=instances)

        # Extract prediction result
        prediction = response.predictions[0]

        # Render the result in result.html
        return render_template('result.html', prediction=prediction)

    except Exception as e:
        # Handle errors
        return jsonify({"error": str(e)}), 500

# Index route
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

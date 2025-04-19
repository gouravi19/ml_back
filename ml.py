from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
from collections import Counter
import os

app = Flask(__name__)
CORS(app)  # Allow frontend to access backend

# Load your trained YOLO model
model = YOLO("best.pt")

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']

    try:
        img = Image.open(file.stream).convert('RGB')
    except Exception as e:
        return jsonify({"error": f"Invalid image format: {str(e)}"}), 400

    # Make prediction
    results = model.predict(img)
    boxes = results[0].boxes.data

    # Extract class names
    class_names = [model.names[int(cls)] for cls in boxes[:, 5]]
    count = Counter(class_names)

    # Format response
    response = {
        "WBC": count.get("WBC", 0),
        "RBC": count.get("RBC", 0),
        "Platelets": count.get("platelets", 0)  # Check class name case
    }

    return jsonify(response)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Use PORT from environment on Render
    app.run(host='0.0.0.0', port=port)

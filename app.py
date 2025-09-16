from flask import Flask, render_template, request, jsonify, send_from_directory
import io
import os
import uuid
import base64
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models
from datetime import datetime
import traceback

from static.model.model_definitions import SwinSegmentationModel

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    model = SwinSegmentationModel().to(device)
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, 2)
    
    try:
        state_dict = torch.load("static/model/best_model.pth", map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        print("Custom SwinSegmentationModel loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return model

model = load_model().to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class_names = ["No Tumor", "Tumor"]

@app.route("/")
def index():
    return render_template("classification.html")

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    try:
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        image = image.resize((224, 224))
        
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        img_t = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_t)  # [1,1,H,W]
            mask = outputs.squeeze().cpu().numpy()  # [H,W]
            mask = (mask > 0.5).astype("uint8") * 255  # binary mask

        mask_img = Image.fromarray(mask).convert("L")

        buffered = io.BytesIO()
        mask_img.save(buffered, format="PNG")
        mask_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        image = image.resize(mask_img.size)
        overlay = Image.blend(image.convert("RGBA"), mask_img.convert("RGBA"), alpha=0.6)

        buffered = io.BytesIO()
        overlay.save(buffered, format="PNG")
        overlay_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return jsonify({
            "image_data": f"data:image/jpeg;base64,{img_str}",
            "mask_data": f"data:image/png;base64,{mask_str}",
            "overlay_data": f"data:image/png;base64,{overlay_str}",
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        print("Error processing image:", str(e))
        traceback.print_exc()
        return jsonify({"error": f"Error processing image: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
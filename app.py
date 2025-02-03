from flask import Flask, render_template, request, jsonify
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import os
import datetime

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load BLIP model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    
    with torch.no_grad():
        caption_ids = model.generate(**inputs)
    
    caption = processor.decode(caption_ids[0], skip_special_tokens=True)
    return caption

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files["image"]
    if image.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Save image with a timestamp to avoid overwriting
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    image_path = os.path.join(UPLOAD_FOLDER, f"captured_{timestamp}.jpg")
    image.save(image_path)

    caption = generate_caption(image_path)

    return jsonify({"caption": caption})

if __name__ == "__main__":
    app.run(debug=True)

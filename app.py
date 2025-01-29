from flask import Flask, request, jsonify, send_file
from PIL import Image
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import os

# Disable GPU warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = Flask(__name__)
hub_module = hub.load("https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2")

# Preprocess image
def load_and_process_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((256, 256))
    img = np.array(img) / 255.0
    return tf.convert_to_tensor(img, dtype=tf.float32)[tf.newaxis, ...]

# Deprocess image
def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        tensor = tensor[0]
    return Image.fromarray(tensor)

# Style transfer function
def perform_style_transfer(content_path, style_path):
    content_image = load_and_process_image(content_path)
    style_image = load_and_process_image(style_path)
    outputs = hub_module(content_image, style_image)
    stylized_image = outputs[0]
    output_path = "output/generated_image.jpg"
    tensor_to_image(stylized_image).save(output_path)
    return output_path

@app.route('/style-transfer', methods=['POST'])
def style_transfer():
    if 'content_image' not in request.files or 'style_image' not in request.files:
        return jsonify({"error": "Please upload both content and style images"}), 400

    content_image = request.files['content_image']
    style_image = request.files['style_image']

    content_path = "uploads/content_image.jpg"
    style_path = "uploads/style_image.jpg"

    os.makedirs("uploads", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    content_image.save(content_path)
    style_image.save(style_path)

    output_path = perform_style_transfer(content_path, style_path)
    return send_file(output_path, mimetype='image/jpeg')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=True)

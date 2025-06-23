from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import model_from_json
import numpy as np
from PIL import Image
import keras
import google.generativeai as genai
import os
import re
from flask_cors import CORS


app = Flask(__name__)
CORS(app)
# Load the model architecture and weights
with open("plant_model.json", "r", encoding="utf-8") as f:
    model = model_from_json(f.read())

model.load_weights("plant_model.weights.h5")
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# 38-class label list (must match model training order exactly)
labels = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
    'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# Image preprocessing
def preprocess_image(image):
    img = Image.open(image).convert("RGB")
    img = img.resize((64, 64))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.route('/')
def home():
    return render_template("index.html")  # Assumes index.html is in /templates

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get("image")
    if not file:
        return jsonify({"result": "No file uploaded"}), 400

    try:
        img = preprocess_image(file)
        prediction = model.predict(img)
        predicted_class = labels[np.argmax(prediction)]
        result = re.sub(r'\b(\w+)( \1\b)+', r'\1', predicted_class.replace("_", " "))
        confidence = float(np.max(prediction)) * 100
        return jsonify({"result": f"{result} ({confidence:.2f}%)"})
    except Exception as e:
        return jsonify({"result": f"Error: {str(e)}"}), 500

# Optional chatbot stub (connect to real chatbot engine or Gemini API)
### --- Chatbot Section --- ###

# Configure Gemini API
api_key = "AIzaSyBusmdXcpH2qvFtwH3sSAP6mZq3m1noFu8"  # Replace your valid API key
genai.configure(api_key=api_key)

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model_ai = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

chat_session = model_ai.start_chat(history=[])

# Predefined Q&A
stored_qa = {
    "What is your name?": "I am a chatbot powered by AGRO360.",
    "How does AI work?": "AI works by using algorithms and data to perform tasks that usually require human intelligence.",
    "Who created you?": "I was created by Surya and Nedesh Kumar.",
    "How do I submit a complaint?": "You can submit a complaint by visiting the complaints portal and filling out the form with the necessary details.",
    "Can I submit an anonymous complaint?": "Yes, anonymous complaints are allowed, but we can't send you updates.",
}

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"response": "Please enter a message."})

    # Directly check predefined Q&A
    response = stored_qa.get(user_message, None)

    if not response:
        # If not found, generate a response using Gemini AI
        chat_response = chat_session.send_message(user_message)
        response = chat_response.text

    return jsonify({"response": response})

if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')  # To prevent UnicodeEncodeError on Windows
    app.run(debug=True)

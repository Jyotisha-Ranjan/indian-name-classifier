# ============================================================
# app.py — Flask API for Indian Name Classifier
# ============================================================

# STEP 1 — Import libraries
from flask import Flask, request, jsonify
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ============================================================
# STEP 2 — Load saved model and files
# ============================================================

# Load the trained LSTM model
model = load_model('indian_name_model.keras')
print("Model loaded successfully! ✅")

# Load the character dictionary
with open('char_to_index.pkl', 'rb') as f:
    char_to_index = pickle.load(f)
print("Dictionary loaded successfully! ✅")

# Load MAX_LEN
with open('max_len.pkl', 'rb') as f:
    MAX_LEN = pickle.load(f)
print("MAX_LEN loaded successfully! ✅")

# ============================================================
# STEP 3 — Create Flask app
# ============================================================

app = Flask(__name__)

# ============================================================
# STEP 4 — Create prediction function
# ============================================================

def predict_name(name):
    # Convert to lowercase
    name = name.lower()

    # Convert characters to numbers
    name_numbers = [char_to_index.get(char, 0) for char in name]

    # Pad to MAX_LEN
    name_padded = pad_sequences(
        [name_numbers],
        maxlen=MAX_LEN,
        padding='post'
    )

    # Get prediction from model
    prediction = model.predict(name_padded, verbose=0)[0][0]

    # Return result
    if prediction > 0.5:
        return {
            "name": name.title(),
            "result": "Indian Name",
            "is_indian": True,
            "confidence": f"{prediction * 100:.2f}%"
        }
    else:
        return {
            "name": name.title(),
            "result": "Not Indian Name",
            "is_indian": False,
            "confidence": f"{(1 - prediction) * 100:.2f}%"
        }

# ============================================================
# STEP 5 — Create API endpoints
# ============================================================

# Endpoint 1 — Home route
# Just to check if API is running
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Indian Name Classifier API is running! ✅",
        "usage": "Send POST request to /predict with name"
    })

# Endpoint 2 — Predict route
# Main endpoint to predict name
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the name from request
        data = request.get_json()

        # Check if name is provided
        if 'name' not in data:
            return jsonify({
                "error": "Please provide a name!",
                "example": {"name": "Rajesh Kumar"}
            }), 400

        # Get the name
        name = data['name']

        # Check if name is empty
        if not name or name.strip() == '':
            return jsonify({
                "error": "Name cannot be empty!"
            }), 400

        # Get prediction
        result = predict_name(name)

        # Return result
        return jsonify(result), 200

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

# ============================================================
# STEP 6 — Run the Flask app
# ============================================================

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )
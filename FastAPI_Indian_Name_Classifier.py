from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model('indian_name_model.keras')
print("Model loaded successfully!")

with open('char_to_index.pkl', 'rb') as f:
    char_to_index = pickle.load(f)
print("Dictionary loaded successfully!")

with open('max_len.pkl', 'rb') as f:
    MAX_LEN = pickle.load(f)
print("MAX_LEN loaded successfully!")

app = FastAPI()

class NameInput(BaseModel):
    name: str

def predict_name(name):
    name = name.strip()
    name = name.lower()
    name_numbers = [char_to_index.get(char, 0) for char in name]

    name_padded = pad_sequences(
        [name_numbers],
        maxlen=MAX_LEN,
        padding='post'
    )

    prediction = model.predict(name_padded, verbose=0)[0][0]

    if prediction > 0.5:
        return {
            "name": name.title(),
            "result": "Indian Name",
            "confidence": f"{prediction * 100:.2f}%"
        }
    else:
        return {
            "name": name.title(),
            "result": "Not Indian Name",
            "confidence": f"{(1 - prediction) * 100:.2f}%"
        }

#Home route
@app.get("/")
def home():
    return {
        "message": "Indian Name Classifier API is running!",
        "usage": "Send POST request to /predict with name"
    }

#Predict route
@app.post("/predict")
def predict(data: NameInput):
    try:
        name = data.name

        if not name or name.strip() == '':
            return {"error": "Name cannot be empty!"}

        result = predict_name(name)
        return result

    except Exception as e:
        return {"error": str(e)}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
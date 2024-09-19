import pickle
from fastapi import FastAPI

app = FastAPI()

# Correct path to your saved model
model_path = "model/logistic_model.pkl"

# Load the model when the app starts
try:
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    print("Model loaded successfully!")
except FileNotFoundError:
    print(f"Model file not found at {model_path}. Please provide the correct path.")
except Exception as e:
    print(f"An error occurred while loading the model: {e}")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Titanic Survival Prediction API!"}

@app.post("/predict/")
def predict(data: dict):
    # Assuming data processing happens here
    prediction = model.predict([data['features']])
    return {"prediction": prediction}

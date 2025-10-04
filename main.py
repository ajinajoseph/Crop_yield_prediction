from fastapi import FastAPI
import pickle
import pandas as pd

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load training columns
with open("columns.pkl", "rb") as f:
    training_columns = pickle.load(f)

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Crop Yield Prediction API is running!"}

@app.post("/predict")
def predict(area: float, crop_year: int, season: str, crop: str, district: str):
    # Convert raw input to DataFrame
    input_data = pd.DataFrame([{
        "Area": area,
        "Crop_Year": crop_year,
        "Season": season,
        "Crop": crop,
        "District_Name": district
    }])

    input_data = pd.get_dummies(input_data)
    input_data = input_data.reindex(columns=training_columns, fill_value=0)

    # Predict yield
    prediction = model.predict(input_data)[0]
    return {"Predicted Yield": float(prediction)}

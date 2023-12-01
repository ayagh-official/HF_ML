from typing import Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import load_model

app = FastAPI()

model_path = "./trainedModel.h5"
binary_model = load_model(model_path)


class Item(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int


@app.post("/predict")
def predict(item: Item):
    # Convert input data to a NumPy array
    input_data = np.array([[
        item.age, item.sex, item.cp, item.trestbps, item.chol, item.fbs, item.restecg,
        item.thalach, item.exang, item.oldpeak, item.slope, item.ca, item.thal
    ]])

    # Make prediction using the loaded model
    prediction = binary_model.predict(input_data)

    # The output is a probability, you can convert it to a class (0 or 1) based on a threshold
    threshold = 0.5
    predicted_class = 1 if prediction[0, 0] > threshold else 0

    # Define output messages based on the predicted class
    output_messages = {
        0: "No Heart Failure predicted for the patient",
        1: "Heart Failure may develope for the patient"
    }

    return {"prediction": predicted_class, "message": output_messages[predicted_class]}

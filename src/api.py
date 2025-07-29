import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.base import BaseEstimator
import mlflow.sklearn

# Define input schema
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Load model
model_path = os.path.join(os.path.dirname(__file__), '../models/best_model')
model: BaseEstimator = mlflow.sklearn.load_model(model_path)

app = FastAPI()

@app.post('/predict')
def predict(input: IrisInput):
    data = np.array([[input.sepal_length, input.sepal_width, input.petal_length, input.petal_width]])
    pred = model.predict(data)[0]
    return {'prediction': int(pred)}

@app.get('/')
def root():
    return {'message': 'Iris model prediction API'}

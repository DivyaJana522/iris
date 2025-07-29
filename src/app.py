import os
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn
from typing import List

app = FastAPI()

# Load the best model
model_path = os.path.join(os.path.dirname(__file__), '../models/best_model')
model = mlflow.sklearn.load_model(model_path)

class PredictRequest(BaseModel):
    inputs: List[List[float]]

class PredictResponse(BaseModel):
    predictions: List[int]

@app.post('/predict', response_model=PredictResponse)
def predict(request: PredictRequest):
def predict(request: PredictRequest):
    if not request.inputs:
        raise HTTPException(status_code=400, detail='No input data provided')
    X = np.array(request.inputs)
    preds = model.predict(X)
    return PredictResponse(predictions=preds.tolist())

# To run: uvicorn app:app --host 0.0.0.0 --port 5000

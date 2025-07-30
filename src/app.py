
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn
import logging
import numpy as np
from typing import List


# Set up logging
logging.basicConfig(
    filename=os.path.join(os.path.dirname(__file__), '../logs/prediction.log'),
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)


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
    if not request.inputs:
        raise HTTPException(status_code=400, detail='No input data provided')
    X = np.array(request.inputs)
    preds = model.predict(X)
    # Log the request and prediction
    logging.info(f"Request: {request.inputs} | Prediction: {preds.tolist()}")
    return PredictResponse(predictions=preds.tolist())

# To run: uvicorn app:app --host 0.0.0.0 --port 5000

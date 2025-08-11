
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn
import logging
import numpy as np
from typing import List
import sqlite3



# Ensure logs directory exists
log_dir = os.path.join(os.path.dirname(__file__), '../logs')
os.makedirs(log_dir, exist_ok=True)

# Set up logging
logging.basicConfig(
    filename=os.path.join(log_dir, 'prediction.log'),
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

# Set up SQLite logging
sqlite_path = os.path.join(log_dir, 'predictions.db')
conn = sqlite3.connect(sqlite_path, check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request TEXT,
    prediction TEXT
)''')
conn.commit()


app = FastAPI()


# Load the best model
model_path = os.path.join(os.path.dirname(__file__), '../models/best_model')
model = mlflow.sklearn.load_model(model_path)


class PredictRequest(BaseModel):
    inputs: List[List[float]]


class PredictResponse(BaseModel):
    predictions: List[int]



@app.post('/predict', response_model=PredictResponse)
async def predict(request: PredictRequest):
    if not request.inputs:
        raise HTTPException(status_code=400, detail='No input data provided')
    X = np.array(request.inputs)
    preds = model.predict(X)
    # Log the request and prediction
    logging.info(f"Request: {request.inputs} | Prediction: {preds.tolist()}")
    # Log to SQLite
    cursor.execute("INSERT INTO logs (request, prediction) VALUES (?, ?)", (str(request.inputs), str(preds.tolist())))
    conn.commit()
    return PredictResponse(predictions=preds.tolist())

# Optional metrics endpoint
@app.get('/metrics')
def metrics():
    cursor.execute("SELECT COUNT(*) FROM logs")
    count = cursor.fetchone()[0]
    return {"prediction_requests": count}

# To run: uvicorn app:app --host 0.0.0.0 --port 5000


import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn
import joblib
import logging
import numpy as np
from typing import List
import sqlite3
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
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

# Prometheus metrics
prediction_counter = Counter('prediction_requests_total', 'Total prediction requests')
app = FastAPI()



def load_model():  # noqa: E305
    model_path = os.path.join(os.path.dirname(__file__), '../models/best_model')
    return mlflow.sklearn.load_model(model_path)

model = load_model()

from pydantic import validator


class PredictRequest(BaseModel):
    inputs: List[List[float]]

    @validator('inputs')
    def check_inputs(cls, v):
        if not v:
            raise ValueError('Input list cannot be empty.')
        if not isinstance(v, list):
            raise ValueError('Inputs must be a list of lists')
        for row in v:
            if not isinstance(row, list) or len(row) != 4:
                raise ValueError('Each input must be a list of 4 floats')
            for val in row:
                if not isinstance(val, (float, int)):
                    raise ValueError('All input values must be float or int')
        return v


# noqa: E303
class PredictResponse(BaseModel):
    predictions: List[int]


@app.post('/predict', response_model=PredictResponse)
async def predict(request: PredictRequest):
    prediction_counter.inc()
    if not request.inputs:
        raise HTTPException(status_code=400, detail='No input data provided')
    X = np.array(request.inputs)
    # Always load the latest model from disk to ensure retrained model is used
    model_path = os.path.join(os.path.dirname(__file__), '../models/best_model/model.pkl')
    if os.path.exists(model_path):
        current_model = joblib.load(model_path)
    else:
        current_model = model
    preds = current_model.predict(X)
    # Log the request and prediction
    logging.info(f"Request: {request.inputs} | Prediction: {preds.tolist()}")
    # Log to SQLite
    cursor.execute("INSERT INTO logs (request, prediction) VALUES (?, ?)", (str(request.inputs), str(preds.tolist())))
    conn.commit()
    return PredictResponse(predictions=preds.tolist())

# Metrics endpoint
@app.get('/metrics')
def metrics():
    cursor.execute("SELECT COUNT(*) FROM logs")
    count = cursor.fetchone()[0]
    return {"prediction_requests": count}

# Prometheus endpoint
@app.get('/prometheus')
def prometheus_metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

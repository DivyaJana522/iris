
# Iris ML Project: End-to-End MLOps Demo

## Part 1: Repository and Data Versioning
1. **Clone the repository:**
   ```bash
   git clone https://github.com/DivyaJana522/iris.git
   cd iris
   ```
2. **Load and preprocess the dataset:**
   - Dataset: `data/iris.csv`
   - Preprocessing handled in `src/load_data.py`.
3. **Track the dataset:**
   - (Optional) Use DVC for versioning large datasets.
4. **Directory structure:**
   - `data/` for datasets
   - `models/` for saved models
   - `src/` for source code
   - `logs/` for logs and SQLite DB

## Part 2: Model Development & Experiment Tracking
1. **Train models:**
   ```bash
   python src/train_and_track.py
   ```
   - Trains Logistic Regression and RandomForest models.
2. **Track experiments with MLflow:**
   - MLflow logs params, metrics, and models in `mlruns/`.
   - View MLflow UI:
     ```bash
     mlflow ui
     # Open http://localhost:5000
     ```
   - Best model saved in `models/best_model/` and registered in MLflow.

## Part 3: API & Docker Packaging
1. **Run FastAPI prediction API locally:**
   ```bash
   uvicorn src.app:app --reload
   ```
2. **Test prediction endpoint:**
   ```bash
   curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"inputs": [[5.1, 3.5, 1.4, 0.2], [6.2, 2.8, 4.8, 1.8]]}'
   ```
3. **Build Docker image:**
   ```bash
   docker build -t iris-api .
   ```
4. **Run Docker container:**
   ```bash
   docker run -p 8000:8000 iris-api
   ```

## Part 4: CI/CD with GitHub Actions
1. **Lint and test on push:**
   - Flake8 and pytest run via `.github/workflows/ci.yml`.
2. **Build and push Docker image:**
   - Docker image pushed to Docker Hub on successful build.
3. **Deploy locally or to cloud:**
   - Use `deploy_local.sh` or `docker run` for local deployment.

## Part 5: Logging and Monitoring
1. **Logging:**
   - All prediction requests and outputs logged to `logs/prediction.log` and `logs/predictions.db` (SQLite).
2. **Monitoring:**
   - `/metrics` endpoint for basic monitoring.
   - `/prometheus` endpoint for Prometheus scraping.
   - Run Prometheus:
     ```bash
     ./prometheus --config.file=prometheus.yml
     # Access http://localhost:9090
     ```

### Example: Monitoring Endpoints

**Check metrics (total prediction requests):**
```bash
curl -X GET "http://localhost:8000/metrics"
```
Example response:
```json
{
   "prediction_requests": 5
}
```

**Check Prometheus metrics:**
```bash
curl -X GET "http://localhost:8000/prometheus"
```
Example output (snippet):
```
# HELP prediction_requests_total Total prediction requests
# TYPE prediction_requests_total counter
prediction_requests_total 5.0
```

### Example: Retrain Endpoint

**Retrain model with new data:**
```bash
curl -X POST "http://localhost:8000/retrain" \
   -H "Content-Type: application/json" \
   -d '{
      "inputs": [
         [5.1, 3.5, 1.4, 0.2],
         [7.0, 3.2, 4.7, 1.4],
         [6.3, 3.3, 6.0, 2.5],
         [5.0, 3.6, 1.4, 0.2],
         [6.7, 3.1, 4.4, 1.4],
         [5.9, 3.0, 5.1, 1.8]
      ],
      "targets": [0, 1, 2, 0, 1, 2]
   }'
```
Example response:
```json
{
   "status": "Retraining complete.",
   "best_model": "LogisticRegression",
   "accuracy": 1.0
}
```

## Bonus Features
- Input validation using Pydantic in API.
### Example: Pydantic Input Validation

**Valid input for /predict:**
```json
{
   "inputs": [
      [5.1, 3.5, 1.4, 0.2],
      [6.2, 2.8, 4.8, 1.8]
   ]
}
```

**Valid /predict request:**
```bash
curl -X POST "http://localhost:8000/predict" \
   -H "Content-Type: application/json" \
   -d '{
      "inputs": [
         [5.1, 3.5, 1.4, 0.2],
         [6.2, 2.8, 4.8, 1.8]
      ]
   }'
```

**Invalid /predict request (empty inputs):**
```bash
curl -X POST "http://localhost:8000/predict" \
   -H "Content-Type: application/json" \
   -d '{
      "inputs": []
   }'
```
Response: Error - "Input list cannot be empty."

**Invalid /predict request (wrong shape):**
```bash
curl -X POST "http://localhost:8000/predict" \
   -H "Content-Type: application/json" \
   -d '{
      "inputs": [[5.1, 3.5, 1.4]]
   }'
```
Response: Error - "Each input must be a list of 4 floats"

**Invalid /predict request (wrong type):**
```bash
curl -X POST "http://localhost:8000/predict" \
   -H "Content-Type: application/json" \
   -d '{
      "inputs": [[5.1, 3.5, "bad", 0.2]]
   }'
```
Response: Error - "All input values must be float or int"

## Bonus Features (continued)

### Pydantic Input Validation Examples

**Valid input for /predict:**
```json
{
   "inputs": [
      [5.1, 3.5, 1.4, 0.2],
      [6.2, 2.8, 4.8, 1.8]
   ]
}
```

**Invalid input examples (will trigger Pydantic validation errors):**
```json
{
   "inputs": []
}
```
Error: "Input list cannot be empty."

```json
{
   "inputs": [
      [5.1, 3.5, 1.4]  // Only 3 values instead of 4
   ]
}
```
Error: "Each input must be a list of 4 floats"

```json
{
   "inputs": [
      [5.1, 3.5, "bad", 0.2]
   ]
}
```
Error: "All input values must be float or int"

**Invalid input examples (will trigger Pydantic validation errors):**
```json
{
   "inputs": []
}
```
Error: "Input list cannot be empty."

```json
{
   "inputs": [
      [5.1, 3.5, 1.4]  // Only 3 values instead of 4
   ]
}
```
Error: "Each input must be a list of 4 floats"

```json
{
   "inputs": [
      [5.1, 3.5, "bad", 0.2]
   ]
}
```
Error: "All input values must be float or int"

- Prometheus integration and Streamlit dashboard (`streamlit_dashboard.py`).

```bash
streamlit run streamlit_dashboard.py
```

- (Optional) Model retraining endpoint (if enabled).

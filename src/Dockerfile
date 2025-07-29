# Use official Python image
FROM python:3.12-slim

# Set work directory
WORKDIR /app

# Copy requirements and install
COPY ../requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy app code and model
COPY . .
COPY ../models ./models

# Expose port
EXPOSE 5000

# Run the FastAPI app with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]

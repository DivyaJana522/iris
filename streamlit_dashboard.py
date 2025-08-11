
import streamlit as st
import requests
import re
import time
import pandas as pd


st.set_page_config(page_title="Iris API Metrics Dashboard", layout="wide")
st.title("Iris API Prometheus Metrics Dashboard")

# Fetch metrics from FastAPI
def fetch_metrics():
    try:
        response = requests.get("http://localhost:5000/prometheus")
        return response.text
    except Exception as e:
        st.error(f"Could not fetch metrics: {e}")
        return ""

metrics = fetch_metrics()


# Parse metrics
def parse_metric(metrics, metric_name, float_type=False):
    pattern = rf'{metric_name} ([0-9.e+-]+)'
    match = re.search(pattern, metrics)
    if match:
        return float(match.group(1)) if float_type else int(float(match.group(1)))
    return 0

count = parse_metric(metrics, "prediction_requests_total")
errors = parse_metric(metrics, "prediction_errors_total")  # If you log errors as a Prometheus counter
availability = 100.0 if count > 0 else 0.0  # Dummy availability metric
created_val = parse_metric(metrics, "prediction_requests_created", float_type=True)

col1, col2, col3 = st.columns(3)
col1.metric("Total Prediction Requests", count)
col2.metric("Total Errors", errors)
col3.metric("Availability (%)", availability)

# Time series chart for requests
if 'history' not in st.session_state:
    st.session_state['history'] = []
st.session_state['history'].append({"time": time.time(), "count": count, "errors": errors, "availability": availability})
df = pd.DataFrame(st.session_state['history'])
if not df.empty:
    st.line_chart(df.set_index('time')[["count", "errors", "availability"]])


if created_val:
    st.write(f"Metric Created (timestamp): {created_val}")

# Show raw Prometheus metrics output
st.subheader("Raw Prometheus Metrics Output")
st.code(metrics)



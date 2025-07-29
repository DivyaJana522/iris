
import os
import pandas as pd
from sklearn.datasets import load_iris

# Load Iris dataset
data = load_iris(as_frame=True)
df = data['frame']
df['target'] = data.target


# Ensure 'data' directory exists
os.makedirs(os.path.join(os.path.dirname(__file__), '../data'), exist_ok=True)

# Save to CSV
csv_path = os.path.join(os.path.dirname(__file__), '../data/iris.csv')
df.to_csv(csv_path, index=False)

print('Iris dataset saved to data/iris.csv')

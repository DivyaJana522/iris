
import os
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

# Iris dataset saved message (flake8: do not use print in production code)
# print('Iris dataset saved to data/iris.csv')

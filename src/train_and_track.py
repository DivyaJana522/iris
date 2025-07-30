
import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import mlflow
import mlflow.sklearn
import mlflow.tracking

# Load data
csv_path = os.path.join(os.path.dirname(__file__), '../data/iris.csv')
df = pd.read_csv(csv_path)
X = df.drop('target', axis=1)
y = df['target']


# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 1. Check class distribution
print('Train class distribution:', y_train.value_counts().to_dict())
print('Test class distribution:', y_test.value_counts().to_dict())

# MLflow experiment
mlflow.set_experiment('iris-classification')

models = {
    'LogisticRegression': LogisticRegression(max_iter=200),
    'RandomForest': RandomForestClassifier(n_estimators=100)
}

best_acc = 0
best_model = None
best_name = ''


for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        cm = confusion_matrix(y_test, preds)
        mlflow.log_param('model', name)
        mlflow.log_metric('accuracy', acc)
        mlflow.sklearn.log_model(model, 'model')
        print(f'{name} accuracy: {acc:.4f}')
        print(f'{name} confusion matrix:\n{cm}')
        # print(f'Predictions: {preds}')
        # print(f'True labels: {y_test.values}')
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name


# 3. Try cross-validation
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5)
    print(f'{name} CV accuracy: {scores.mean():.4f} \u00b1 {scores.std():.4f}')

# Save best model (remove if exists)
model_save_path = os.path.join(os.path.dirname(__file__), '../models/best_model')
if os.path.exists(model_save_path):
    shutil.rmtree(model_save_path)
mlflow.sklearn.save_model(best_model, model_save_path)
print(f'Best model: {best_name} with accuracy {best_acc:.4f}')

# Register the best model in MLflow Model Registry
client = mlflow.tracking.MlflowClient()
run_id = None
for run in client.search_runs(
    experiment_ids=[client.get_experiment_by_name("iris-classification").experiment_id],
    order_by=["metrics.accuracy DESC"]
):
    if run.data.params.get("model") == best_name:
        run_id = run.info.run_id
        break
if run_id:
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri, "IrisBestModel")
    print("Registered best model as 'IrisBestModel' in MLflow Model Registry.")
else:
    print("Could not find run ID for best model; registration skipped.")

import pandas as pd
import joblib
import yaml
import json
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

def evaluate():
    with open("src/config.yaml") as f:
        config = yaml.safe_load(f)

    model = joblib.load(config['test']['model_path'])
    test_df = pd.read_csv(config['test']['testset_path'])

    X_test = test_df.drop('charges', axis=1)
    y_test = test_df['charges']

    predictions = model.predict(X_test)
    
    metrics = {
        "r2": r2_score(y_test, predictions),
        "rmse": np.sqrt(mean_squared_error(y_test, predictions))
    }

    with open("report/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    evaluate()
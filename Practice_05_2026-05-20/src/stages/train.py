import pandas as pd
import joblib
import yaml
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

def train():
    with open("src/config.yaml") as f:
        config = yaml.safe_load(f)
        
    df_train = pd.read_csv(config['data_split']['trainset_path'])
    X = df_train.drop('charges', axis=1)
    y = df_train['charges']

    rf = RandomForestRegressor(random_state=42)
    params = {
        'n_estimators': config['train']['n_estimators'],
        'max_depth': config['train']['max_depth']
    }
    
    grid = GridSearchCV(rf, params, cv=config['train']['cv'])
    grid.fit(X, y)
    
    # Сохраняем модель
    os.makedirs("models", exist_ok=True)
    joblib.dump(grid.best_estimator_, config['train']['model_path'])

if __name__ == "__main__":
    train()
    
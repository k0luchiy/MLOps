import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
from mlflow.models import infer_signature
import joblib

CLEAN_DATA_PATH = "/tmp/insurance_clean.csv"
MODEL_SAVE_PATH = "/tmp/rf_insurance.pkl"

def scale_frame(frame):
    df = frame.copy()
    # Целевая переменная - расходы на страховку (charges)
    X, y = df.drop(columns=['charges']), df['charges']
    
    # Масштабируем только признаки X
    scaler = StandardScaler()
    X_scale = scaler.fit_transform(X)
    return X_scale, y.values, scaler

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def train():
    df = pd.read_csv(CLEAN_DATA_PATH)
    X, y, scaler = scale_frame(df)
    
    # Разбиваем на трейн и валидацию
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Сетка гиперпараметров для Случайного леса
    params = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5]
    }
    
    # Называем эксперимент в MLflow
    mlflow.set_experiment("insurance_medical_charges")
    
    with mlflow.start_run():
        rf = RandomForestRegressor(random_state=42)
        clf = GridSearchCV(rf, params, cv=3, n_jobs=-1)
        clf.fit(X_train, y_train)
        
        # Получаем лучшую модель
        best = clf.best_estimator_
        y_pred = best.predict(X_val)
        
        # Считаем метрики
        (rmse, mae, r2) = eval_metrics(y_val, y_pred)
        
        # Логируем лучшие параметры леса
        mlflow.log_param("n_estimators", best.n_estimators)
        mlflow.log_param("max_depth", best.max_depth)
        mlflow.log_param("min_samples_split", best.min_samples_split)
        
        # Логируем метрики
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        
        # Логируем саму модель в MLflow
        predictions = best.predict(X_train)
        signature = infer_signature(X_train, predictions)
        mlflow.sklearn.log_model(best, "model", signature=signature)
        
        # Сохраняем модель локально
        with open(MODEL_SAVE_PATH, "wb") as file:
            joblib.dump(best, file)
            
    return True

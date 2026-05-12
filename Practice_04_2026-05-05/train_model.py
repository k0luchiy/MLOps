import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from mlflow.models import infer_signature
import joblib

if __name__ == "__main__":
    # Загрузка данных
    df = pd.read_csv("./df_insurance_clear.csv")
    X = df.drop('charges', axis=1)
    y = df['charges']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.set_experiment("Medical_Insurance_Costs")
    
    with mlflow.start_run(run_name="Balanced_Model_Production"):
        # Используем лучшие параметры из ноутбука
        n_estimators = 100
        max_depth = 5
        
        rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        rf.fit(X_train, y_train)
        
        # Предсказания и метрики
        predictions = rf.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        # Логирование
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        
        # Сохранение сигнатуры и модели
        signature = infer_signature(X_test, predictions)
        mlflow.sklearn.log_model(rf, "model", signature=signature)
        
        # Получаем ID текущего запуска
        run_id = mlflow.active_run().info.run_id
        # Получаем локальный путь к артефакту 'model'
        # Это самый надежный способ для MLflow
        local_model_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model")
        
        # Записываем ТОЛЬКО путь в файл, без лишних принтов
        with open("best_model.txt", "w") as f:
            f.write(local_model_path)
            
        print(f"\nSUCCESS: Model saved to {local_model_path}")
from datetime import datetime
import json
import logging
import os
import pandas as pd

from airflow.sdk import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.timetables.interval import CronDataIntervalTimetable
from hooks import CarsHook  # ← импортируем из plugins/hooks/

def _fetch_cars(conn_id: str, templates_dict: dict, batch_size: int = 1000, **_):
    logger = logging.getLogger(__name__)
    output_path = templates_dict["output_path"]

    logger.info("Fetching all cars from the API...")
    hook = CarsHook(conn_id=conn_id)
    cars = list(hook.get_cars(batch_size=batch_size))
    logger.info(f"Fetched {len(cars)} car records")

    # Убедимся, что директория существует
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(cars, f)

    logger.info(f"Saved cars to {output_path}")

def _clean_cars_data(templates_dict: dict, **_):
    logger = logging.getLogger(__name__)
    input_path = templates_dict["input_path"]
    output_path = templates_dict["output_path"]

    logger.info(f"Loading raw data from {input_path}")
    df = pd.read_json(input_path)

    # 1. Удаление дубликатов и пропусков
    initial_len = len(df)
    df = df.dropna()
    df = df.drop_duplicates()
    logger.info(f"Dropped {initial_len - len(df)} rows (duplicates or NaNs).")

    # Преобразование категориальных признаков в числовые
    # Находим все колонки с текстовым типом данных (object/string)
    categorical_cols = df.select_dtypes(include=['object', 'string']).columns 
    for col in categorical_cols:
        df[col] = pd.factorize(df[col])[0]

    logger.info(f"Categorical columns converted to numeric: {list(categorical_cols)}")

    # 3. Сохрание очищенного датасета
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    df.to_json(output_path, orient="records")
    logger.info(f"Cleaned data saved to {output_path}")

with DAG(
    dag_id="02_hook",
    description="Fetches car data from the custom API using a custom hook.",
    start_date=datetime(2026, 2, 3),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
) as dag:

    fetch_task = PythonOperator(
        task_id="fetch_cars",
        python_callable=_fetch_cars,
        op_kwargs={"conn_id": "carsapi"},  # ← имя Airflow Connection
        templates_dict={
            "output_path": "/data/custom_hook/cars.json",
        },
    )

    clean_task = PythonOperator(
        task_id="clean_cars_data",
        python_callable=_clean_cars_data,
        templates_dict={
            "input_path": "/data/custom_hook/cars.json",
            "output_path": "/data/cleaned/cars_cleaned.json",
        }
    )

    fetch_task >> clean_task
    
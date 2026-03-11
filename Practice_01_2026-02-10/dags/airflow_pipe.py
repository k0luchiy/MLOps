
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
import os

# Импортируем функцию из второго файла
from train_model import train

# Задаем пути для сохранения
RAW_DATA_PATH = "/tmp/insurance.csv"
CLEAN_DATA_PATH = "/tmp/insurance_clean.csv"

def download_data():
    # Прямая ссылка на датасет по медицинским страховкам
    url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
    df = pd.read_csv(url)
    df.to_csv(RAW_DATA_PATH, index=False)
    print("Downloaded data shape: ", df.shape)
    return True

def clear_data():
    df = pd.read_csv(RAW_DATA_PATH)
    
    # 1. Удаление полных дубликатов
    df = df.drop_duplicates()
    
    # 2. Очистка по здравому смыслу (удаление экстремальных выбросов)
    # Индекс массы тела (BMI) больше 50 - редкая аномалия
    df = df[df['bmi'] < 50]
    
    # Расходы более 55000 встречаются крайне редко, отфильтруем их
    df = df[df['charges'] < 55000]
    
    df = df.reset_index(drop=True)
    
    # 3. Предобработка категориальных признаков
    cat_columns = ['sex', 'smoker', 'region']
    
    ordinal = OrdinalEncoder()
    df[cat_columns] = ordinal.fit_transform(df[cat_columns])
    
    # Сохраняем очищенные данные (обязательно index=False)
    df.to_csv(CLEAN_DATA_PATH, index=False)
    print("Cleaned data shape: ", df.shape)
    return True

# Создаем DAG
dag_insurance = DAG(
    dag_id="insurance_train_pipe", # Новое имя DAG
    start_date=datetime(2025, 2, 3),
    max_active_tasks=4,
    schedule="@hourly",
    max_active_runs=1,
    catchup=False,
)

# Задачи
download_task = PythonOperator(python_callable=download_data, task_id="download_insurance", dag=dag_insurance)
clear_task = PythonOperator(python_callable=clear_data, task_id="clear_insurance", dag=dag_insurance)
train_task = PythonOperator(python_callable=train, task_id="train_insurance_model", dag=dag_insurance)

# Порядок выполнения
download_task >> clear_task >> train_task

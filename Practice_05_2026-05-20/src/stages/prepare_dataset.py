import pandas as pd
import yaml
from sklearn.preprocessing import LabelEncoder
import os

def load_config(config_path):
    with open(config_path) as conf_file:
        return yaml.safe_load(conf_file)

def clear_data(df):
    # В страховке данных обычно мало, можно просто удалить дубликаты
    df = df.drop_duplicates()
    # Кодирование категориальных признаков (как ты делал в лабе №3)
    le = LabelEncoder()
    for col in ['sex', 'smoker', 'region']:
        df[col] = le.fit_transform(df[col])
    return df

def featurize(df, config):
    # Этап 2: Генерация новых признаков
    # Например, создадим признак "overweight" на основе BMI
    df['is_obese'] = (df['bmi'] >= 30).astype(int)
    # Взаимодействие возраста и курения
    df['age_smoker_interaction'] = df['age'] * df['smoker']
    
    df.to_csv(config['featurize']['features_path'], index=False)

if __name__ == "__main__":
    config = load_config("src/config.yaml")
    raw_data = pd.read_csv(config['data_load']['dataset_csv'])
    clean_df = clear_data(raw_data)
    featurize(clean_df, config)
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def download_data():
    url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
    df = pd.read_csv(url)
    df.to_csv("insurance.csv", index=False)
    return df

def clear_data(path2df):
    df = pd.read_csv(path2df)
    
    # Кодирование категориальных признаков
    le = LabelEncoder()
    for col in ['sex', 'smoker', 'region']:
        df[col] = le.fit_transform(df[col])
    
    # Сохраняем предобработанные данные
    df.to_csv('df_insurance_clear.csv', index=False)
    return True

if __name__ == "__main__":
    download_data()
    clear_data("insurance.csv")
    print("Data downloaded and preprocessed.")
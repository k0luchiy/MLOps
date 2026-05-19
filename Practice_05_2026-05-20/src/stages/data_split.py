import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

def data_split():
    with open("src/config.yaml") as f:
        config = yaml.safe_load(f)

    df = pd.read_csv(config['featurize']['features_path'])
    train, test = train_test_split(df, test_size=config['data_split']['test_size'], random_state=42)

    train.to_csv(config['data_split']['trainset_path'], index=False)
    test.to_csv(config['data_split']['testset_path'], index=False)

if __name__ == "__main__":
    data_split()
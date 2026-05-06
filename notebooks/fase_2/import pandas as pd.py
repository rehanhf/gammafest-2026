import pandas as pd
train = pd.read_csv('./data/processed/train_cleaned.csv', dtype={'date': str}, nrows=5)
print(train.columns.tolist())
print(train.head(2))
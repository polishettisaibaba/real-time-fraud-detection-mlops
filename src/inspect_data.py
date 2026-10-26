import pandas as pd

file_path = "data/raw/train_transaction.csv"

df = pd.read_csv(file_path,nrows=5)

print(df.head())
print("Shape:", df.shape)

# Inspect the dataset's columns and data types :

print("\n First 15 column names:")
print(df.columns[:15].tolist())

print("\n Data types of first 15 columns:")
print(df.dtypes.head(15))

print("\n Missing values in first 15 columns:")
print(df.iloc[:, :15].isnull().sum())

#Read the complete isFraud Column

fraud_labels = pd.read_csv(file_path,usecols=["isFraud"])

print("\n Full dataset shape :", fraud_labels.shape)

print("\n Fraud class count :")
print(fraud_labels["isFraud"].value_counts())

print("\n Fraud class perencentage:")
print(
    fraud_labels["isFraud"].value_counts(normalize=True)*100
)



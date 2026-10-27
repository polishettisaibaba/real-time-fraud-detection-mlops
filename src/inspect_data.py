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

# Inspect sample memory usage 

sample_df = pd.read_csv(file_path,nrows= 1000)

print("\n Sample shape:", sample_df.shape)

print("\n Sample Memory usage in (MiB)")
print(
    sample_df.memory_usage(deep=True).sum()/(1024**2)
)

# Missing Values assessment 
# Missing Values in the 1000 -rows

missing_count = sample_df.isnull().sum()

missing_percentage = (
    sample_df.isnull().mean()*100
)

missing_summary = pd.DataFrame({
    "Missing_count" : missing_count,
    "Missing_Percentage": missing_percentage
})

missing_summary = missing_summary.sort_values(
    by = "Missing_Percentage",
    ascending= False
)

print("\n Top 10 Columns with highest missing percentage")
print(missing_summary.head(10))

#Read the Full Data set in chunks 
# Full data set Missing values analysis

chunk_size = 50000

total_missing = None
total_rows =0

for chunk_number,chunk in enumerate(
    pd.read_csv(file_path, chunksize=chunk_size),
    start=1
):
    
    chunk_missing = chunk.isnull().sum()
    
    if total_missing is None :
        total_missing = chunk_missing
    else:
        total_missing = total_missing.add(chunk_missing,fill_value=0)
        
    total_rows += len(chunk)
    
    print("Processed chunk :", chunk_number)
    
print("\n Total rows processed :", total_rows)


full_missing_percentage = (
    total_missing/total_rows
)*100

full_missing_summary = pd.DataFrame({
    "Missing_Count" : total_missing.astype(int),
    "Missing_Percentage" : full_missing_percentage
})


full_missing_summary = full_missing_summary.sort_values(
    by="Missing_Percentage",
    ascending= False
)

print("\n Top 10 Missing columns - full dataset")
print(full_missing_summary.head(10))

## Over all missing assessment

print("\n Total Columns:", len(full_missing_summary))

print(
    "Columns with more than 50% Missing : ",
    (full_missing_summary["Missing_Percentage"]>50).sum()
)

print(
    "Columns with more than 80% Missing:",
    (full_missing_summary["Missing_Percentage"]>80).sum()
)

print(
    "Columns with more than 90% Missing:",
    (full_missing_summary["Missing_Percentage"]>90).sum()
)

print(
    "Columns with NO MISSING values:",
    (full_missing_summary["Missing_Percentage"]==0).sum()
)
        
    
   

  










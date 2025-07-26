import pandas as pd

# Load crop yield data
df_crop = pd.read_csv("data/raw/state_crop_yield_states1997_2020.csv")

print("Data Shape:", df_crop.shape)
print("\nColumns:", df_crop.columns)
print("\nSample Data:\n", df_crop.head())
print("\nMissing values:\n", df_crop.isnull().sum())

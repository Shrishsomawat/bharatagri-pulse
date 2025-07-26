import pandas as pd

# Load rainfall data
df_rain = pd.read_csv("data/raw/subdivision_monthly_rainfall_1901_2017.csv")

print("Data Shape:", df_rain.shape)
print("\nColumns:", df_rain.columns)
print("\nSample Data:\n", df_rain.head())
print("\nMissing values:\n", df_rain.isnull().sum())

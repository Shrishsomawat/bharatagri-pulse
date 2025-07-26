import pandas as pd

# Load raw data
df = pd.read_csv("data/raw/subdivision_monthly_rainfall_1901_2017.csv")

# 🔹 Step 1: Clean column names
df.columns = df.columns.str.strip().str.upper()

# 🔹 Step 2: Drop rows with too many missing values
df_clean = df.dropna(thresh=12)  # If more than 7 nulls, drop

# 🔹 Step 3: Fill missing values with regional monthly averages
monthly_cols = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
                'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

for col in monthly_cols:
    df_clean[col] = df_clean.groupby('SUBDIVISION')[col].transform(lambda x: x.fillna(x.mean()))

# 🔹 Step 4: Recalculate ANNUAL, JF, MAM, JJAS, OND from monthly
df_clean['JF'] = df_clean['JAN'] + df_clean['FEB']
df_clean['MAM'] = df_clean['MAR'] + df_clean['APR'] + df_clean['MAY']
df_clean['JJAS'] = df_clean['JUN'] + df_clean['JUL'] + df_clean['AUG'] + df_clean['SEP']
df_clean['OND'] = df_clean['OCT'] + df_clean['NOV'] + df_clean['DEC']
df_clean['ANNUAL'] = df_clean['JF'] + df_clean['MAM'] + df_clean['JJAS'] + df_clean['OND']

# 🔹 Step 5: Save cleaned data
df_clean.to_csv("data/cleaned/cleaned_rainfall.csv", index=False)

print("✅ Rainfall data cleaned and saved!")

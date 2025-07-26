import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

# Step 1: Load merged data
data_path = "C:/Users/ssomawat/Desktop/bharatagri-pulse/data/cleaned/final_merged_data.csv"
data = pd.read_csv(data_path)

# Step 1.5: Convert YIELD to kg/ha if needed
data['YIELD'] = data['YIELD'] * 1000


# Step 2: Define columns
categorical_cols = ['CROP', 'SEASON', 'STATE']
numerical_cols = ['AREA', 'FERTILIZER', 'PESTICIDE']
rainfall_cols = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC', 'ANNUAL', 'JF', 'MAM', 'JJAS', 'OND']

# Step 3: Combine numerical + rainfall
full_numerical = numerical_cols + rainfall_cols

# Step 4: Encode categoricals
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_cats = encoder.fit_transform(data[categorical_cols])
encoded_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_cols))

# Step 5: Final dataset
X = pd.concat([encoded_df.reset_index(drop=True), data[full_numerical].reset_index(drop=True)], axis=1)
y = data['YIELD']

# Step 6: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 8: Evaluate
y_pred = model.predict(X_test)
print("🔍 Evaluation Metrics:")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")

# Step 9: Save model and encoder
model_dir = "C:/Users/ssomawat/Desktop/bharatagri-pulse/models"
os.makedirs(model_dir, exist_ok=True)

joblib.dump(model, os.path.join(model_dir, "yield_model.pkl"))
joblib.dump(encoder, os.path.join(model_dir, "encoder.pkl"))

print("✅ Model and encoder saved successfully to:")
print(f"   🔹 {model_dir}/yield_model.pkl")
print(f"   🔹 {model_dir}/encoder.pkl")

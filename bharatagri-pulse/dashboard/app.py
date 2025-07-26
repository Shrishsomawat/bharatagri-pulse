import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# === Page Config ===
st.set_page_config(page_title="BharatAgri Pulse", layout="centered")
st.title("🌾 BharatAgri Pulse: Crop Yield Prediction")
st.markdown("📍 Select your inputs below to predict **Crop Yield (in kg/ha)**.")

# === Load Model and Encoder ===
model = joblib.load("C:/Users/ssomawat/Desktop/bharatagri-pulse/models/yield_model.pkl")
encoder = joblib.load("C:/Users/ssomawat/Desktop/bharatagri-pulse/models/encoder.pkl")

# === Input UI ===
state = st.selectbox("🗺️ State", ['Assam', 'Punjab', 'Maharashtra', 'Karnataka', 'Tamil Nadu'])
season = st.selectbox("🕒 Season", ['Kharif', 'Rabi', 'Summer', 'Whole Year', 'Autumn', 'Winter'])
crop = st.selectbox("🌿 Crop", ['Rice', 'Wheat', 'Maize', 'Sugarcane', 'Cotton(lint)', 'Arecanut'])

area = st.number_input("📐 Area (in hectares)", min_value=1.0, value=100.0)
fertilizer = st.number_input("🧪 Fertilizer Used (kg)", min_value=0.0, value=500.0)
pesticide = st.number_input("🦟 Pesticide Used (kg)", min_value=0.0, value=50.0)

# === Monthly Rainfall ===
st.markdown("🌧️ **Monthly Rainfall (mm)**")
rainfall_inputs = {}
months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
for month in months:
    rainfall_inputs[month] = st.number_input(f"🌦️ {month}", min_value=0.0, value=100.0)

# === Derived Rainfall Features ===
rainfall_inputs['ANNUAL'] = sum(rainfall_inputs[m] for m in months)
rainfall_inputs['JF'] = rainfall_inputs['JAN'] + rainfall_inputs['FEB']
rainfall_inputs['MAM'] = rainfall_inputs['MAR'] + rainfall_inputs['APR'] + rainfall_inputs['MAY']
rainfall_inputs['JJAS'] = rainfall_inputs['JUN'] + rainfall_inputs['JUL'] + rainfall_inputs['AUG'] + rainfall_inputs['SEP']
rainfall_inputs['OND'] = rainfall_inputs['OCT'] + rainfall_inputs['NOV'] + rainfall_inputs['DEC']

# === Create Input DataFrame ===
input_df = pd.DataFrame({
    'CROP': [crop],
    'SEASON': [season],
    'STATE': [state],
    'AREA': [area],
    'FERTILIZER': [fertilizer],
    'PESTICIDE': [pesticide],
    **{key: [val] for key, val in rainfall_inputs.items()}
})

# === Encode Categorical Features ===
encoded = encoder.transform(input_df[['CROP', 'SEASON', 'STATE']])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['CROP', 'SEASON', 'STATE']))

# === Final Input ===
numerical_cols = ['AREA', 'FERTILIZER', 'PESTICIDE'] + list(rainfall_inputs.keys())
final_input = pd.concat([encoded_df.reset_index(drop=True), input_df[numerical_cols].reset_index(drop=True)], axis=1)

# === Predict Yield ===
if st.button("🔍 Predict Yield"):
    try:
        prediction = model.predict(final_input)[0]
        st.success(f"🌾 Estimated Yield: **{prediction:.2f} kg/ha**")
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")

# === Feature Importance Chart ===
st.subheader("📊 Feature Importance (Top 15 Factors Influencing Prediction)")

# Get feature names
categorical_cols = ['CROP', 'SEASON', 'STATE']
encoded_feature_names = encoder.get_feature_names_out(categorical_cols)
numerical_cols = ['AREA', 'FERTILIZER', 'PESTICIDE'] + months + ['ANNUAL', 'JF', 'MAM', 'JJAS', 'OND']
feature_names = list(encoded_feature_names) + numerical_cols

# Build importance dataframe
importances = model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by="Importance", ascending=False).head(15)

# Plot it
fig = px.bar(importance_df, x="Importance", y="Feature", orientation='h',
             title="Top 15 Most Important Features in Yield Prediction",
             labels={'Importance': 'Importance Score'},
             height=600)

fig.update_layout(yaxis=dict(autorange="reversed"))
st.plotly_chart(fig)

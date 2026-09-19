# BharatAgri Pulse

A crop yield prediction dashboard for Indian agriculture. It combines state-wise crop production data (1997–2020) with subdivision monthly rainfall data (1901–2017), trains a Random Forest model, and serves predictions through an interactive Streamlit app.

## Features

- Data pipeline to fetch, clean and merge crop-yield and rainfall datasets
- Random Forest regression model with one-hot encoded categorical features
- Streamlit dashboard: choose the state, season, crop, area, fertilizer, pesticide and monthly rainfall to get a predicted yield in kg/ha
- Plotly charts for exploring the inputs

## Project structure

```
bharatagri-pulse/
├── dashboard/app.py        # Streamlit app
├── data/
│   ├── raw/                # Source datasets
│   └── cleaned/            # Cleaned and merged data
├── models/                 # Trained model and encoder (.pkl)
└── scripts/
    ├── fetch_crop_yield.py
    ├── fetch_rainfall.py
    ├── clean_data.py
    ├── merge_data.py
    └── train_model.py
```

## Tech stack

Python · pandas · scikit-learn · Streamlit · Plotly · joblib

## Getting started

```bash
git clone https://github.com/Shrishsomawat/bharatagri-pulse.git
cd bharatagri-pulse/bharatagri-pulse
pip install -r requirements.txt

# (Optional) retrain the model
python scripts/train_model.py

# Launch the dashboard
streamlit run dashboard/app.py
```

## Data sources

- State-wise crop production statistics, India (1997–2020)
- IMD subdivision-wise monthly rainfall, India (1901–2017)

import pandas as pd
import os

# ✅ Step 1: Load the cleaned rainfall data and raw crop yield data
rain = pd.read_csv("data/cleaned/cleaned_rainfall.csv")
crop = pd.read_csv("data/raw/state_crop_yield_states1997_2020.csv")
region_map = pd.read_csv("data/raw/region_mapping.csv")
# region_map.columns = region_map.columns.str.strip()


# ✅ Step 2: Strip all column names to remove trailing/leading spaces
rain.columns = rain.columns.str.strip()
crop.columns = crop.columns.str.strip()
region_map.columns = region_map.columns.str.strip()

# ✅ Step 3: Merge rainfall data with region mapping to attach 'STATE'
rain_mapped = rain.merge(region_map, on="SUBDIVISION")
rain_mapped.columns = rain_mapped.columns.str.strip()

# Step 4: Group rainfall by STATE and YEAR (averaging values across subdivisions)
rain_mapped = rain.merge(region_map, on="SUBDIVISION")
rain_grouped = rain_mapped.groupby(["STATE", "YEAR"]).mean(numeric_only=True).reset_index()

# Step 5: Clean crop column names
crop.rename(columns={"Crop_Year": "YEAR", "State": "STATE"}, inplace=True)
crop.columns = crop.columns.str.strip().str.upper()
rain_grouped.columns = rain_grouped.columns.str.strip().str.upper()

# DEBUG PRINTS
print("CROP COLUMNS:", crop.columns.tolist())
print("RAIN GROUPED COLUMNS:", rain_grouped.columns.tolist())

# Step 6: Merge final
merged = crop.merge(rain_grouped, on=["STATE", "YEAR"], how="inner")


# ✅ Step 7: Save final merged data
os.makedirs("data/cleaned", exist_ok=True)
merged.to_csv("data/cleaned/final_merged_data.csv", index=False)

print("✅ Data merged successfully!")
print("Final Shape:", merged.shape)
print("Sample Rows:\n", merged.head())

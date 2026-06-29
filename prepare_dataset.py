import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SOURCE_CSV = DATA_DIR / "used_car_price_prediction_1M - Copy.csv"
TARGET_CSV = DATA_DIR / "used_car_price_prediction.csv"

if not SOURCE_CSV.exists():
    raise FileNotFoundError(
        f"Source file not found: {SOURCE_CSV}\n"
        "Place the original dataset file named 'used_car_price_prediction_1M - Copy.csv' in the data/ folder, "
        "or update SOURCE_CSV in this script to match the file name."
    )

# Load dataset
print(f"Loading source dataset from {SOURCE_CSV}")
df = pd.read_csv(SOURCE_CSV)

# Feature engineering
print("Creating generated features...")
df["Car_Age"] = 2026 - df["Year"]
df["Price_Difference"] = df["Present_Price"] - df["Selling_Price"]
df["PricePerKm"] = df["Selling_Price"] / (df["Kms_Driven"] + 1)
df["Log_Kms_Driven"] = np.log1p(df["Kms_Driven"])
df["PricePerYear"] = df["Selling_Price"] / df["Car_Age"]
df["PricePerOwner"] = df["Selling_Price"] / (df["Owner"] + 1)

# Encode categorical variables
print("Encoding categorical variables...")
df["Fuel_Type_Encoded"] = df["Fuel_Type"].map({"Petrol": 0, "Diesel": 1, "CNG": 2})
df["Seller_Type_Encoded"] = df["Seller_Type"].map({"Dealer": 0, "Individual": 1})
df["Transmission_Encoded"] = df["Transmission"].map({"Manual": 0, "Automatic": 1})

# Extract brand info
print("Extracting brand information...")
df["Car_Brand"] = df["Car_Name"].str.split(" ").str[0]
df["Brand_Frequency"] = df.groupby("Car_Brand")["Car_Brand"].transform("count")

# Save updated dataset
print(f"Saving prepared dataset to {TARGET_CSV}")
df.to_csv(TARGET_CSV, index=False)
print("✅ Updated dataset saved as used_car_price_prediction.csv")

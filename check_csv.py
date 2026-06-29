import os
import pandas as pd

file_path = "data/used_car_price_prediction.csv"

print("File exists:", os.path.exists(file_path))

if os.path.exists(file_path):
    try:
        df = pd.read_csv(file_path)
        print("Shape:", df.shape)
        print("Head:\n", df.head())
    except Exception as e:
        print("Error reading CSV:", e)

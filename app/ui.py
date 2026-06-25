import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.title("Used Car Price Prediction")

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "ensemble_model.pkl"
try:
    model = joblib.load(MODEL_PATH)
except Exception as exc:
    st.error(f"Unable to load model from {MODEL_PATH}: {exc}")
    st.stop()

st.write("Enter car details to predict selling price:")

brand = st.selectbox(
    "Brand",
    [
        "Maruti",
        "Hyundai",
        "Honda",
        "Toyota",
        "Ford",
        "Mahindra",
        "Tata",
        "BMW",
        "Mercedes",
        "Audi",
    ],
)
year = st.number_input("Year", min_value=1990, max_value=2026, value=2015)
present_price = st.number_input("Present Price (in lakhs)", min_value=0.0, value=5.0)
kms_driven = st.number_input("Kms Driven", min_value=0, value=20000)
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

input_data = pd.DataFrame(
    {
        "Brand": [brand],
        "Year": [year],
        "Present_Price": [present_price],
        "Kms_Driven": [kms_driven],
        "Fuel_Type": [fuel_type],
        "Seller_Type": [seller_type],
        "Transmission": [transmission],
    }
)

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Selling Price: ₹{prediction:.2f} lakhs")

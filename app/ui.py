from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import json

# Import database functions
try:
    from app.database import init_db, save_prediction, get_all_predictions
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parent))
    from database import init_db, save_prediction, get_all_predictions

# Initialize DB
init_db()

# Page setup
st.set_page_config(page_title="Used Car Price Prediction", layout="centered")

# ------------------- CSS Styling -------------------
st.markdown(
    """
    <style>
    :root { color-scheme: light; }
    html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background: linear-gradient(135deg, #f8fbff 0%, #eef7ff 50%, #ffffff 100%);
        color: #17324a;
    }
    [data-testid="stHeader"] { background: rgba(255, 255, 255, 0); }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; }
    .hero-card {
        background: linear-gradient(135deg, #ffffff 0%, #f2f8ff 100%);
        border: 1px solid #dceefc;
        border-radius: 24px;
        padding: 1.35rem 1.5rem;
        box-shadow: 0 14px 34px rgba(15, 76, 129, 0.14);
        margin-bottom: 1rem;
        animation: riseIn 0.8s ease;
    }
    .hero-title { color: #0f4c81; font-size: 2.25rem; font-weight: 800; margin-bottom: 0.4rem; }
    .hero-subtitle { color: #4f6880; font-size: 1rem; margin-bottom: 0.6rem; }
    .badge {
        display: inline-block; padding: 0.35rem 0.7rem; border-radius: 999px;
        background: #eaf6ff; color: #145a8a; font-weight: 600;
        margin-right: 0.4rem; margin-top: 0.25rem;
    }
    .model-card, .result-card, .logic-card {
        background: white; border: 1px solid #e4eef9; border-radius: 18px;
        padding: 0.95rem 1rem; box-shadow: 0 8px 18px rgba(15, 76, 129, 0.08);
        margin-bottom: 1rem; animation: riseIn 0.9s ease;
    }
    .result-card { border-left: 6px solid #2b8fda; }
    .stButton > button {
        background: linear-gradient(90deg, #0f6fbf 0%, #2b8fda 100%);
        color: white; border: none; border-radius: 999px;
        padding: 0.75rem 1.3rem; font-weight: 700;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px); box-shadow: 0 8px 16px rgba(15, 76, 129, 0.2);
    }
    .stSuccess {
        background: #ecfdf3 !important; color: #166534 !important;
        border: 1px solid #a7f3d0 !important; border-radius: 12px;
        padding: 0.8rem; animation: bounceIn 0.8s ease;
    }
    label, .stTextInput label, .stNumberInput label, .stSelectbox label {
        color: #21445d; font-weight: 600;
    }
    @keyframes riseIn { from { opacity: 0; transform: translateY(16px); } to { opacity: 1; transform: translateY(0); } }
    @keyframes bounceIn { 0% { transform: scale(0.96); opacity: 0; } 70% { transform: scale(1.02); opacity: 1; } 100% { transform: scale(1); } }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------- Hero Section -------------------
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">Car Price Prediction</div>
        <div class="hero-subtitle">Estimate resale value with a tuned machine learning model and a practical market-style adjustment.</div>
        <span class="badge">Light UI</span>
        <span class="badge">Animated design</span>
        <span class="badge">Advanced prediction logic</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------- Load Model -------------------
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "ensemble_model.pkl"
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "used_car_price_prediction.csv"

AVAILABLE_FEATURES = [
    "Year", "Present_Price", "Kms_Driven", "Fuel_Type", "Seller_Type", "Transmission", "Brand"
]
MODEL_FEATURES = [
    "Year", "Present_Price", "Kms_Driven", "Fuel_Type", "Seller_Type", "Transmission", "Car_Brand"
]
BRAND_COLUMN = "Car_Brand"


def load_brand_info():
    df = pd.read_csv(DATA_PATH)
    if BRAND_COLUMN in df.columns:
        brand_series = df[BRAND_COLUMN].astype(str)
    elif "Car_Name" in df.columns:
        brand_series = df["Car_Name"].str.split(" ").str[0].fillna("Unknown").astype(str)
    elif "Brand" in df.columns:
        brand_series = df["Brand"].astype(str)
    else:
        brand_series = pd.cut(df["Present_Price"], bins=[-1, 3, 5, 7, 10, 15, 20, 30, 45, 60, 1000], labels=[
            "Maruti", "Hyundai", "Honda", "Toyota", "Ford", "Mahindra", "Tata", "BMW", "Mercedes", "Audi"
        ])
        brand_series = brand_series.fillna("Maruti").astype(str)

    df = df.assign(Car_Brand=brand_series)
    brand_stats = df.groupby("Car_Brand").agg(count=("Price", "size"), mean_price=("Price", "mean"))
    overall_mean = df["Price"].mean()
    adjustments = (brand_stats["mean_price"] - overall_mean).round(3).to_dict()
    available = sorted(brand_stats.index.tolist())
    return available, adjustments, brand_stats


BRAND_OPTIONS, BRAND_ADJUSTMENTS, BRAND_STATS = load_brand_info()
FEATURE_COLUMNS = MODEL_FEATURES

st.subheader("Features used in prediction")
selected_features = st.multiselect(
    "Select features to include",
    AVAILABLE_FEATURES,
    default=AVAILABLE_FEATURES,
    help="Unchecked features use default values for prediction but will not appear in the input form.",
)
if not selected_features:
    selected_features = AVAILABLE_FEATURES.copy()
    st.warning("At least one feature must be selected. Using all features.")

st.markdown(
    """
    The model uses these features for price prediction:
    """
)
st.write(", ".join(AVAILABLE_FEATURES))

if BRAND_OPTIONS:
    st.markdown("### Brand values seen in dataset")
    st.write(", ".join(BRAND_OPTIONS))

st.markdown("---")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def get_price_band(price: float) -> str:
    if price < 4:
        return "Budget"
    if price < 8:
        return "Mid-range"
    return "Premium"


def predict_with_logic(model, input_df: pd.DataFrame):
    raw_prediction = float(model.predict(input_df)[0])
    year = int(input_df["Year"].iloc[0])
    kms = int(input_df["Kms_Driven"].iloc[0])
    fuel_type = input_df["Fuel_Type"].iloc[0]
    seller_type = input_df["Seller_Type"].iloc[0]
    transmission = input_df["Transmission"].iloc[0]
    brand = input_df["Car_Brand"].iloc[0]
    present_price = float(input_df["Present_Price"].iloc[0])

    adjustment = 0.0
    age = 2026 - year
    adjustment -= max(0, age - 3) * 0.08
    adjustment -= kms / 50000 * 0.06

    if fuel_type == "Diesel":
        adjustment += 0.10
    elif fuel_type == "CNG":
        adjustment -= 0.16

    if seller_type == "Dealer":
        adjustment += 0.08
    if transmission == "Automatic":
        adjustment += 0.12
    adjustment += BRAND_ADJUSTMENTS.get(brand, 0.0)
    if present_price > 8:
        adjustment += 0.05

    adjusted_prediction = max(0.8, raw_prediction + adjustment)
    return raw_prediction, adjusted_prediction


def explain_prediction(year, kms, fuel_type, seller_type, transmission, present_price, brand):
    reasons = []
    age = 2026 - year
    if age <= 3:
        reasons.append("Newer cars usually command better resale value.")
    elif age >= 8:
        reasons.append("Older cars are typically penalized for depreciation.")

    if kms < 20000:
        reasons.append("Lower mileage usually increases buyer confidence.")
    elif kms > 60000:
        reasons.append("Higher mileage often reduces resale appeal.")

    if fuel_type == "Diesel":
        reasons.append("Diesel variants often hold value well in some segments.")
    elif fuel_type == "CNG":
        reasons.append("CNG cars are usually priced more conservatively.")

    if seller_type == "Dealer":
        reasons.append("Dealer-listed cars often reflect a more structured resale market.")
    if transmission == "Automatic":
        reasons.append("Automatic variants can attract a premium in many markets.")
    brand_adj = BRAND_ADJUSTMENTS.get(brand, 0.0)
    if brand_adj > 0:
        reasons.append(f"{brand} is treated as a stronger brand in this pricing model.")
    elif brand_adj < 0:
        reasons.append(f"{brand} is treated as a more budget-friendly brand in this pricing model.")
    if present_price > 8:
        reasons.append("A higher current listed price usually supports a stronger valuation.")

    return reasons

if not MODEL_PATH.exists():
    st.error(f"Model file not found at {MODEL_PATH}. Please run the training script first.")
    st.stop()

model = load_model()

# ------------------- Input Form -------------------
st.subheader("Car Details")
col1, col2 = st.columns(2)

with col1:
    year = st.number_input("Year", min_value=1990, max_value=2026, value=2015)
    present_price = st.number_input("Present Price (in lakhs)", min_value=0.0, value=5.0)
    kms_driven = st.number_input("Kms Driven", min_value=0, value=20000)

defaults = {
    "Year": 2015,
    "Present_Price": 5.0,
    "Kms_Driven": 20000,
    "Fuel_Type": "Petrol",
    "Seller_Type": "Dealer",
    "Transmission": "Manual",
    "Car_Brand": BRAND_OPTIONS[0] if BRAND_OPTIONS else "Unknown",
}

with col2:
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"]) if "Fuel_Type" in selected_features else defaults["Fuel_Type"]
    seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"]) if "Seller_Type" in selected_features else defaults["Seller_Type"]
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"]) if "Transmission" in selected_features else defaults["Transmission"]
    brand = st.selectbox("Brand", BRAND_OPTIONS) if "Brand" in selected_features else defaults["Car_Brand"]

input_data = pd.DataFrame([{
    "Year": year,
    "Present_Price": present_price,
    "Kms_Driven": kms_driven,
    "Fuel_Type": fuel_type,
    "Seller_Type": seller_type,
    "Transmission": transmission,
    "Car_Brand": brand,
}], columns=FEATURE_COLUMNS)

# ------------------- Prediction -------------------
if st.button("Predict Price"):
    raw_prediction, prediction = predict_with_logic(model, input_data)
    price_band = get_price_band(prediction)
    reasons = explain_prediction(year, kms_driven, fuel_type, seller_type, transmission, present_price, brand)

    st.markdown(
        f"""
        <div class="result-card">
            <h3 style="margin-top:0; color:#0f4c81;">Prediction Result</h3>
            <div style="font-size:1.7rem; font-weight:800; color:#145a8a;">₹{prediction:.2f} lakhs</div>
            <div style="margin-top:0.35rem; color:#4f6880;"><b>Model estimate:</b> ₹{raw_prediction:.2f} lakhs</div>
            <div style="margin-top:0.35rem;"><span class="badge">{price_band}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="logic-card">
            <strong>Prediction logic</strong>
            <ul>
                <li>The trained model first estimates the price from the input features.</li>
                <li>A small market-style adjustment is then added for age, mileage, fuel type, seller type, and transmission.</li>
                <li>The final value is shown as a more practical resale estimate.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Why this price?", expanded=True):
        for reason in reasons:
            st.write(f"• {reason}")

    st.success("Prediction completed successfully.")

    # Save to DB
    save_prediction(year, present_price, kms_driven, fuel_type, seller_type, transmission, brand, prediction)

# ------------------- History -------------------
st.subheader("Prediction History")
history = get_all_predictions()
if not history.empty:
    display_columns = [
        col for col in [
            "id", "year", "present_price", "kms_driven", "fuel_type",
            "seller_type", "transmission", "brand", "predicted_price", "created_at"
        ] if col in history.columns
    ]
    st.table(history[display_columns])
else:
    st.info("No predictions saved yet.")

# ------------------- Model Info -------------------
METRICS_PATH = Path(__file__).resolve().parents[1] / "models" / "model_metrics.json"

def load_metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

metrics = load_metrics()

if metrics:
    st.markdown(
        f"""
        <div class="model-card">
            <strong>Best Model:</strong> {metrics['best_model']}<br>
            <em>Validation R²:</em> {metrics['tuned_validation_metrics']['val_r2']}<br>
            <em>Validation MAE:</em> {metrics['tuned_validation_metrics']['val_mae']}<br>
            <em>Validation RMSE:</em> {metrics['tuned_validation_metrics']['val_rmse']}
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.warning("Model metrics not found. Please train the model first.")

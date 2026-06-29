import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

# Paths
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "used_car_price_prediction.csv"
MODEL_DIR = ROOT / "models"
METRICS_PATH = MODEL_DIR / "model_metrics.json"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BRAND_COLUMN = "Car_Brand"
FEATURE_COLUMNS = ["Year", "Present_Price", "Kms_Driven", "Fuel_Type", "Seller_Type", "Transmission", BRAND_COLUMN]
TARGET_COLUMN = "Price"

BRAND_BINS = [-1, 3, 5, 7, 10, 15, 20, 30, 45, 60, 1000]
BRAND_LABELS = [
    "Maruti", "Hyundai", "Honda", "Toyota", "Ford", "Mahindra", "Tata", "BMW", "Mercedes", "Audi"
]


def get_feature_columns():
    return FEATURE_COLUMNS.copy()


def add_brand_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if BRAND_COLUMN in df.columns:
        return df
    if "Car_Name" in df.columns:
        df[BRAND_COLUMN] = df["Car_Name"].str.split(" ").str[0].fillna("Unknown").astype(str)
        return df
    if "Brand" in df.columns:
        df[BRAND_COLUMN] = df["Brand"].astype(str)
        return df
    if "Present_Price" in df.columns:
        df[BRAND_COLUMN] = pd.cut(df["Present_Price"], bins=BRAND_BINS, labels=BRAND_LABELS)
        df[BRAND_COLUMN] = df[BRAND_COLUMN].fillna(BRAND_LABELS[0]).astype(str)
        return df
    raise ValueError("Unable to derive brand feature from dataset; add Car_Brand or Car_Name.")


def rmse(y_true, y_pred) -> float:
    return mean_squared_error(y_true, y_pred) ** 0.5

# Load dataset
print(f"Loading data from {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
df = add_brand_feature(df)

missing = [col for col in FEATURE_COLUMNS + [TARGET_COLUMN] if col not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing required columns: {missing}")

X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing
cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object", "string"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), cat_cols),
    ]
)

# Models
rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)),
])

xgb_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        objective="reg:squarederror",
        verbosity=0,
    )),
])

ensemble_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", VotingRegressor([
        ("rf", RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)),
        ("xgb", XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42, objective="reg:squarederror", verbosity=0)),
    ])),
])

stacking_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", StackingRegressor(
        estimators=[
            ("rf", RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)),
            ("xgb", XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42, objective="reg:squarederror", verbosity=0)),
        ],
        final_estimator=RidgeCV(),
        passthrough=True,
        n_jobs=-1,
    )),
])

models = {
    "RandomForest": rf_pipeline,
    "XGBoost": xgb_pipeline,
    "VotingEnsemble": ensemble_pipeline,
    "StackingEnsemble": stacking_pipeline,
}

# Train and evaluate
results = {}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    results[name] = {
        "train_r2": round(r2_score(y_train, train_pred), 4),
        "val_r2": round(r2_score(y_val, val_pred), 4),
        "train_mae": round(mean_absolute_error(y_train, train_pred), 4),
        "val_mae": round(mean_absolute_error(y_val, val_pred), 4),
        "train_rmse": round(rmse(y_train, train_pred), 4),
        "val_rmse": round(rmse(y_val, val_pred), 4),
        "overfit_gap": round(r2_score(y_train, train_pred) - r2_score(y_val, val_pred), 4),
    }

# Pick best model
best_name = max(results, key=lambda n: results[n]["val_r2"])
print("\nModel comparison:")
for name, metrics in results.items():
    print(f"{name}: train R²={metrics['train_r2']}, val R²={metrics['val_r2']}, gap={metrics['overfit_gap']}")

print(f"Tuning best model: {best_name}")

# Hyperparameter tuning
if best_name == "StackingEnsemble":
    tuned_pipeline = stacking_pipeline
    param_grid = {
        "model__passthrough": [True, False],
        "model__cv": [2, 3],
        "model__final_estimator__alphas": [[0.1, 1.0, 10.0], [0.01, 0.1, 1.0]],
    }
elif best_name == "XGBoost":
    tuned_pipeline = xgb_pipeline
    param_grid = {
        "model__n_estimators": [100, 200],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [3, 5, 7],
    }
else:
    tuned_pipeline = rf_pipeline
    param_grid = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [4, 6, 8, 10],
        "model__min_samples_leaf": [1, 2, 4],
    }

search = GridSearchCV(estimator=tuned_pipeline, param_grid=param_grid, scoring="r2", cv=3, n_jobs=-1)
search.fit(X_train, y_train)

best_model = search.best_estimator_
val_pred = best_model.predict(X_val)

# Final metrics
val_r2 = r2_score(y_val, val_pred)
val_mae = mean_absolute_error(y_val, val_pred)
val_rmse = rmse(y_val, val_pred)

print(f"Best parameters: {search.best_params_}")
print(f"Best CV score: {round(search.best_score_, 4)}")
print(f"Tuned validation R²: {round(val_r2, 4)}")
print(f"Tuned validation MAE: {round(val_mae, 4)}")
print(f"Tuned validation RMSE: {round(val_rmse, 4)}")

# Save model and metrics
print("Saving tuned model")
joblib.dump(best_model, MODEL_DIR / "ensemble_model.pkl")

with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump({
        "best_model": best_name,
        "best_params": search.best_params_,
        "cv_best_score": round(search.best_score_, 4),
        "tuned_validation_metrics": {
            "val_r2": round(val_r2, 4),
            "val_mae": round(val_mae, 4),
            "val_rmse": round(val_rmse, 4),
        },
        "baseline_metrics": results,
    }, f, indent=2)

print(f"Saved model to {MODEL_DIR / 'ensemble_model.pkl'}")
print(f"Saved evaluation metrics to {METRICS_PATH}")

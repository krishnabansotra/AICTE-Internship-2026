import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "used_car_price_prediction.csv"
MODEL_DIR = ROOT / "models"
METRICS_PATH = MODEL_DIR / "model_metrics.json"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BRAND_NAMES = [
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
]
BRAND_BINS = [-1, 3, 5, 7, 10, 15, 20, 30, 45, 60, 1000]


def add_brand_feature(df: pd.DataFrame) -> pd.DataFrame:
    if "Brand" in df.columns:
        return df
    df = df.copy()
    df["Brand"] = pd.cut(df["Present_Price"], bins=BRAND_BINS, labels=BRAND_NAMES)
    df["Brand"] = df["Brand"].fillna(BRAND_NAMES[0]).astype(str)
    return df


print(f"Loading data from {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
df = add_brand_feature(df)

X = df.drop("Price", axis=1)
y = df["Price"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), num_cols),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            cat_cols,
        ),
    ]
)

rf_pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42),
        ),
    ]
)

xgb_pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        (
            "model",
            XGBRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                objective="reg:squarederror",
                verbosity=0,
            ),
        ),
    ]
)

ensemble_pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        (
            "model",
            VotingRegressor(
                [
                    (
                        "rf",
                        RandomForestRegressor(
                            n_estimators=200, max_depth=8, random_state=42
                        ),
                    ),
                    (
                        "xgb",
                        XGBRegressor(
                            n_estimators=200,
                            learning_rate=0.1,
                            max_depth=5,
                            random_state=42,
                            objective="reg:squarederror",
                            verbosity=0,
                        ),
                    ),
                ]
            ),
        ),
    ]
)

stacking_pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        (
            "model",
            StackingRegressor(
                estimators=[
                    (
                        "rf",
                        RandomForestRegressor(
                            n_estimators=200, max_depth=8, random_state=42
                        ),
                    ),
                    (
                        "xgb",
                        XGBRegressor(
                            n_estimators=200,
                            learning_rate=0.1,
                            max_depth=5,
                            random_state=42,
                            objective="reg:squarederror",
                            verbosity=0,
                        ),
                    ),
                ],
                final_estimator=RidgeCV(),
                passthrough=True,
                n_jobs=-1,
            ),
        ),
    ]
)

models = {
    "RandomForest": rf_pipeline,
    "XGBoost": xgb_pipeline,
    "VotingEnsemble": ensemble_pipeline,
    "StackingEnsemble": stacking_pipeline,
}

results = {}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    train_r2 = r2_score(y_train, train_pred)
    val_r2 = r2_score(y_val, val_pred)
    results[name] = {
        "train_r2": round(train_r2, 4),
        "val_r2": round(val_r2, 4),
        "train_mae": round(mean_absolute_error(y_train, train_pred), 4),
        "val_mae": round(mean_absolute_error(y_val, val_pred), 4),
        "train_rmse": round(mean_squared_error(y_train, train_pred, squared=False), 4),
        "val_rmse": round(mean_squared_error(y_val, val_pred, squared=False), 4),
        "overfit_gap": round(train_r2 - val_r2, 4),
    }

best_name = max(results, key=lambda n: results[n]["val_r2"])
best_model = models[best_name]

print("\nModel comparison:")
for name, metrics in results.items():
    print(f"{name}: train R²={metrics['train_r2']}, val R²={metrics['val_r2']}, gap={metrics['overfit_gap']}")

print(f"Saving best model: {best_name}")
joblib.dump(best_model, MODEL_DIR / "ensemble_model.pkl")

with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump({"best_model": best_name, "metrics": results}, f, indent=2)

print(f"Saved model to {MODEL_DIR / 'ensemble_model.pkl'}")
print(f"Saved evaluation metrics to {METRICS_PATH}")



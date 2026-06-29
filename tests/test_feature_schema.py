import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import train


def test_feature_schema_matches_prediction_inputs():
    expected = [
        "Year",
        "Present_Price",
        "Kms_Driven",
        "Fuel_Type",
        "Seller_Type",
        "Transmission",
        "Car_Brand",
    ]
    assert train.get_feature_columns() == expected

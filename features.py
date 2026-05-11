import pandas as pd
import numpy as np

# These must match exactly what generate_data.py produces
ROUTES = [
    "Route 33 (CBD-Umoja)",
    "Route 58 (CBD-Westlands)",
    "Route 9 (CBD-Karen)",
    "Route 111 (CBD-Rongai)",
    "Route 46 (CBD-Kikuyu)",
]

WEATHER_ORDER = {
    "sunny":      0,
    "cloudy":     1,
    "light_rain": 2,
    "heavy_rain": 3,
}

FEATURE_COLUMNS = [
    "weather_encoded",
    "is_rush_hour",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "route_Route 33 (CBD-Umoja)",
    "route_Route 46 (CBD-Kikuyu)",
    "route_Route 58 (CBD-Westlands)",
    "route_Route 9 (CBD-Karen)",
    "route_Route 111 (CBD-Rongai)",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw matatu trip columns into model-ready features.
    Called by both train.py and api.py — never duplicate this logic.
    """
    df = df.copy()

    # 1. Fill missing values before any transformation
    df["weather"]     = df["weather"].fillna("sunny")
    df["hour"]        = df["hour"].fillna(df["hour"].median() if len(df) > 1 else 12)
    df["day_of_week"] = df["day_of_week"].fillna("Monday")

    # 2. Ordinal encode weather
    df["weather_encoded"] = df["weather"].map(WEATHER_ORDER).fillna(0).astype(int)

    # 3. Rush hour flag (7-9am and 5-7pm)
    df["hour"] = df["hour"].astype(float)
    df["is_rush_hour"] = df["hour"].apply(
        lambda h: 1 if (7 <= h <= 9) or (17 <= h <= 19) else 0
    )

    # 4. Weekend flag
    df["is_weekend"] = df["day_of_week"].apply(
        lambda d: 1 if d in ["Saturday", "Sunday"] else 0
    )

    # 5. Cyclical hour encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # 6. One-hot encode route
    df = pd.get_dummies(df, columns=["route"], prefix="route")

    # 7. Ensure all expected route columns exist
    # (if a route is missing from this batch, add it as zeros)
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    return df[FEATURE_COLUMNS]


def prepare_single_prediction(route: str, weather: str,hour: int, day_of_week: str) -> pd.DataFrame:
    """
    Build a single-row DataFrame for API prediction.
    Wraps engineer_features so the API never touches raw feature logic.
    """
    row = pd.DataFrame([{
        "route":       route,
        "weather":     weather,
        "hour":        hour,
        "day_of_week": day_of_week,
    }])
    return engineer_features(row)

if __name__ == "__main__":
    test = pd.DataFrame([{
        "route":       "Route 111 (CBD-Rongai)",
        "weather":     "heavy_rain",
        "hour":        8,
        "day_of_week": "Monday",
    }])

    features = engineer_features(test)
    print("Feature columns:", features.columns.tolist())
    print("\nFeature values:")
    print(features.T)  # Transpose so it's easier to read
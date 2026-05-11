import joblib
import numpy as np
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

from features import prepare_single_prediction, ROUTES, WEATHER_ORDER

# ── Load model and explainer once at startup ───────────────────────────────────
MODEL_PATH     = Path("models/delay_model.pkl")
EXPLAINER_PATH = Path("models/shap_explainer.pkl")

if not MODEL_PATH.exists():
    raise FileNotFoundError("Model not found. Run train.py first.")
if not EXPLAINER_PATH.exists():
    raise FileNotFoundError("SHAP explainer not found. Run train.py first.")

model     = joblib.load(MODEL_PATH)
explainer = joblib.load(EXPLAINER_PATH)

print("Model loaded.")
print("SHAP explainer loaded.")

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Nairobi Matatu Delay Predictor",
    description="Predicts matatu arrival delay with optional SHAP explainability.",
    version="2.0.0",
)

# ── Input schema ───────────────────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    route:       str = Field(..., example="Route 111 (CBD-Rongai)")
    weather:     str = Field(..., example="heavy_rain")
    hour:        int = Field(..., ge=0, le=23, example=8)
    day_of_week: str = Field(..., example="Monday")

    @field_validator("route")
    @classmethod
    def route_must_be_valid(cls, v):
        if v not in ROUTES:
            raise ValueError(f"route must be one of: {ROUTES}")
        return v

    @field_validator("weather")
    @classmethod
    def weather_must_be_valid(cls, v):
        if v not in WEATHER_ORDER:
            raise ValueError(f"weather must be one of: {list(WEATHER_ORDER.keys())}")
        return v

    @field_validator("day_of_week")
    @classmethod
    def day_must_be_valid(cls, v):
        valid = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        if v not in valid:
            raise ValueError(f"day_of_week must be one of: {valid}")
        return v


# ── Output schemas ─────────────────────────────────────────────────────────────
class FeatureContribution(BaseModel):
    feature:      str
    contribution: float
    direction:    str    # "increases_delay" or "reduces_delay"


class PredictionResponse(BaseModel):
    predicted_delay_minutes: float
    confidence:              str
    plain_english_summary:   str
    top_factors:             Optional[List[FeatureContribution]] = None
    inputs_received:         dict


# ── Helpers ────────────────────────────────────────────────────────────────────
def get_confidence(prediction: float) -> str:
    if prediction < 10:
        return "high"
    elif prediction < 20:
        return "medium"
    else:
        return "low"


FEATURE_LABELS = {
    "weather_encoded":               "Weather conditions",
    "is_rush_hour":                  "Rush hour",
    "is_weekend":                    "Weekend",
    "hour_sin":                      "Time of day (cyclical)",
    "hour_cos":                      "Time of day (cyclical)",
    "route_Route 33 (CBD-Umoja)":    "Route: CBD-Umoja",
    "route_Route 46 (CBD-Kikuyu)":   "Route: CBD-Kikuyu",
    "route_Route 58 (CBD-Westlands)":"Route: CBD-Westlands",
    "route_Route 9 (CBD-Karen)":     "Route: CBD-Karen",
    "route_Route 111 (CBD-Rongai)":  "Route: CBD-Rongai",
}


def build_explanation(shap_values, feature_names) -> List[FeatureContribution]:
    """Convert raw SHAP values into sorted feature contributions.
    Merges the two cyclical hour columns into one combined entry."""
    contributions = {}

    for feat, val in zip(feature_names, shap_values):
        label = FEATURE_LABELS.get(feat, feat)
        # Merge hour_sin and hour_cos into one label
        if "Time of day" in label:
            contributions[label] = contributions.get(label, 0) + val
        else:
            contributions[label] = val

    result = []
    for label, value in contributions.items():
        result.append(FeatureContribution(
            feature=label,
            contribution=round(float(value), 2),
            direction="increases_delay" if value > 0 else "reduces_delay",
        ))

    # Sort by absolute contribution, largest first
    result.sort(key=lambda x: abs(x.contribution), reverse=True)
    return result


def build_plain_english(prediction: float,top_factors: List[FeatureContribution]) -> str:
    """Turn top SHAP factors into one sentence a commuter can understand."""
    # Only mention factors that moved the prediction by at least 1 minute
    meaningful = [f for f in top_factors if abs(f.contribution) >= 1.0][:3]

    if not meaningful:
        return (
            f"Predicted delay is {prediction} min. "
            "Conditions are close to average."
        )

    parts = []
    for f in meaningful:
        verb = "adding" if f.contribution > 0 else "saving"
        parts.append(f"{f.feature} ({verb} ~{abs(f.contribution):.0f} min)")

    return (
        f"Predicted delay of {prediction} min is mainly driven by: "
        + ", ".join(parts) + "."
    )


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status":  "ok",
        "message": "Matatu Delay Predictor API is running.",
        "tip":     "POST to /predict or /predict?explain=true for SHAP explanations.",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(
    request: PredictionRequest,
    explain: bool = Query(
        default=False,
        description="Set to true to include per-prediction SHAP feature contributions.",
    ),
):
    try:
        # Build features using the shared module — same logic as training
        features = prepare_single_prediction(
            route=request.route,
            weather=request.weather,
            hour=request.hour,
            day_of_week=request.day_of_week,
        )

        # Predict and floor at zero
        prediction = float(round(model.predict(features)[0], 1))
        prediction = max(0.0, prediction)

        # SHAP — only computed when explain=true
        top_factors = None
        if explain:
            shap_vals   = explainer.shap_values(features)[0]
            top_factors = build_explanation(shap_vals, features.columns.tolist())

        # Plain English summary uses SHAP factors if available,
        # falls back to a simple message if explain=false
        summary = build_plain_english(prediction, top_factors or [])

        return PredictionResponse(
            predicted_delay_minutes=prediction,
            confidence=get_confidence(prediction),
            plain_english_summary=summary,
            top_factors=top_factors,
            inputs_received=request.dict(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-info")
def model_info():
    return {
        "model_type":      type(model).__name__,
        "n_estimators":    model.n_estimators,
        "training_rmse":   4.76,
        "training_mae":    3.60,
        "training_r2":     0.8247,
        "features_used":   model.n_features_in_,
        "explainability":  "SHAP TreeExplainer (optional via ?explain=true)",
        "routes":          ROUTES,
        "weather_options": list(WEATHER_ORDER.keys()),
    }
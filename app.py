import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Config ─────────────────────────────────────────────────────────────────────
API_URL = "http://localhost:8000"

ROUTES = [
    "Route 33 (CBD-Umoja)",
    "Route 58 (CBD-Westlands)",
    "Route 9 (CBD-Karen)",
    "Route 111 (CBD-Rongai)",
    "Route 46 (CBD-Kikuyu)",
]

WEATHER_OPTIONS = ["sunny", "cloudy", "light_rain", "heavy_rain"]
WEATHER_LABELS  = {
    "sunny":      "Sunny",
    "cloudy":     "Cloudy",
    "light_rain": "Light rain",
    "heavy_rain": "Heavy rain",
}

DAYS = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

CONFIDENCE_COLOUR = {
    "high":   "#2ecc71",
    "medium": "#f39c12",
    "low":    "#e74c3c",
}

# ── Page setup ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nairobi Matatu Delay Predictor",
    page_icon="🚌",
    layout="centered",
)

st.title("🚌 Nairobi Matatu Delay Predictor")
st.caption("Powered by a Random Forest model trained on synthetic Nairobi trip data.")

# ── Sidebar: API health check ──────────────────────────────────────────────────
with st.sidebar:
    st.header("API Status")
    try:
        health = requests.get(f"{API_URL}/", timeout=3)
        if health.status_code == 200:
            st.success("API is running")
        else:
            st.error("API returned an error")
    except Exception:
        st.error("Cannot reach API. Is uvicorn running?")

    st.divider()
    st.header("Model Info")
    try:
        info = requests.get(f"{API_URL}/model-info", timeout=3).json()
        st.metric("Model", info["model_type"])
        st.metric("R² Score", info["training_r2"])
        st.metric("RMSE", f"{info['training_rmse']} min")
        st.metric("MAE", f"{info['training_mae']} min")
    except Exception:
        st.warning("Could not load model info.")


# ── Inputs ─────────────────────────────────────────────────────────────────────
st.subheader("Trip details")

col1, col2 = st.columns(2)

with col1:
    route = st.selectbox("Route", ROUTES)
    weather_key = st.selectbox(
        "Weather",
        WEATHER_OPTIONS,
        format_func=lambda x: WEATHER_LABELS[x],
    )

with col2:
    hour = st.slider("Hour of departure", min_value=5, max_value=22, value=8)
    day  = st.selectbox("Day of week", DAYS)

# Show a human-readable time label next to the slider
am_pm = "AM" if hour < 12 else "PM"
display_hour = hour if hour <= 12 else hour - 12
st.caption(f"Departure time: {display_hour}:00 {am_pm}")


# ── Predict button ─────────────────────────────────────────────────────────────
if st.button("Predict delay", type="primary", use_container_width=True):

    payload = {
        "route":       route,
        "weather":     weather_key,
        "hour":        hour,
        "day_of_week": day,
    }

    with st.spinner("Getting prediction..."):
        try:
            response = requests.post(
                f"{API_URL}/predict?explain=true",
                json=payload,
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the API. Make sure uvicorn is running on port 8000.")
            st.stop()
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()

    # ── Results ────────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Prediction")

    delay      = data["predicted_delay_minutes"]
    confidence = data["confidence"]
    summary    = data["plain_english_summary"]
    factors    = data["top_factors"]

    # Main metric
    conf_colour = CONFIDENCE_COLOUR[confidence]
    col_a, col_b = st.columns([1, 2])

    with col_a:
        st.metric(
            label="Predicted delay",
            value=f"{delay} min",
            delta=None,
        )
        st.markdown(
            f"Confidence: <span style='color:{conf_colour};font-weight:700'>"
            f"{confidence.upper()}</span>",
            unsafe_allow_html=True,
        )

    with col_b:
        st.info(summary)

    # ── SHAP waterfall chart ───────────────────────────────────────────────────
    if factors:
        st.subheader("Why this prediction?")
        st.caption(
            "Each bar shows how much a feature added to (+) or reduced (-) "
            "the predicted delay relative to the average trip."
        )

        # Build DataFrame from top_factors
        df_factors = pd.DataFrame([
            {
                "Feature":      f["feature"],
                "Contribution": f["contribution"],
                "Direction":    f["direction"],
            }
            for f in factors
        ])

        # Only show factors with abs contribution >= 0.1
        df_factors = df_factors[df_factors["Contribution"].abs() >= 0.1]
        df_factors = df_factors.sort_values("Contribution", ascending=True)

        fig, ax = plt.subplots(figsize=(8, max(3, len(df_factors) * 0.5)))
        colours = [
            "#e74c3c" if d == "increases_delay" else "#2ecc71"
            for d in df_factors["Direction"]
        ]
        bars = ax.barh(
            df_factors["Feature"],
            df_factors["Contribution"],
            color=colours,
            edgecolor="white",
            linewidth=0.5,
            height=0.6,
        )

        # Value labels on bars
        for bar, val in zip(bars, df_factors["Contribution"]):
            x_pos = bar.get_width() + (0.1 if val >= 0 else -0.1)
            ha    = "left" if val >= 0 else "right"
            ax.text(
                x_pos, bar.get_y() + bar.get_height() / 2,
                f"{val:+.1f} min",
                va="center", ha=ha, fontsize=9,
            )

        ax.axvline(0, color="#333", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Contribution to delay (minutes)")
        ax.set_title("Feature contributions (SHAP values)")

        red_patch   = mpatches.Patch(color="#e74c3c", label="Increases delay")
        green_patch = mpatches.Patch(color="#2ecc71", label="Reduces delay")
        ax.legend(handles=[red_patch, green_patch], fontsize=9)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Raw response expander ──────────────────────────────────────────────────
    with st.expander("Raw API response"):
        st.json(data)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption("MLlabwithMel — Matatu Delay Predictor | Built with FastAPI + Streamlit")
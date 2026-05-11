import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from features import engineer_features

# ── Load and clean ─────────────────────────────────────────────────────────────
df = pd.read_csv("data/matatu_trips.csv")

# Drop rows where the target is missing — can't train on unknowns
df = df.dropna(subset=["delay_minutes"])

print(f"Training on {len(df)} rows after dropping missing targets")

# ── Engineer features ──────────────────────────────────────────────────────────
# This calls our shared features.py — never duplicate this logic here
X = engineer_features(df)
y = df["delay_minutes"].values

print(f"Features: {X.columns.tolist()}")
print(f"Target range: {y.min():.1f} to {y.max():.1f} minutes")

# ── Split: 80% train, 20% test ─────────────────────────────────────────────────
# random_state=42 means the split is reproducible — same split every run
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain rows: {len(X_train)}, Test rows: {len(X_test)}")


# ── Helper: evaluate any model the same way ────────────────────────────────────
def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    mae   = mean_absolute_error(y_test, preds)
    r2    = r2_score(y_test, preds)
    print(f"\n{name}")
    print(f"  RMSE : {rmse:.2f} minutes")
    print(f"  MAE  : {mae:.2f} minutes")
    print(f"  R²   : {r2:.4f}")
    return preds, rmse, mae, r2


# ── Model 1: Linear Regression baseline ───────────────────────────────────────
print("\n" + "="*50)
print("BASELINE: Linear Regression")
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds, lr_rmse, lr_mae, lr_r2 = evaluate("Linear Regression", lr, X_test, y_test)


# ── Model 2: Random Forest ─────────────────────────────────────────────────────
print("\n" + "="*50)
print("MODEL: Random Forest")
rf = RandomForestRegressor(
    n_estimators=100,   # 100 decision trees
    max_depth=10,       # each tree goes at most 10 levels deep
    random_state=42,
    n_jobs=-1           # use all CPU cores
)
rf.fit(X_train, y_train)
rf_preds, rf_rmse, rf_mae, rf_r2 = evaluate("Random Forest", rf, X_test, y_test)


# ── Improvement summary ────────────────────────────────────────────────────────
print("\n" + "="*50)
print("IMPROVEMENT OVER BASELINE")
print(f"  RMSE improvement : {lr_rmse - rf_rmse:.2f} minutes")
print(f"  R² improvement   : {rf_r2 - lr_r2:.4f}")


# ── Feature importance ─────────────────────────────────────────────────────────
importance = pd.DataFrame({
    "feature":    X.columns,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False)

print("\nFeature importances (Random Forest):")
print(importance.to_string(index=False))

# ── Save SHAP explainer ────────────────────────────────────────────────────────
# TreeExplainer is optimised for Random Forest — much faster than generic SHAP
explainer = shap.TreeExplainer(rf)
joblib.dump(explainer, "models/shap_explainer.pkl")
print("SHAP explainer saved to models/shap_explainer.pkl")

# Quick sanity check: explain the first 5 test rows
shap_values = explainer.shap_values(X_test.iloc[:5])
print("\nSHAP values for first test row (feature contributions):")
for feat, val in zip(X.columns, shap_values[0]):
    direction = "+" if val >= 0 else ""
    print(f"  {feat:<40} {direction}{val:.2f} min")

# ── Plots ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Random Forest — Model Evaluation", fontsize=14, fontweight="bold")
fig.subplots_adjust(wspace=0.35)

# Plot 1: Predicted vs actual
ax1 = axes[0]
ax1.scatter(y_test, rf_preds, alpha=0.3, s=10, color="#2ecc71")
min_val = min(y_test.min(), rf_preds.min())
max_val = max(y_test.max(), rf_preds.max())
ax1.plot([min_val, max_val], [min_val, max_val],
         color="#e74c3c", linestyle="--", linewidth=1.5, label="Perfect prediction")
ax1.set_xlabel("Actual delay (minutes)")
ax1.set_ylabel("Predicted delay (minutes)")
ax1.set_title("Predicted vs actual")
ax1.legend(fontsize=9)

# Plot 2: Residuals
residuals = y_test - rf_preds
ax2 = axes[1]
ax2.hist(residuals, bins=40, color="#3498db", edgecolor="white", linewidth=0.5)
ax2.axvline(0, color="#e74c3c", linestyle="--", linewidth=1.5)
ax2.set_xlabel("Residual (actual - predicted)")
ax2.set_ylabel("Count")
ax2.set_title("Residual distribution")

# Plot 3: Feature importance
ax3 = axes[2]
ax3.barh(importance["feature"], importance["importance"],
         color="#9b59b6", edgecolor="white", linewidth=0.5)
ax3.set_xlabel("Importance score")
ax3.set_title("Feature importances")
ax3.invert_yaxis()
plt.tight_layout()  # leave space for suptitle
plt.savefig("data/model_evaluation.png", dpi=150, bbox_inches="tight")
plt.show()



# ── Save the model ─────────────────────────────────────────────────────────────
Path("models").mkdir(exist_ok=True)
joblib.dump(rf, "models/delay_model.pkl")
print("\nModel saved to models/delay_model.pkl")
print("Training complete.")
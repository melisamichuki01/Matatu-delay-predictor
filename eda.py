import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("data/matatu_trips.csv")

print("=== Basic Info ===")
print(f"Shape: {df.shape}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nDelay stats:\n{df['delay_minutes'].describe().round(2)}")

# Drop rows where target is missing (can't train on those)
df_clean = df.dropna(subset=["delay_minutes"]).copy()

# Fill missing categoricals with mode for EDA purposes only
df_clean["weather"]     = df_clean["weather"].fillna(df_clean["weather"].mode()[0])
df_clean["hour"]        = df_clean["hour"].fillna(df_clean["hour"].median())
df_clean["day_of_week"] = df_clean["day_of_week"].fillna("Unknown")

print(f"\nRows after dropping missing targets: {len(df_clean)}")

# ── Plot ───────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 12))
fig.suptitle("Nairobi Matatu Delay — Exploratory Analysis", fontsize=15, fontweight="bold", y=1.01)
#fig,axes = plt.subplots(2, 2, figsize=(16, 12))     
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

# 1. Delay distribution
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(df_clean["delay_minutes"], bins=40, color="#2ecc71", edgecolor="white", linewidth=0.5)
ax1.axvline(df_clean["delay_minutes"].mean(),   color="#e74c3c", linestyle="--", linewidth=1.5, label=f"Mean: {df_clean['delay_minutes'].mean():.1f} min")
ax1.axvline(df_clean["delay_minutes"].median(), color="#f39c12", linestyle="--", linewidth=1.5, label=f"Median: {df_clean['delay_minutes'].median():.1f} min")
ax1.set_title("Delay distribution")
ax1.set_xlabel("Delay (minutes)")
ax1.set_ylabel("Count")
ax1.legend(fontsize=9)

# 2. Average delay by hour
ax2 = fig.add_subplot(gs[0, 1])
hourly = df_clean.groupby("hour")["delay_minutes"].mean().reset_index()
ax2.plot(hourly["hour"], hourly["delay_minutes"], color="#3498db", linewidth=2, marker="o", markersize=4)
ax2.axvspan(7, 9,   alpha=0.15, color="#e74c3c", label="Morning rush hours")
ax2.axvspan(17, 19, alpha=0.15, color="#e67e22", label="Evening rush hours")
ax2.set_title("Average delay by hour of day")
ax2.set_xlabel("Hour")
ax2.set_ylabel("Avg delay (minutes)")
ax2.set_xticks(range(5, 23))
ax2.legend(fontsize=9)

# 3. Delay by weather condition
ax3 = fig.add_subplot(gs[1, 0])
weather_order = ["sunny", "cloudy", "light_rain", "heavy_rain"]
weather_means = df_clean.groupby("weather")["delay_minutes"].mean().reindex(weather_order)
colors = ["#f1c40f", "#95a5a6", "#5dade2", "#1a5276"]
bars = ax3.bar(weather_means.index, weather_means.values, color=colors, edgecolor="white", linewidth=0.5)
ax3.set_title("Average delay by weather")
ax3.set_xlabel("Weather condition")
ax3.set_ylabel("Avg delay (minutes)")
for bar, val in zip(bars, weather_means.values):
    ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
             f"{val:.1f}", ha="center", va="bottom", fontsize=9)

# 4. Delay by route (sorted)
ax4 = fig.add_subplot(gs[1, 1])
route_means = df_clean.groupby("route")["delay_minutes"].mean().sort_values(ascending=True)
short_names = [r.split("(")[1].replace(")", "") for r in route_means.index]
ax4.barh(short_names, route_means.values, color="#9b59b6", edgecolor="white", linewidth=0.5)
ax4.set_title("Average delay by route")
ax4.set_xlabel("Avg delay (minutes)")
for i, val in enumerate(route_means.values):
    ax4.text(val + 0.2, i, f"{val:.1f}", va="center", fontsize=9)

plt.savefig("data/eda_plots.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved to data/eda_plots.png")
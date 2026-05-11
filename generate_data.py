import pandas as pd
import numpy as np
import random
from pathlib import Path

np.random.seed(42)
random.seed(42)

ROUTES = [
    "Route 33 (CBD-Umoja)",
    "Route 58 (CBD-Westlands)",
    "Route 9 (CBD-Karen)",
    "Route 111 (CBD-Rongai)",
    "Route 46 (CBD-Kikuyu)",
]

# Higher base = worse route (CBD congestion, distance)
ROUTE_BASE_DELAY = {
    "Route 33 (CBD-Umoja)":    8,
    "Route 58 (CBD-Westlands)": 12,
    "Route 9 (CBD-Karen)":      10,
    "Route 111 (CBD-Rongai)":   18,
    "Route 46 (CBD-Kikuyu)":    14,
}

WEATHER_MULTIPLIER = {
    "sunny":       1.0,
    "cloudy":      1.2,
    "light_rain":  1.6,
    "heavy_rain":  2.4,
}

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def is_rush_hour(hour):
    return (7 <= hour <= 9) or (17 <= hour <= 19)


def generate_delay(route, weather, hour, day):
    base = ROUTE_BASE_DELAY[route]
    weather_mult = WEATHER_MULTIPLIER[weather]
    rush_bonus = 10 if is_rush_hour(hour) else 0
    weekend_discount = -4 if day in ["Saturday", "Sunday"] else 0

    # Noise: real delays aren't perfectly predictable
    noise = np.random.normal(0, 4)

    delay = (base * weather_mult) + rush_bonus + weekend_discount + noise

    # Delays can't be negative
    return max(0, round(delay, 1))


def generate_matatu_data(n=8000):
    records = []

    for _ in range(n):
        route   = random.choice(ROUTES)
        weather = random.choice(list(WEATHER_MULTIPLIER.keys()))
        hour    = random.randint(5, 22)
        day     = random.choice(DAYS)

        delay_minutes = generate_delay(route, weather, hour, day)

        record = {
            "route":          route,
            "weather":        weather,
            "hour":           hour,
            "day_of_week":    day,
            "delay_minutes":  delay_minutes,
        }

        # Inject realistic messiness (~5% missing values per column)
        if random.random() < 0.05:
            record["weather"] = None
        if random.random() < 0.03:
            record["hour"] = None
        if random.random() < 0.04:
            record["delay_minutes"] = None

        records.append(record)

    return pd.DataFrame(records)


if __name__ == "__main__":
    Path("data").mkdir(exist_ok=True)

    df = generate_matatu_data(8000)
    df.to_csv("data/matatu_trips.csv", index=False)

    print(f"Dataset shape: {df.shape}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nDelay stats:\n{df['delay_minutes'].describe().round(2)}")
    print(f"\nSample rows:")
    print(df.head(10).to_string())
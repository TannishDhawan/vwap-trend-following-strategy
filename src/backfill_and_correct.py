import pandas as pd
import numpy as np

# Load HF data (raw, full history)
df_hf = pd.read_parquet("data/QQQ_hf_full_raw.parquet")

# Load the daily ratio series we already computed
daily_ratio = pd.read_csv("logs/hf_alpaca_daily_ratio.csv", index_col=0)
daily_ratio.index = pd.to_datetime(daily_ratio.index, utc=True).tz_convert("America/New_York")
daily_ratio = daily_ratio.iloc[:, 0]

# Fit a simple linear trend to the ratio over the overlap period
x = (daily_ratio.index - daily_ratio.index[0]).days.astype(float)
y = daily_ratio.values
slope, intercept = np.polyfit(x, y, 1)

print(f"Ratio trend: slope={slope:.8f}/day, intercept={intercept:.6f}")
print(f"Ratio at overlap start (2020-07-27): {intercept:.6f}")

# Extrapolate ratio back to 2018-01-01
target_date = pd.Timestamp("2018-01-01", tz="America/New_York")
days_before_overlap = (daily_ratio.index[0] - target_date).days
extrapolated_ratio_2018 = intercept - slope * days_before_overlap
print(f"Extrapolated ratio at 2018-01-01: {extrapolated_ratio_2018:.6f}")

# Build a full correction factor series for every day in the backfill window
backfill_start = pd.Timestamp("2018-01-01", tz="America/New_York")
backfill_end = pd.Timestamp("2020-07-26", tz="America/New_York")

all_days = pd.date_range(backfill_start, backfill_end, freq="D", tz="America/New_York")
days_from_overlap_start = (daily_ratio.index[0] - all_days).days.astype(float)
correction_series = pd.Series(
    intercept - slope * days_from_overlap_start, index=all_days
)

# Apply correction
df_backfill = df_hf.loc[backfill_start:backfill_end].copy()
df_backfill["date_only"] = df_backfill.index.normalize()
df_backfill["correction"] = df_backfill["date_only"].map(correction_series)
df_backfill["correction"] = df_backfill["correction"].ffill().bfill()

for col in ["open", "high", "low", "close"]:
    df_backfill[col] = df_backfill[col] / df_backfill["correction"]

df_backfill = df_backfill.drop(columns=["date_only", "correction"])

print(f"\nBackfill segment: {len(df_backfill)} rows")
print(df_backfill.head())
print(df_backfill.tail())

df_backfill.to_parquet("data/QQQ_backfill_corrected_2018_2020.parquet")
print("\nSaved corrected backfill segment.")

# ------------------ SPLICE WITH ALPACA DATA ------------------
df_alpaca = pd.read_parquet("data/QQQ_1min_paper.parquet")

df_full = pd.concat([df_backfill, df_alpaca])
df_full = df_full[~df_full.index.duplicated(keep="last")]
df_full = df_full.sort_index()

print(f"\nFull merged dataset: {len(df_full)} rows")
print(f"Date range: {df_full.index.min()} to {df_full.index.max()}")

df_full.to_parquet("data/QQQ_1min_FULL_2018_2023.parquet")
print("Saved data/QQQ_1min_FULL_2018_2023.parquet")
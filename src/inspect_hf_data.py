from datasets import load_dataset
import pandas as pd

print("Loading HF dataset...")
ds = load_dataset("Maxim37/timeseries-1m-QQQ-10y")
df_hf = ds["train"].to_pandas()

# Parse and localize timestamps
df_hf["Timestamp"] = pd.to_datetime(df_hf["Timestamp"])
df_hf = df_hf.set_index("Timestamp")
df_hf.index = df_hf.index.tz_localize("America/New_York")


df_hf = df_hf.rename(columns={
    "Open": "open", "High": "high", "Low": "low",
    "Close": "close", "Volume": "volume"
})

# RTH filter, same as Alpaca pipeline
df_hf = df_hf.between_time("09:30", "15:59")

print(f"HF data range: {df_hf.index.min()} to {df_hf.index.max()}")
print(f"HF rows after RTH filter: {len(df_hf)}")

# Load existing Alpaca QQQ data
df_alpaca = pd.read_parquet("data/QQQ_1min_paper.parquet")
print(f"Alpaca data range: {df_alpaca.index.min()} to {df_alpaca.index.max()}")

# Find overlap window
overlap_start = max(df_hf.index.min(), df_alpaca.index.min())
overlap_end = min(df_hf.index.max(), df_alpaca.index.max())
print(f"Overlap window: {overlap_start} to {overlap_end}")

hf_overlap = df_hf.loc[overlap_start:overlap_end, ["close", "volume"]]
alpaca_overlap = df_alpaca.loc[overlap_start:overlap_end, ["close", "volume"]]

merged = hf_overlap.join(alpaca_overlap, lsuffix="_hf", rsuffix="_alpaca", how="inner")
print(f"Matched timestamps in overlap: {len(merged)}")

merged["price_diff_pct"] = (
    (merged["close_hf"] - merged["close_alpaca"]).abs() / merged["close_alpaca"] * 100
)

print(f"\nMean price diff: {merged['price_diff_pct'].mean():.4f}%")
print(f"Median price diff: {merged['price_diff_pct'].median():.4f}%")
print(f"Max price diff: {merged['price_diff_pct'].max():.4f}%")
print(f"% of bars with >0.5% diff: {(merged['price_diff_pct'] > 0.5).mean()*100:.2f}%")

print("\nWorst 10 mismatches:")
print(merged.sort_values("price_diff_pct", ascending=False).head(10))

df_hf.to_parquet("data/QQQ_hf_full_raw.parquet")
print("\nSaved full HF dataset to data/QQQ_hf_full_raw.parquet for later use.")
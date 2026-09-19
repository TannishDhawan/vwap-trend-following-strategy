import pandas as pd
import matplotlib.pyplot as plt

df_hf = pd.read_parquet("data/QQQ_hf_full_raw.parquet")
df_alpaca = pd.read_parquet("data/QQQ_1min_paper.parquet")

overlap_start = df_alpaca.index.min()
overlap_end = df_alpaca.index.max()

hf_overlap = df_hf.loc[overlap_start:overlap_end, ["close"]]
alpaca_overlap = df_alpaca.loc[overlap_start:overlap_end, ["close"]]

merged = hf_overlap.join(alpaca_overlap, lsuffix="_hf", rsuffix="_alpaca", how="inner")
merged["ratio"] = merged["close_hf"] / merged["close_alpaca"]

daily_ratio = merged["ratio"].resample("D").mean().dropna()

print(daily_ratio.describe())
print("\nFirst 10 days:")
print(daily_ratio.head(10))
print("\nLast 10 days:")
print(daily_ratio.tail(10))

daily_ratio.to_csv("logs/hf_alpaca_daily_ratio.csv")

plt.figure(figsize=(12, 5))
daily_ratio.plot()
plt.title("HF/Alpaca Price Ratio Over Time (Overlap Period)")
plt.ylabel("Ratio (HF close / Alpaca close)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/hf_alpaca_ratio.png", dpi=150)
plt.show()
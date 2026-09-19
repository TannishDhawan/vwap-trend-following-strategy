import pandas as pd
import numpy as np

def analyze_trades(ticker, trades_csv, initial_capital=25000.0):
    print(f"\n{'='*60}")
    print(f"BREAKDOWN: {ticker}")
    print(f"{'='*60}")
    
    trades = pd.read_csv(trades_csv)
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True, errors="coerce").dt.tz_convert("America/New_York")
    trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True, errors="coerce").dt.tz_convert("America/New_York")
    
    n_bad = trades["entry_time"].isna().sum()
    if n_bad > 0:
        print(f"WARNING: {n_bad} rows failed to parse entry_time and will be dropped.")
        trades = trades.dropna(subset=["entry_time"])
    
    trades["year"] = trades["entry_time"].dt.year
    trades["entry_hour"] = trades["entry_time"].dt.hour
    trades["entry_minute"] = trades["entry_time"].dt.minute
    
    # ---------- YEAR-BY-YEAR BREAKDOWN ----------
    print("\n--- Performance by Year ---")
    yearly = trades.groupby("year").agg(
        num_trades=("pnl", "count"),
        total_pnl=("pnl", "sum"),
        avg_pnl=("pnl", "mean"),
        win_rate=("pnl", lambda x: (x > 0).mean() * 100),
        stop_out_rate=("stopped_out", "mean")
    )
    yearly["stop_out_rate"] = yearly["stop_out_rate"] * 100
    yearly["total_pnl"] = yearly["total_pnl"].round(2)
    yearly["avg_pnl"] = yearly["avg_pnl"].round(2)
    yearly["win_rate"] = yearly["win_rate"].round(1)
    yearly["stop_out_rate"] = yearly["stop_out_rate"].round(1)
    print(yearly)
    
    print("\nTotal PnL as % of initial capital, by year:")
    print((yearly["total_pnl"] / initial_capital * 100).round(2))
    
    # ---------- HOUR-OF-DAY BREAKDOWN ----------
    print("\n--- Performance by Entry Hour (ET) ---")
    hourly = trades.groupby("entry_hour").agg(
        num_trades=("pnl", "count"),
        total_pnl=("pnl", "sum"),
        avg_pnl=("pnl", "mean"),
        win_rate=("pnl", lambda x: (x > 0).mean() * 100)
    )
    hourly["total_pnl"] = hourly["total_pnl"].round(2)
    hourly["avg_pnl"] = hourly["avg_pnl"].round(2)
    hourly["win_rate"] = hourly["win_rate"].round(1)
    print(hourly)
    
    print(f"\n--- Overall Summary ---")
    print(f"Total trades logged: {len(trades)}")
    print(f"Total PnL: ${trades['pnl'].sum():,.2f}")
    print(f"Best single trade: ${trades['pnl'].max():,.2f}")
    print(f"Worst single trade: ${trades['pnl'].min():,.2f}")
    
    return trades, yearly, hourly

qqq_trades, qqq_yearly, qqq_hourly = analyze_trades("QQQ", "logs/qqq_trade_log.csv")
tqqq_trades, tqqq_yearly, tqqq_hourly = analyze_trades("TQQQ", "logs/tqqq_trade_log.csv")
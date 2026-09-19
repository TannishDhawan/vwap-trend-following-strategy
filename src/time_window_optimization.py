"""
Time-Window Filtering Experiment
---------------------------------
Tests whether restricting trading activity to the highest-edge portion of the
trading day (identified in the full-day backtest breakdown) improves risk-adjusted
performance relative to trading the entire session.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import time

INITIAL_CAPITAL = 25000.0
COMMISSION_PER_SHARE = 0.0

WINDOWS = {
    "Full Day (Baseline)": (time(9, 30), time(16, 0)),
    "9:30-10:30": (time(9, 30), time(10, 30)),
    "9:30-11:00": (time(9, 30), time(11, 0)),
    "9:30-12:00": (time(9, 30), time(12, 0)),
}


def run_windowed_backtest(df, window_start, window_end, initial_capital=INITIAL_CAPITAL,
                           commission_per_share=COMMISSION_PER_SHARE):
    df = df.copy()
    df['tp'] = (df['high'] + df['low'] + df['close']) / 3.0

    capital = initial_capital
    daily_equity, daily_dates, trade_log = [], [], []

    for date, day_df in df.groupby(df.index.date):
        if len(day_df) < 10:
            continue

        day_df = day_df.copy()
        day_df['cum_v'] = day_df['volume'].cumsum()
        day_df['cum_tpv'] = (day_df['tp'] * day_df['volume']).cumsum()
        day_df['vwap'] = day_df['cum_tpv'] / day_df['cum_v']

        cash = capital
        position_shares = 0
        entry_price = None
        entry_time = None
        first_bar = True
        window_closed_today = False

        for row in day_df.itertuples():
            idx = row.Index
            close = float(row.close)
            vwap = float(row.vwap)
            bar_time = idx.time()

            in_window = window_start <= bar_time <= window_end

            if first_bar:
                first_bar = False
                continue

            if not in_window and position_shares != 0 and not window_closed_today:
                abs_shares = abs(position_shares)
                current_sign = np.sign(position_shares)
                comm = abs_shares * commission_per_share
                pnl = (close - entry_price) * abs_shares * current_sign - 2 * comm
                cash += abs_shares * close * current_sign - comm
                trade_log.append({
                    "date": date, "entry_time": entry_time, "entry_price": entry_price,
                    "exit_time": idx, "exit_price": close, "shares": abs_shares,
                    "pnl": pnl, "reason": "window_close"
                })
                position_shares = 0
                window_closed_today = True

            if not in_window:
                continue

            desired_sign = 1 if close > vwap else -1
            current_sign = np.sign(position_shares)

            if position_shares == 0:
                shares = int(cash / close)
                if shares >= 1:
                    position_shares = shares * desired_sign
                    comm = shares * commission_per_share
                    cash -= shares * close * desired_sign + comm
                    entry_price = close
                    entry_time = idx

            elif current_sign != desired_sign:
                abs_shares = abs(position_shares)
                comm = abs_shares * commission_per_share
                pnl = (close - entry_price) * abs_shares * current_sign - 2 * comm
                cash += abs_shares * close * current_sign - comm
                trade_log.append({
                    "date": date, "entry_time": entry_time, "entry_price": entry_price,
                    "exit_time": idx, "exit_price": close, "shares": abs_shares,
                    "pnl": pnl, "reason": "flip"
                })
                position_shares = 0

                shares = int(cash / close)
                if shares >= 1:
                    position_shares = shares * desired_sign
                    comm2 = shares * commission_per_share
                    cash -= shares * close * desired_sign + comm2
                    entry_price = close
                    entry_time = idx

        if position_shares != 0:
            abs_shares = abs(position_shares)
            last_close = float(day_df['close'].iloc[-1])
            current_sign = np.sign(position_shares)
            comm = abs_shares * commission_per_share
            pnl = (last_close - entry_price) * abs_shares * current_sign - 2 * comm
            cash += abs_shares * last_close * current_sign - comm
            trade_log.append({
                "date": date, "entry_time": entry_time, "entry_price": entry_price,
                "exit_time": day_df.index[-1], "exit_price": last_close,
                "shares": abs_shares, "pnl": pnl, "reason": "eod_flatten"
            })

        capital = cash
        daily_equity.append(capital)
        daily_dates.append(date)

    equity = pd.Series(daily_equity, index=pd.to_datetime(daily_dates), name="Equity")
    trades_df = pd.DataFrame(trade_log)
    return equity, trades_df


def compute_metrics(equity, trades_df, initial_capital=INITIAL_CAPITAL):
    returns = equity.pct_change().dropna()
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else np.nan
    rolling_max = equity.cummax()
    max_dd = ((equity - rolling_max) / rolling_max).min()
    total_return = (equity.iloc[-1] / initial_capital - 1) * 100
    win_rate = (trades_df['pnl'] > 0).mean() * 100 if len(trades_df) > 0 else 0
    avg_trades_per_day = len(trades_df) / len(equity) if len(equity) > 0 else 0

    return {
        "Final Equity": equity.iloc[-1],
        "Total Return (%)": total_return,
        "Sharpe": sharpe,
        "Max Drawdown (%)": max_dd * 100,
        "Total Trades": len(trades_df),
        "Avg Trades/Day": avg_trades_per_day,
        "Win Rate (%)": win_rate,
    }


def main():
    print("Loading QQQ full 2018-2023 dataset...")
    df = pd.read_parquet("data/QQQ_1min_FULL_2018_2023.parquet")

    results = {}
    equity_curves = {}

    for label, window in WINDOWS.items():
        w_start, w_end = window
        print(f"\nRunning: {label} ({w_start} - {w_end})...")
        equity, trades = run_windowed_backtest(df, w_start, w_end)
        metrics = compute_metrics(equity, trades)
        results[label] = metrics
        equity_curves[label] = equity

    results_df = pd.DataFrame(results).T
    results_df = results_df.round(2)
    print("\n" + "=" * 80)
    print("SUMMARY: QQQ Strategy Performance by Trading Window (2018-2023)")
    print("=" * 80)
    print(results_df.to_string())

    results_df.to_csv("results/time_window_comparison.csv")
    print("\nSaved: results/time_window_comparison.csv")

    plt.figure(figsize=(12, 7))
    for label, equity in equity_curves.items():
        equity.plot(label=label, linewidth=1.8)
    plt.yscale("log")
    plt.title("QQQ VWAP Strategy: Equity Curve by Trading Window Restriction")
    plt.ylabel("Account Value ($, log scale)")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/time_window_equity_curves.png", dpi=150)
    print("Saved: results/time_window_equity_curves.png")

    plt.figure(figsize=(9, 5))
    sharpes = results_df["Sharpe"].sort_values(ascending=False)
    colors = ["#2b8a3e" if s == sharpes.max() else "#4a90d9" for s in sharpes]
    plt.bar(sharpes.index, sharpes.values, color=colors)
    plt.title("Annualized Sharpe Ratio by Trading Window")
    plt.ylabel("Sharpe Ratio")
    plt.xticks(rotation=20, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/time_window_sharpe_comparison.png", dpi=150)
    print("Saved: results/time_window_sharpe_comparison.png")

    best_window = sharpes.idxmax()
    baseline_sharpe = results_df.loc["Full Day (Baseline)", "Sharpe"]
    best_sharpe = sharpes.max()
    improvement = ((best_sharpe - baseline_sharpe) / abs(baseline_sharpe)) * 100

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print(f"Best-performing window: {best_window} (Sharpe = {best_sharpe:.2f})")
    print(f"Baseline (full day) Sharpe: {baseline_sharpe:.2f}")
    print(f"Improvement: {improvement:+.1f}%")


if __name__ == "__main__":
    main()
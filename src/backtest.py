import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_backtest(ticker, initial_capital=25000.0, commission_per_share=0.0):
    print(f"\n--- Strategy Replication: {ticker} ---")
    
    # 1. Load data
    if ticker == "QQQ":
        df = pd.read_parquet("data/QQQ_1min_FULL_2018_2023.parquet")
    else:
        df = pd.read_parquet(f"data/{ticker}_1min_paper.parquet")
    
    print(f"Data range: {df.index.min()} to {df.index.max()}")
    print(f"Total rows: {len(df)}")
    
    df['tp'] = (df['high'] + df['low'] + df['close']) / 3.0
    
    capital = initial_capital
    daily_equity = []
    daily_dates = []
    trade_log = []
    
    day_groups = df.groupby(df.index.date)
    
    for date, day_df in day_groups:
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
        
        for row in day_df.itertuples():
            idx = row.Index
            close = float(row.close)
            vwap = float(row.vwap)
            
            if first_bar:
                first_bar = False
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
                exit_price = close
                pnl = (exit_price - entry_price) * abs_shares * current_sign - 2 * comm
                cash += abs_shares * exit_price * current_sign - comm
                
                trade_log.append({
                    "date": date,
                    "side": "LONG" if current_sign == 1 else "SHORT",
                    "entry_time": entry_time,
                    "entry_price": entry_price,
                    "exit_time": idx,
                    "exit_price": exit_price,
                    "shares": abs_shares,
                    "pnl": pnl,
                    "stopped_out": True,
                    "eod_flatten": False
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
                "date": date,
                "side": "LONG" if current_sign == 1 else "SHORT",
                "entry_time": entry_time,
                "entry_price": entry_price,
                "exit_time": day_df.index[-1],
                "exit_price": last_close,
                "shares": abs_shares,
                "pnl": pnl,
                "stopped_out": False,
                "eod_flatten": True
            })
            position_shares = 0
        
        capital = cash
        daily_equity.append(capital)
        daily_dates.append(date)
    
    equity = pd.Series(daily_equity, index=pd.to_datetime(daily_dates), name="Equity")
    returns = equity.pct_change().dropna()
    
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else np.nan
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    max_dd = drawdown.min()
    total_return = (equity.iloc[-1] / initial_capital - 1) * 100
    
    trades_df = pd.DataFrame(trade_log)
    win_rate = (trades_df['pnl'] > 0).mean() * 100 if len(trades_df) > 0 else 0
    avg_pnl = trades_df['pnl'].mean() if len(trades_df) > 0 else 0
    
    print(f"\nFinal Wealth: ${equity.iloc[-1]:,.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Annualized Sharpe: {sharpe:.2f}")
    print(f"Max Drawdown: {max_dd*100:.2f}%")
    print(f"Total Trading Days: {len(trades_df)}")
    print(f"Stop-Out Rate: N/A (single-trade-per-day mode)")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Avg PnL per Trade: ${avg_pnl:.2f}")
    
    return equity, trades_df

# --- EXECUTION ---
qqq_equity, qqq_trades = run_backtest("QQQ")
tqqq_equity, tqqq_trades = run_backtest("TQQQ")

qqq_trades.to_csv("logs/qqq_trade_log.csv", index=False)
tqqq_trades.to_csv("logs/tqqq_trade_log.csv", index=False)

plt.figure(figsize=(12, 7))
qqq_equity.plot(label='QQQ Strategy')
tqqq_equity.plot(label='TQQQ Strategy')
plt.yscale('log')
plt.title("VWAP Trend-Following Replication (Log Scale)")
plt.xlabel("Date")
plt.ylabel("Account Value ($, log scale)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/equity_curves.png", dpi=150)
plt.show()
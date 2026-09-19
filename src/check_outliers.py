import pandas as pd

def check_outlier_trades(trades_csv, ticker):
    trades = pd.read_csv(trades_csv, parse_dates=["entry_time", "exit_time"])
    trades["pnl_pct"] = trades["pnl"] / (trades["entry_price"] * trades["shares"]) * 100
    
    print(f"\n--- {ticker} Outlier Check ---")
    print(f"Worst 10 trades by PnL:")
    print(trades.nsmallest(10, "pnl")[["date", "side", "entry_price", "exit_price", "pnl", "pnl_pct"]])
    
    print(f"\nTrades with >5% single-trade loss:")
    big_losers = trades[trades["pnl_pct"] < -5]
    print(f"Count: {len(big_losers)}")
    print(big_losers[["date", "entry_price", "exit_price", "pnl_pct"]].head(20))

check_outlier_trades("logs/qqq_trade_log.csv", "QQQ")
check_outlier_trades("logs/tqqq_trade_log.csv", "TQQQ")
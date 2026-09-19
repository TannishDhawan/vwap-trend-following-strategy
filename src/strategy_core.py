import pandas as pd
import numpy as np


def run_strategy(df, position_fraction=1.0, confirmation_bars=1, vwap_threshold_pct=0.0,
                  initial_capital=25000.0, commission_per_share=0.0):
    """
    Parameterized VWAP flip strategy.

    position_fraction   : fraction of available cash deployed per trade (0-1]
    confirmation_bars    : consecutive bars a signal must persist beyond the VWAP
                            threshold before a flip executes (stop-loss "patience")
    vwap_threshold_pct   : % distance from VWAP required for a bar to count as a
                            directional signal (deadband / execution-timing filter)
    """
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
        confirm_counter = 0
        confirm_side = 0

        for row in day_df.itertuples():
            idx = row.Index
            close = float(row.close)
            vwap = float(row.vwap)

            if first_bar:
                first_bar = False
                continue

            upper = vwap * (1 + vwap_threshold_pct / 100.0)
            lower = vwap * (1 - vwap_threshold_pct / 100.0)

            if close > upper:
                raw_signal = 1
            elif close < lower:
                raw_signal = -1
            else:
                raw_signal = 0

            current_sign = np.sign(position_shares)

            if raw_signal != 0 and raw_signal != current_sign:
                if raw_signal == confirm_side:
                    confirm_counter += 1
                else:
                    confirm_side = raw_signal
                    confirm_counter = 1
            else:
                confirm_side = 0
                confirm_counter = 0

            desired_sign = current_sign
            if confirm_counter >= confirmation_bars and confirm_side != 0:
                desired_sign = confirm_side

            if position_shares == 0 and desired_sign != 0:
                capital_to_use = cash * position_fraction
                shares = int(capital_to_use / close)
                if shares >= 1:
                    position_shares = shares * desired_sign
                    comm = shares * commission_per_share
                    cash -= shares * close * desired_sign + comm
                    entry_price = close
                    entry_time = idx
                    confirm_counter = 0
                    confirm_side = 0

            elif desired_sign != 0 and desired_sign != current_sign and position_shares != 0:
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
                confirm_counter = 0
                confirm_side = 0

                capital_to_use = cash * position_fraction
                shares = int(capital_to_use / close)
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


def compute_sharpe(equity):
    returns = equity.pct_change().dropna()
    if len(returns) < 2 or returns.std() == 0:
        return -10.0
    return (returns.mean() / returns.std()) * np.sqrt(252)


def compute_robust_score(equity, dd_penalty=0.5, min_points_per_year=20):
    """
    Robust training objective: penalizes year-to-year instability and large
    drawdowns, rather than rewarding a single blended Sharpe that can hide
    both problems.
    """
    returns = equity.pct_change().dropna()
    if len(returns) < min_points_per_year:
        return -10.0

    yearly_sharpes = []
    for yr, group in returns.groupby(returns.index.year):
        if len(group) < min_points_per_year or group.std() == 0:
            continue
        s = (group.mean() / group.std()) * np.sqrt(252)
        yearly_sharpes.append(s)

    if len(yearly_sharpes) == 0:
        return -10.0

    mean_s = np.mean(yearly_sharpes)
    std_s = np.std(yearly_sharpes)

    rolling_max = equity.cummax()
    max_dd = abs(((equity - rolling_max) / rolling_max).min())

    return mean_s - std_s - dd_penalty * max_dd


def compute_metrics(equity, trades_df, initial_capital=25000.0):
    if len(equity) == 0:
        return {"Sharpe": -10.0, "Total Return (%)": -100.0,
                "Max Drawdown (%)": -100.0, "Total Trades": 0, "Final Equity": initial_capital}
    sharpe = compute_sharpe(equity)
    rolling_max = equity.cummax()
    max_dd = ((equity - rolling_max) / rolling_max).min()
    total_return = (equity.iloc[-1] / initial_capital - 1) * 100
    return {
        "Sharpe": sharpe,
        "Total Return (%)": total_return,
        "Max Drawdown (%)": max_dd * 100,
        "Total Trades": len(trades_df),
        "Final Equity": equity.iloc[-1],
    }
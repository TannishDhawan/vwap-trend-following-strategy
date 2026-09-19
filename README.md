# VWAP Trend-Following Strategy: Replication & Extension

An independent replication and extension of the VWAP-based intraday trend-following
strategy described in Zarattini & Aziz (2023), *"Volume Weighted Average Price (VWAP):
The Holy Grail for Day Trading Systems"* (SSRN 4631351), tested on QQQ and TQQQ.

## Overview

This project implements an intraday, always-in-the-market trend-following strategy that
goes long when price is above the session VWAP and short when price is below it, flipping
positions on every VWAP cross and flattening at the close. The goal was to (1) independently
reproduce the paper's published performance metrics as closely as possible given free data
constraints, (2) extend the analysis to test signal persistence and regime-dependence across
market conditions the original paper did not explicitly break out, and (3) jointly optimize
strategy parameters using Bayesian optimization under walk-forward validation, benchmarked
against grid search.

## Strategy Logic

- **Signal:** Long when 1-minute close > session VWAP; short when close < session VWAP
- **VWAP:** Cumulative volume-weighted typical price `(High + Low + Close) / 3`, reset daily
- **Execution:** Always in the market during regular trading hours (9:30 AM–4:00 PM ET);
  position flips immediately on every VWAP cross
- **Exit:** Forced flat at market close (no overnight exposure)
- **Position sizing:** Full account equity reinvested each trade (compounding), baseline;
  fractional sizing explored in the optimization phase
- **Costs:** $0 commission baseline (matches the paper's assumption); slippage sensitivity
  planned as a future extension

## Data

| Ticker | Source | Date Range | Notes |
|---|---|---|---|
| QQQ | Alpaca (IEX feed, 2020-07-27 to 2023-09-29) + Hugging Face community dataset (2018-01-02 to 2020-07-24) | 2018-01-02 to 2023-09-29 | See "Data Construction" below |
| TQQQ | Alpaca (IEX feed) | 2020-07-27 to 2023-09-29 | No free pre-2020 source found; shorter window than QQQ |

**Why two sources for QQQ:** Alpaca's free tier only provides IEX-sourced minute bars from
mid-2020 onward. To approximate the paper's full 2018–2023 window, a community-published
1-minute QQQ dataset (`Maxim37/timeseries-1m-QQQ-10y` on Hugging Face) was used to backfill
2018–mid-2020.

**Data quality correction:** The Hugging Face dataset was found to have a systematic ~1.4–1.6%
price offset relative to Alpaca's data in the overlapping period (2020-07-27 to 2023-09-29),
consistent with a static dividend-adjustment factor applied uniformly across the full history
rather than a proper time-varying adjustment. This was empirically corrected by fitting a
linear trend to the daily price ratio in the overlap window and applying the extrapolated
correction factor to the pre-2020 backfill segment. Full methodology in
`backfill_and_correct.py`.

**Known limitation:** Trading volume conventions differ between the two source periods
(Hugging Face's source vs. Alpaca's IEX feed, which represents only a subset of consolidated
market volume). This means VWAP calculated on the pre-2020 backfill segment is not computed
on a strictly consistent volume basis relative to the 2020+ segment. Price levels were
validated and corrected; volume was not.

## Results

### QQQ (2018-01-02 to 2023-09-29)

| Metric | This Replication | Published (Zarattini & Aziz, 2023) |
|---|---|---|
| Total Return | +276.2% | +671% |
| Annualized Sharpe | 1.34 | 2.1 |
| Max Drawdown | -23.6% | -9.4% |
| Total Trades | 23,145 (~16/day) | Not disclosed |

### TQQQ (2020-07-27 to 2023-09-29, shorter window)

| Metric | This Replication | Published (Zarattini & Aziz, 2023) |
|---|---|---|
| Total Return | +8.3% | +8,242% (full 2018–2023 window) |
| Annualized Sharpe | 0.32 | Not separately disclosed |
| Max Drawdown | -69.5% | Comparable to QQQ (~9–10%) |

**The TQQQ result materially diverges from the published claim** that the leveraged
extension maintains a similar drawdown profile to QQQ. In this replication, TQQQ's
max drawdown (-69.5%) is roughly 3x worse than QQQ's over a comparable window,
suggesting leverage amplifies the strategy's whipsaw losses more than its trend-following
gains outside of strongly trending years. See "Regime Dependence" below.

## Key Findings

### 1. Signal persistence (QQQ)
The QQQ strategy was profitable in **every calendar year from 2018–2023**, including the
2022 bear market and the volatile 2020 COVID period, indicating the underlying signal is
not a one-regime artifact.

| Year | Total PnL (% of capital) |
|---|---|
| 2018 | +48.8% |
| 2019 | +13.9% |
| 2020 | +106.5% |
| 2021 | +51.3% |
| 2022 | +34.8% |
| 2023 | +20.9% |

### 2. Edge concentration at the market open
Performance breakdown by entry hour shows the 9:30–10:00 AM window drives a disproportionate
share of total PnL and win rate relative to the rest of the trading day, consistent with
independent third-party replications of this strategy. Mid-day hours (10 AM–2 PM) contribute
comparatively little.

### 3. Regime dependence and leverage decay (TQQQ)
Unlike QQQ, TQQQ's year-by-year results are highly inconsistent — a single strongly trending
year (2021: +105%) accounts for nearly all of its cumulative gains, while three of four years
(2020, 2022, 2023) were net losers, including a -61.7% year in 2023 despite QQQ itself being
positive over the same period. This is consistent with volatility decay: 3x leverage
amplifies losses from VWAP whipsaws in choppy/declining markets more than it amplifies gains
from sustained trends.

### 4. Reproduction gap vs. published results
This replication falls short of the paper's published Sharpe (1.34 vs. 2.1) and total return
(+276% vs. +671%) for QQQ. Plausible contributing factors:
- **Data feed:** Alpaca's free tier uses IEX (a subset of consolidated volume), not the full
  SIP feed likely used in the original paper; this affects VWAP precision, especially in the
  most volatile/whipsaw-prone minutes
- **VWAP approximation:** Typical price `(H+L+C)/3` is a coarser proxy for true volume-weighted
  price than tick-level trade data
- **Underspecified original methodology:** The paper does not fully disclose position-sizing
  mechanics, exact tie-breaking rules, or whether any minimum-move filter is applied before
  flipping

### 5. Time-window filtering does not improve risk-adjusted returns
Despite the market open (9:30–10:00 AM) contributing a disproportionate share of per-trade
profitability (see Finding #2), restricting trading exclusively to early-session windows
(9:30–10:30, 9:30–11:00, 9:30–12:00) produced substantially *worse* Sharpe ratios (0.42–0.83)
than trading the full session (1.34). This indicates the strategy's edge, while unevenly
distributed by hour, depends on cumulative trade frequency and full-day compounding rather
than being isolable to a single high-edge window. See `time_window_optimization.py` and
`time_window_comparison.csv` for the full sensitivity analysis.

## Bayesian Optimization & Walk-Forward Validation

### Objective
Jointly tune three strategy parameters — position-sizing (`position_fraction`), stop-loss
patience (`confirmation_bars`), and an execution-timing deadband (`vwap_threshold_pct`) —
using Gaussian-process-based Bayesian optimization (`scikit-optimize`), validated under
expanding-window walk-forward testing (train on years 1..N, test out-of-sample on year N+1,
roll forward). Bayesian optimization was benchmarked against exhaustive grid search on
identical folds and an identical training objective.

### Training objective
Rather than optimizing raw Sharpe ratio on the full training window — which can reward
parameter settings that happen to look good on one blended number while hiding severe
year-to-year inconsistency or drawdown risk — a custom **robust score** was used:
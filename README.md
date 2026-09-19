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


```
robust_score = mean(yearly Sharpes) − std(yearly Sharpes) − 0.5 × |max drawdown|
```

This penalizes both instability across years and large drawdowns, directly targeting the
overfitting risk inherent in tuning on a small number of noisy historical years.

### Iteration 1: Narrow search space, single-seed Bayesian
**Result:** Bayesian underperformed both grid search and a fixed-parameter baseline
(OOS Sharpe 1.12 vs. grid 1.18 vs. baseline 1.22). Diagnostics showed both optimizers
repeatedly selecting parameter values at the edges of their allowed ranges
(`confirmation_bars` hitting its upper bound of 5 in 3 of 4 folds; `position_fraction`
hitting its lower bound of 0.5 in 3 of 4 folds) — a signature that the true optimum likely
lay outside the tested range.

### Iteration 2: Widened search space, robust objective, still single-seed
**Result:** Widening the bounds (`position_fraction`: 0.1–1.0, `confirmation_bars`: 1–10,
`vwap_threshold_pct`: 0.0–0.5) and switching to the robust training objective did not fix
the underlying issue — Bayesian's OOS Sharpe actually dropped further (0.86 vs. grid 1.27
vs. baseline 1.22). Selected parameters varied dramatically across folds
(`position_fraction`: 0.12, 0.12, 0.66, 1.0), indicating the Gaussian-process surrogate was
being misled by noise in a small-sample (4–6 years of training data), non-smooth financial
objective — a well-documented limitation of Bayesian optimization when the evaluation
budget is low relative to the noise level of the objective function.

### Iteration 3: Multi-seed averaging
**Fix:** Ran `gp_minimize` with 3 independent random seeds per fold and averaged the
resulting parameters (median for the integer `confirmation_bars`, mean for continuous
parameters), a standard technique for reducing variance introduced by GP surrogate
initialization randomness.

**Result:** This resolved the instability. Bayesian optimization achieved the best
full-period out-of-sample Sharpe of the three methods tested:

| Method | Full-Period OOS Sharpe | Avg. Search Time/Fold | Evaluations/Fold |
|---|---|---|---|
| Fixed baseline (no tuning) | 1.22 | — | — |
| Grid search | 1.27 | 73.0s | 27 |
| **Bayesian optimization (multi-seed)** | **1.32** | 125.7s | 45 (3 seeds × 15) |

Parameter stability across folds also improved substantially — `position_fraction`
selection tightened from a cross-fold standard deviation of 0.374 (single-seed) to 0.076
(multi-seed averaged), a meaningfully more consistent and generalizable parameter choice.

| Test Year | Baseline Sharpe | Grid Sharpe | Bayesian Sharpe (multi-seed) |
|---|---|---|---|
| 2020 | 2.55 | 1.32 | 1.25 |
| 2021 | 1.18 | 1.19 | **2.35** |
| 2022 | 0.52 | **1.44** | 1.27 |
| 2023 | 0.65 | **0.99** | 0.77 |

### Honest tradeoff: compute cost
Multi-seed Bayesian optimization required ~72% more wall-clock time per fold than grid
search (125.7s vs. 73.0s), since the stability gain came from running three independent
optimization passes rather than one. In this project's regime (a small, cheap 3-parameter
search space), grid search remains a fully viable and nearly-as-effective alternative. The
efficiency advantage typically cited for Bayesian optimization (fewer evaluations to reach
a comparable optimum) is most pronounced in **higher-dimensional parameter spaces** where
grid search's evaluation count grows exponentially with each added parameter — a scenario
not fully realized in this 3-parameter setup, but relevant for future extensions with more
tunable parameters.

### Key takeaway
This experiment demonstrates that Bayesian optimization's advantage over grid search is
not automatic — with a limited evaluation budget on a noisy financial objective, a single
optimization run can be *less* reliable than exhaustive search over a coarse grid. Reducing
that noise (via a multi-year-stability-aware objective and multi-seed averaging) was
necessary before Bayesian optimization's theoretical advantages materialized in practice.
This is a documented, reproducible instance of the broader overfitting risk inherent in
any hyperparameter tuning on limited historical financial data — precisely the kind of
regime-dependence and overfitting risk this project set out to investigate.

Full results and diagnostics: `results/walk_forward_summary_v3.csv`,
`results/walk_forward_comparison_v3.png`

## Repository Structure

```
vwap_strategy/
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── vwap_strategy.py            # Alpaca data download
│   ├── inspect_hf_data.py          # Hugging Face dataset loading + cross-source validation
│   ├── analyze_gap.py              # Diagnoses the HF/Alpaca price offset
│   ├── backfill_and_correct.py     # Corrects and splices pre-2020 QQQ data
│   ├── backtest.py                 # Core VWAP strategy backtest
│   ├── analyze_trades.py           # Year-by-year / hour-by-hour breakdown
│   ├── check_outliers.py           # Data quality / outlier trade diagnostics
│   ├── time_window_optimization.py # Time-window filtering sensitivity experiment
│   ├── strategy_core.py            # Parameterized VWAP strategy engine (reusable)
│   └── bayesian_walkforward.py     # Bayesian optimization + walk-forward validation
│
├── data/                            # Parquet files (git-ignored; regenerate via src/ scripts)
├── logs/                            # Trade logs and intermediate diagnostic CSVs
└── results/                         # Final plots and summary tables
    └── optimization_iterations/     # v1/v2 Bayesian optimization iterations
```

## Limitations

- Free-tier IEX data (not full SIP consolidated tape) may understate true intraday volume
  and introduce minor VWAP calculation noise relative to the original paper
- QQQ pre-2020 data required a third-party community dataset with an empirically-corrected
  systematic pricing offset; correction methodology is documented but not independently
  audited against a second reference source
- TQQQ history limited to 2020-07-27 onward due to lack of a free pre-2020 1-minute data source
- Backtest assumes zero commission and zero slippage (matching the paper's stated
  assumption), which is unrealistic for live trading, especially at ~16 trades/day
- No transaction cost or market impact modeling for the ~23,000 QQQ trades over the test period
- Bayesian optimization was tuned on a robust, drawdown-penalized score rather than raw
  Sharpe; results may differ if a different objective function were used
- Walk-forward validation used only 4 expanding-window folds due to limited historical data
  depth; more folds (e.g., via rolling fixed-length windows) would improve confidence in the
  optimization comparison

## Planned Extensions

- [x] Bayesian optimization (Gaussian-process surrogate) over position-sizing, stop-loss,
      and execution-timing parameters under walk-forward validation
- [ ] Commission and slippage sensitivity analysis
- [x] Comparison of Bayesian optimization vs. grid search for out-of-sample robustness
- [x] Restricting entries to the highest-edge time window (9:30–10:30 AM) as a filtered variant
      — tested; did not improve Sharpe (see Finding #5), retained as documented negative result
- [ ] Extend optimization to a higher-dimensional parameter space to better demonstrate
      Bayesian optimization's sample-efficiency advantage over grid search
- [ ] Apply the validated strategy and optimization framework to additional tickers to
      further test signal persistence

## Reference

Zarattini, C., & Aziz, A. (2023). *Volume Weighted Average Price (VWAP): The Holy Grail for
Day Trading Systems.* SSRN. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4631351
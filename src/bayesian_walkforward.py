"""
Bayesian Optimization + Walk-Forward Validation (v3 - multi-seed)
--------------------------------------------------------------------
Widened parameter search space, a robust drawdown/stability-penalized
training objective, and multi-seed averaging for Bayesian optimization
to reduce variance from GP surrogate initialization randomness on a
noisy financial objective. Benchmarked against grid search.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import time as timer

from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

from strategy_core import run_strategy, compute_sharpe, compute_robust_score

INITIAL_CAPITAL = 25000.0

SPACE = [
    Real(0.1, 1.0, name="position_fraction"),
    Integer(1, 10, name="confirmation_bars"),
    Real(0.0, 0.5, name="vwap_threshold_pct"),
]

GRID = {
    "position_fraction": [0.25, 0.5, 1.0],
    "confirmation_bars": [1, 5, 10],
    "vwap_threshold_pct": [0.0, 0.2, 0.5],
}


def load_data():
    return pd.read_parquet("data/QQQ_1min_FULL_2018_2023.parquet")


def make_folds(df):
    years = sorted(df.index.year.unique())
    folds = []
    for i in range(2, len(years)):
        train_years = years[:i]
        test_year = years[i]
        train_df = df[df.index.year.isin(train_years)]
        test_df = df[df.index.year == test_year]
        folds.append((train_years, test_year, train_df, test_df))
    return folds


def objective_factory(train_df):
    @use_named_args(SPACE)
    def objective(position_fraction, confirmation_bars, vwap_threshold_pct):
        equity, _ = run_strategy(
            train_df,
            position_fraction=position_fraction,
            confirmation_bars=confirmation_bars,
            vwap_threshold_pct=vwap_threshold_pct,
            initial_capital=INITIAL_CAPITAL,
        )
        return -compute_robust_score(equity)
    return objective


def run_bayesian_opt(train_df, n_calls=15, seeds=(42, 7, 123)):
    objective = objective_factory(train_df)

    seed_results = []
    for seed in seeds:
        result = gp_minimize(
            objective, SPACE, n_calls=n_calls, random_state=seed,
            acq_func="EI", n_initial_points=max(5, n_calls // 3),
        )
        seed_results.append({
            "position_fraction": result.x[0],
            "confirmation_bars": result.x[1],
            "vwap_threshold_pct": result.x[2],
            "score": -result.fun,
        })

    pf_vals = [r["position_fraction"] for r in seed_results]
    cb_vals = [r["confirmation_bars"] for r in seed_results]
    vt_vals = [r["vwap_threshold_pct"] for r in seed_results]
    score_vals = [r["score"] for r in seed_results]

    avg_params = {
        "position_fraction": float(np.mean(pf_vals)),
        "confirmation_bars": int(round(np.median(cb_vals))),
        "vwap_threshold_pct": float(np.mean(vt_vals)),
    }

    diagnostics = {
        "pf_std_across_seeds": float(np.std(pf_vals)),
        "cb_std_across_seeds": float(np.std(cb_vals)),
        "vt_std_across_seeds": float(np.std(vt_vals)),
        "score_mean": float(np.mean(score_vals)),
        "score_std": float(np.std(score_vals)),
    }

    return avg_params, np.mean(score_vals), diagnostics


def run_grid_search(train_df):
    best_score, best_params = -np.inf, None
    combos = list(itertools.product(*GRID.values()))
    for pf, cb, vt in combos:
        equity, _ = run_strategy(
            train_df, position_fraction=pf, confirmation_bars=cb,
            vwap_threshold_pct=vt, initial_capital=INITIAL_CAPITAL
        )
        score = compute_robust_score(equity)
        if score > best_score:
            best_score = score
            best_params = {
                "position_fraction": pf,
                "confirmation_bars": cb,
                "vwap_threshold_pct": vt,
            }
    return best_params, best_score, len(combos)


def stitch(equity_list, initial_capital=INITIAL_CAPITAL):
    full_series = []
    running_capital = initial_capital
    for eq in equity_list:
        if len(eq) == 0:
            continue
        rescale = running_capital / initial_capital
        rescaled = eq * rescale
        full_series.append(rescaled)
        running_capital = rescaled.iloc[-1]
    return pd.concat(full_series)


def main():
    print("Loading data...")
    df = load_data()
    folds = make_folds(df)

    print(f"\nWalk-forward folds: {len(folds)}")
    for train_years, test_year, _, _ in folds:
        print(f"  Train: {train_years[0]}-{train_years[-1]}  ->  Test: {test_year}")

    print(f"\nGrid search space size: {len(list(itertools.product(*GRID.values())))} combinations")

    bayes_eqs, grid_eqs, base_eqs, fold_rows = [], [], [], []

    for train_years, test_year, train_df, test_df in folds:
        print(f"\n{'='*70}\nFOLD: Train {train_years[0]}-{train_years[-1]} -> Test {test_year}\n{'='*70}")

        t0 = timer.time()
        bayes_params, bayes_train_score, bayes_diag = run_bayesian_opt(train_df, n_calls=15)
        t_bayes = timer.time() - t0
        print(f"Bayesian: {bayes_params} | train robust_score={bayes_train_score:.2f} | {t_bayes:.1f}s | 45 evals (3 seeds x 15)")
        print(f"  Cross-seed spread: pf_std={bayes_diag['pf_std_across_seeds']:.3f} | "
              f"cb_std={bayes_diag['cb_std_across_seeds']:.2f} | "
              f"vt_std={bayes_diag['vt_std_across_seeds']:.3f}")

        t0 = timer.time()
        grid_params, grid_train_score, n_evals = run_grid_search(train_df)
        t_grid = timer.time() - t0
        print(f"Grid:     {grid_params} | train robust_score={grid_train_score:.2f} | {t_grid:.1f}s | {n_evals} evals")

        base_params = {"position_fraction": 1.0, "confirmation_bars": 1, "vwap_threshold_pct": 0.0}

        bayes_eq, _ = run_strategy(test_df, **bayes_params, initial_capital=INITIAL_CAPITAL)
        grid_eq, _ = run_strategy(test_df, **grid_params, initial_capital=INITIAL_CAPITAL)
        base_eq, _ = run_strategy(test_df, **base_params, initial_capital=INITIAL_CAPITAL)

        b_s, g_s, base_s = compute_sharpe(bayes_eq), compute_sharpe(grid_eq), compute_sharpe(base_eq)
        print(f"\nOOS Sharpe {test_year}: baseline={base_s:.2f} | grid={g_s:.2f} | bayesian={b_s:.2f}")

        fold_rows.append({
            "test_year": test_year, "baseline_oos_sharpe": base_s,
            "grid_oos_sharpe": g_s, "bayes_oos_sharpe": b_s,
            "bayes_params": bayes_params, "grid_params": grid_params,
            "bayes_train_score": bayes_train_score, "grid_train_score": grid_train_score,
            "bayes_time_sec": t_bayes, "grid_time_sec": t_grid, "grid_evals": n_evals,
            "bayes_pf_std": bayes_diag["pf_std_across_seeds"],
            "bayes_cb_std": bayes_diag["cb_std_across_seeds"],
            "bayes_vt_std": bayes_diag["vt_std_across_seeds"],
        })

        bayes_eqs.append(bayes_eq)
        grid_eqs.append(grid_eq)
        base_eqs.append(base_eq)

    bayes_full, grid_full, base_full = stitch(bayes_eqs), stitch(grid_eqs), stitch(base_eqs)
    summary = pd.DataFrame(fold_rows)

    print(f"\n{'='*70}\nWALK-FORWARD SUMMARY\n{'='*70}")
    print(summary[["test_year", "baseline_oos_sharpe", "grid_oos_sharpe", "bayes_oos_sharpe"]].to_string(index=False))
    print(f"\nFull-period OOS Sharpe -> baseline: {compute_sharpe(base_full):.2f} | "
          f"grid: {compute_sharpe(grid_full):.2f} | bayesian: {compute_sharpe(bayes_full):.2f}")
    print(f"Avg search time/fold -> grid: {summary['grid_time_sec'].mean():.1f}s ({summary['grid_evals'].iloc[0]} evals) | "
          f"bayesian: {summary['bayes_time_sec'].mean():.1f}s (45 evals = 3 seeds x 15)")

    print("\nParameter stability check across FOLDS (std across folds - lower is more stable):")
    for param in ["position_fraction", "confirmation_bars", "vwap_threshold_pct"]:
        bayes_vals = [row["bayes_params"][param] for row in fold_rows]
        grid_vals = [row["grid_params"][param] for row in fold_rows]
        print(f"  {param}: bayesian std={np.std(bayes_vals):.3f} | grid std={np.std(grid_vals):.3f}")

    print("\nBayesian cross-seed stability WITHIN each fold (lower = more reliable single run):")
    print(f"  avg pf_std={summary['bayes_pf_std'].mean():.3f} | "
          f"avg cb_std={summary['bayes_cb_std'].mean():.2f} | "
          f"avg vt_std={summary['bayes_vt_std'].mean():.3f}")

    summary.to_csv("results/walk_forward_summary_v3.csv", index=False)

    plt.figure(figsize=(12, 7))
    base_full.plot(label="Baseline (fixed params)", alpha=0.8)
    grid_full.plot(label="Grid Search (tuned per fold)", alpha=0.8)
    bayes_full.plot(label="Bayesian Optimization, multi-seed (tuned per fold)", linewidth=2)
    plt.yscale("log")
    plt.title("Walk-Forward OOS Equity: Baseline vs Grid Search vs Bayesian Optimization (v3)")
    plt.ylabel("Account Value ($, log scale)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/walk_forward_comparison_v3.png", dpi=150)
    print("\nSaved: results/walk_forward_comparison_v3.png, results/walk_forward_summary_v3.csv")


if __name__ == "__main__":
    main()
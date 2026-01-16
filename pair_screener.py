import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint


def download_universe_data(tickers, start, end):
    print(f"Downloading data for {len(tickers)} tickers...")
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError("No data downloaded.")

    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("Expected MultiIndex columns from yfinance.")

    close = df["Close"].copy()
    close = close[tickers].dropna()

    print(f"Downloaded {len(close)} rows.")
    return close


def split_train_test(df: pd.DataFrame, split_date: str):
    train = df[df.index < split_date].copy()
    test = df[df.index >= split_date].copy()

    if train.empty or test.empty:
        raise ValueError("Train/test split produced empty dataset.")

    return train, test


def cointegration_test(train_df: pd.DataFrame, ticker1: str, ticker2: str):
    score, pvalue, _ = coint(train_df[ticker1], train_df[ticker2])
    return score, pvalue


def estimate_hedge_ratio_ols(train_df: pd.DataFrame, ticker1: str, ticker2: str):
    y = train_df[ticker1]
    x = sm.add_constant(train_df[ticker2])
    model = sm.OLS(y, x).fit()

    intercept = model.params["const"]
    beta = model.params[ticker2]

    return intercept, beta


def compute_spread(df: pd.DataFrame, ticker1: str, ticker2: str, intercept: float, beta: float):
    out = df.copy()
    out["spread"] = out[ticker1] - (intercept + beta * out[ticker2])
    return out


def compute_rolling_spread(df: pd.DataFrame, ticker1: str, ticker2: str, rolling_window: int = 60):
    out = df.copy()
    intercepts = []
    betas = []

    for i in range(len(out)):
        if i < rolling_window:
            intercepts.append(np.nan)
            betas.append(np.nan)
            continue

        sub = out.iloc[i - rolling_window:i]
        y = sub[ticker1]
        x = sm.add_constant(sub[ticker2])
        model = sm.OLS(y, x).fit()

        intercepts.append(model.params["const"])
        betas.append(model.params[ticker2])

    out["rolling_intercept"] = intercepts
    out["rolling_beta"] = betas
    out["spread"] = out[ticker1] - (out["rolling_intercept"] + out["rolling_beta"] * out[ticker2])

    return out


def generate_pairs_signals(df, window=20, entry_z=1.5, exit_z=0.5):
    df = df.copy()

    df["spread_mean"] = df["spread"].rolling(window).mean()
    df["spread_std"] = df["spread"].rolling(window).std()
    df["zscore"] = (df["spread"] - df["spread_mean"]) / df["spread_std"]

    pair_position = np.zeros(len(df))

    for i in range(1, len(df)):
        prev_pos = pair_position[i - 1]
        z = df["zscore"].iloc[i]

        if np.isnan(z):
            pair_position[i] = prev_pos
            continue

        if prev_pos == 0:
            if z < -entry_z:
                pair_position[i] = 1   # long spread
            elif z > entry_z:
                pair_position[i] = -1  # short spread
            else:
                pair_position[i] = 0
        elif prev_pos == 1:
            if z > -exit_z:
                pair_position[i] = 0
            else:
                pair_position[i] = 1
        elif prev_pos == -1:
            if z < exit_z:
                pair_position[i] = 0
            else:
                pair_position[i] = -1

    df["pair_position"] = pair_position
    return df


def run_pairs_backtest(df, ticker1, ticker2, beta=None, beta_col=None, cost_per_turnover=0.0005):
    df = df.copy()

    df[f"{ticker1}_ret"] = df[ticker1].pct_change().fillna(0)
    df[f"{ticker2}_ret"] = df[ticker2].pct_change().fillna(0)

    if beta_col is not None:
        df["beta_used"] = df[beta_col].ffill()
    elif beta is not None:
        df["beta_used"] = beta
    else:
        raise ValueError("Must provide either beta or beta_col.")

    df = df.dropna(subset=["spread", "beta_used"]).copy()

    df["pair_position_lagged"] = df["pair_position"].shift(1).fillna(0)

    beta_abs = df["beta_used"].abs()
    w1 = 1.0 / (1.0 + beta_abs)
    w2 = beta_abs / (1.0 + beta_abs)

    df["w1"] = df["pair_position_lagged"] * w1
    df["w2"] = -df["pair_position_lagged"] * np.sign(df["beta_used"]) * w2

    df["strategy_gross_return"] = (
        df["w1"] * df[f"{ticker1}_ret"] +
        df["w2"] * df[f"{ticker2}_ret"]
    )

    df["turnover"] = df["pair_position"].diff().abs().fillna(0)
    df["transaction_cost"] = df["turnover"] * cost_per_turnover
    df["strategy_net_return"] = df["strategy_gross_return"] - df["transaction_cost"]

    df["benchmark_return"] = 0.5 * df[f"{ticker1}_ret"] + 0.5 * df[f"{ticker2}_ret"]

    df["cum_strategy"] = (1 + df["strategy_net_return"]).cumprod()
    df["cum_benchmark"] = (1 + df["benchmark_return"]).cumprod()

    return df


def compute_metrics(returns: pd.Series):
    returns = returns.dropna()

    if len(returns) == 0:
        return {}

    mean_daily = returns.mean()
    std_daily = returns.std()

    annual_return = (1 + mean_daily) ** 252 - 1
    annual_vol = std_daily * np.sqrt(252)
    sharpe = annual_return / annual_vol if annual_vol != 0 else np.nan

    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = cumulative / rolling_max - 1
    max_drawdown = drawdown.min()

    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else np.nan
    win_rate = (returns > 0).mean()

    return {
        "annual_return": annual_return,
        "annual_volatility": annual_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar,
        "win_rate": win_rate
    }


def evaluate_strategy_for_pair(
    train_pair_df,
    test_pair_df,
    ticker1,
    ticker2,
    window,
    entry_z,
    exit_z,
    cost,
    use_rolling_beta=True,
    rolling_beta_window=60
):
    if use_rolling_beta:
        train_spread_df = compute_rolling_spread(train_pair_df, ticker1, ticker2, rolling_window=rolling_beta_window)
        test_spread_df = compute_rolling_spread(test_pair_df, ticker1, ticker2, rolling_window=rolling_beta_window)

        train_signal = generate_pairs_signals(train_spread_df, window=window, entry_z=entry_z, exit_z=exit_z)
        test_signal = generate_pairs_signals(test_spread_df, window=window, entry_z=entry_z, exit_z=exit_z)

        train_bt = run_pairs_backtest(
            train_signal, ticker1, ticker2, beta_col="rolling_beta", cost_per_turnover=cost
        )
        test_bt = run_pairs_backtest(
            test_signal, ticker1, ticker2, beta_col="rolling_beta", cost_per_turnover=cost
        )

        intercept = np.nan
        beta = np.nan

    else:
        intercept, beta = estimate_hedge_ratio_ols(train_pair_df, ticker1, ticker2)

        train_spread_df = compute_spread(train_pair_df, ticker1, ticker2, intercept, beta)
        test_spread_df = compute_spread(test_pair_df, ticker1, ticker2, intercept, beta)

        train_signal = generate_pairs_signals(train_spread_df, window=window, entry_z=entry_z, exit_z=exit_z)
        test_signal = generate_pairs_signals(test_spread_df, window=window, entry_z=entry_z, exit_z=exit_z)

        train_bt = run_pairs_backtest(
            train_signal, ticker1, ticker2, beta=beta, cost_per_turnover=cost
        )
        test_bt = run_pairs_backtest(
            test_signal, ticker1, ticker2, beta=beta, cost_per_turnover=cost
        )

    train_metrics = compute_metrics(train_bt["strategy_net_return"]) if not train_bt.empty else {}
    test_metrics = compute_metrics(test_bt["strategy_net_return"]) if not test_bt.empty else {}

    return {
        "intercept": intercept,
        "beta": beta,
        "train_bt": train_bt,
        "test_bt": test_bt,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics
    }


def grid_search_pair(
    train_pair_df,
    test_pair_df,
    ticker1,
    ticker2,
    cost=0.0005,
    use_rolling_beta=False,
    rolling_beta_window=60
):
    windows = [10, 20, 30, 40]
    entry_thresholds = [1.0, 1.5, 2.0]
    exit_thresholds = [0.25, 0.5, 1.0]

    results = []
    saved_runs = {}

    for window in windows:
        for entry_z in entry_thresholds:
            for exit_z in exit_thresholds:
                if exit_z >= entry_z:
                    continue

                try:
                    eval_result = evaluate_strategy_for_pair(
                        train_pair_df=train_pair_df,
                        test_pair_df=test_pair_df,
                        ticker1=ticker1,
                        ticker2=ticker2,
                        window=window,
                        entry_z=entry_z,
                        exit_z=exit_z,
                        cost=cost,
                        use_rolling_beta=use_rolling_beta,
                        rolling_beta_window=rolling_beta_window
                    )

                    train_metrics = eval_result["train_metrics"]
                    test_metrics = eval_result["test_metrics"]

                    row = {
                        "window": window,
                        "entry_z": entry_z,
                        "exit_z": exit_z,
                        "train_annual_return": train_metrics.get("annual_return", np.nan),
                        "train_sharpe": train_metrics.get("sharpe_ratio", np.nan),
                        "train_max_drawdown": train_metrics.get("max_drawdown", np.nan),
                        "test_annual_return": test_metrics.get("annual_return", np.nan),
                        "test_sharpe": test_metrics.get("sharpe_ratio", np.nan),
                        "test_max_drawdown": test_metrics.get("max_drawdown", np.nan),
                    }
                    results.append(row)

                    key = (window, entry_z, exit_z)
                    saved_runs[key] = eval_result

                except Exception as e:
                    continue

    results_df = pd.DataFrame(results)
    if results_df.empty:
        return results_df, None, None

    results_df = results_df.sort_values(by="train_sharpe", ascending=False).reset_index(drop=True)

    best = results_df.iloc[0]
    best_key = (int(best["window"]), float(best["entry_z"]), float(best["exit_z"]))
    best_eval = saved_runs[best_key]

    return results_df, best, best_eval


def plot_pair_backtest(df, ticker1, ticker2, filename):
    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

    axes[0].plot(df.index, df[ticker1], label=ticker1)
    axes[0].plot(df.index, df[ticker2], label=ticker2)
    axes[0].legend()
    axes[0].set_title(f"{ticker1} and {ticker2} Prices")

    axes[1].plot(df.index, df["spread"], color="purple")
    axes[1].set_title("Spread")

    axes[2].plot(df.index, df["zscore"], color="orange")
    axes[2].axhline(0, color="black")
    axes[2].set_title("Spread Z-score")

    axes[3].plot(df.index, df["cum_strategy"], label="Strategy", color="green")
    axes[3].plot(df.index, df["cum_benchmark"], label="Benchmark", color="gray")
    axes[3].legend()
    axes[3].set_title("Cumulative Returns")

    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def main():
    os.makedirs("outputs", exist_ok=True)

    tickers = ["KO", "PEP", "XOM", "CVX", "JPM", "BAC", "V", "MA", "SPY", "IVV"]
    start = "2020-01-01"
    end = "2025-01-01"
    split_date = "2024-01-01"

    cost = 0.0005
    max_pvalue = 0.10
    use_rolling_beta = True
    rolling_beta_window = 60

    prices = download_universe_data(tickers, start, end)
    train_prices, test_prices = split_train_test(prices, split_date)

    all_pairs = list(itertools.combinations(tickers, 2))
    print(f"Testing {len(all_pairs)} candidate pairs...\n")

    summary_rows = []
    saved_pair_details = {}

    for ticker1, ticker2 in all_pairs:
        try:
            train_pair_df = train_prices[[ticker1, ticker2]].dropna().copy()
            test_pair_df = test_prices[[ticker1, ticker2]].dropna().copy()

            if len(train_pair_df) < 100 or len(test_pair_df) < 50:
                continue

            score, pvalue = cointegration_test(train_pair_df, ticker1, ticker2)

            print(f"Evaluating {ticker1}-{ticker2} | coint p-value={pvalue:.4f}")

            if pvalue > max_pvalue:
                print("  Skipping due to weak cointegration.\n")
                continue

            grid_results, best_params, best_eval = grid_search_pair(
                train_pair_df=train_pair_df,
                test_pair_df=test_pair_df,
                ticker1=ticker1,
                ticker2=ticker2,
                cost=cost,
                use_rolling_beta=use_rolling_beta,
                rolling_beta_window=rolling_beta_window
            )

            if grid_results.empty or best_params is None or best_eval is None:
                print("  No valid parameter combinations.\n")
                continue

            train_metrics = best_eval["train_metrics"]
            test_metrics = best_eval["test_metrics"]

            summary_row = {
                "pair": f"{ticker1}-{ticker2}",
                "ticker1": ticker1,
                "ticker2": ticker2,
                "cointegration_score": score,
                "cointegration_pvalue": pvalue,
                "best_window": int(best_params["window"]),
                "best_entry_z": float(best_params["entry_z"]),
                "best_exit_z": float(best_params["exit_z"]),
                "train_annual_return": train_metrics.get("annual_return", np.nan),
                "train_sharpe": train_metrics.get("sharpe_ratio", np.nan),
                "train_max_drawdown": train_metrics.get("max_drawdown", np.nan),
                "test_annual_return": test_metrics.get("annual_return", np.nan),
                "test_sharpe": test_metrics.get("sharpe_ratio", np.nan),
                "test_max_drawdown": test_metrics.get("max_drawdown", np.nan),
            }

            summary_rows.append(summary_row)
            saved_pair_details[f"{ticker1}-{ticker2}"] = {
                "grid_results": grid_results,
                "best_eval": best_eval,
                "ticker1": ticker1,
                "ticker2": ticker2
            }

            print(
                f"  Best train Sharpe={summary_row['train_sharpe']:.4f}, "
                f"test Sharpe={summary_row['test_sharpe']:.4f}\n"
            )

        except Exception as e:
            print(f"  Error on {ticker1}-{ticker2}: {e}\n")
            continue

    if not summary_rows:
        print("No pairs passed screening.")
        return

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(by="test_sharpe", ascending=False).reset_index(drop=True)
    summary_df.to_csv("outputs/pair_screen_results.csv", index=False)

    print("\nTop pairs by out-of-sample Sharpe:")
    print(summary_df.head(10))

    top_pair = summary_df.iloc[0]["pair"]
    top_info = saved_pair_details[top_pair]

    top_grid = top_info["grid_results"]
    top_eval = top_info["best_eval"]
    ticker1 = top_info["ticker1"]
    ticker2 = top_info["ticker2"]

    top_grid.to_csv(f"outputs/{top_pair}_grid_results.csv", index=False)
    top_eval["train_bt"].to_csv(f"outputs/{top_pair}_train_backtest.csv")
    top_eval["test_bt"].to_csv(f"outputs/{top_pair}_test_backtest.csv")

    plot_pair_backtest(top_eval["train_bt"], ticker1, ticker2, f"outputs/{top_pair}_train_plot.png")
    plot_pair_backtest(top_eval["test_bt"], ticker1, ticker2, f"outputs/{top_pair}_test_plot.png")

    print(f"\nBest pair: {top_pair}")
    print("Saved:")
    print("- pair_screen_results.csv")
    print(f"- {top_pair}_grid_results.csv")
    print(f"- {top_pair}_train_backtest.csv")
    print(f"- {top_pair}_test_backtest.csv")
    print(f"- {top_pair}_train_plot.png")
    print(f"- {top_pair}_test_plot.png")


if __name__ == "__main__":
    main()
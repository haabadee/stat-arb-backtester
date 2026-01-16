import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint


def download_pair_data(ticker1: str, ticker2: str, start: str, end: str) -> pd.DataFrame:
    print(f"Downloading {ticker1} and {ticker2} from {start} to {end}...")
    df = yf.download([ticker1, ticker2], start=start, end=end, auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError("No data downloaded.")

    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("Expected MultiIndex columns when downloading multiple tickers.")

    close = df["Close"].copy()
    close = close[[ticker1, ticker2]].dropna()
    close.columns = [ticker1, ticker2]

    print(f"Downloaded {len(close)} rows.")
    return close


def split_train_test(df: pd.DataFrame, split_date: str):
    train = df[df.index < split_date].copy()
    test = df[df.index >= split_date].copy()

    if train.empty or test.empty:
        raise ValueError("Train/test split produced an empty dataset.")

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

    return intercept, beta, model


def compute_spread(df: pd.DataFrame, ticker1: str, ticker2: str, intercept: float, beta: float):
    df = df.copy()
    df["spread"] = df[ticker1] - (intercept + beta * df[ticker2])
    return df


def compute_rolling_hedge_ratio(df: pd.DataFrame, ticker1: str, ticker2: str, window: int = 60):
    """
    Rolling OLS hedge ratio estimation.
    """
    df = df.copy()
    betas = []
    intercepts = []

    for i in range(len(df)):
        if i < window:
            betas.append(np.nan)
            intercepts.append(np.nan)
            continue

        sub = df.iloc[i - window:i]
        y = sub[ticker1]
        x = sm.add_constant(sub[ticker2])
        model = sm.OLS(y, x).fit()

        intercepts.append(model.params["const"])
        betas.append(model.params[ticker2])

    df["rolling_intercept"] = intercepts
    df["rolling_beta"] = betas
    df["rolling_spread"] = df[ticker1] - (df["rolling_intercept"] + df["rolling_beta"] * df[ticker2])

    return df


def generate_pairs_signals(
    df: pd.DataFrame,
    spread_col: str = "spread",
    window: int = 20,
    entry_z: float = 1.5,
    exit_z: float = 0.5
):
    df = df.copy()

    df["spread_mean"] = df[spread_col].rolling(window).mean()
    df["spread_std"] = df[spread_col].rolling(window).std()
    df["zscore"] = (df[spread_col] - df["spread_mean"]) / df["spread_std"]

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


def run_pairs_backtest(
    df: pd.DataFrame,
    ticker1: str,
    ticker2: str,
    beta_col: str = None,
    static_beta: float = None,
    cost_per_turnover: float = 0.0005
):
    df = df.copy()

    df[f"{ticker1}_ret"] = df[ticker1].pct_change().fillna(0)
    df[f"{ticker2}_ret"] = df[ticker2].pct_change().fillna(0)

    df["pair_position_lagged"] = df["pair_position"].shift(1).fillna(0)

    if beta_col is not None:
        beta_series = df[beta_col].ffill()
    elif static_beta is not None:
        beta_series = pd.Series(static_beta, index=df.index)
    else:
        raise ValueError("Provide either beta_col or static_beta.")

    # if rolling beta still starts with NaNs, drop those rows safely
    df["beta_series"] = beta_series
    df = df.dropna(subset=["beta_series"]).copy()

    # Need to recompute lagged positions after dropping rows
    df["pair_position_lagged"] = df["pair_position"].shift(1).fillna(0)

    # Normalize gross exposure so |w1| + |w2| = 1
    w1 = 1.0 / (1.0 + df["beta_series"].abs())
    w2 = df["beta_series"].abs() / (1.0 + df["beta_series"].abs())

    # long spread: +w1 in ticker1, -w2 in ticker2
    # short spread: -w1 in ticker1, +w2 in ticker2
    df["w1"] = df["pair_position_lagged"] * w1
    df["w2"] = -df["pair_position_lagged"] * np.sign(df["beta_series"]) * w2

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


def compute_metrics(returns: pd.Series) -> dict:
    returns = returns.dropna()

    if len(returns) == 0:
        return {}

    mean_daily = returns.mean()
    std_daily = returns.std()

    annual_return = (1 + mean_daily) ** 252 - 1
    annual_volatility = std_daily * np.sqrt(252)
    sharpe = annual_return / annual_volatility if annual_volatility != 0 else np.nan

    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = cumulative / rolling_max - 1
    max_drawdown = drawdown.min()

    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else np.nan
    win_rate = (returns > 0).mean()

    return {
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar,
        "win_rate": win_rate
    }


def evaluate_pair_strategy(
    df: pd.DataFrame,
    ticker1: str,
    ticker2: str,
    spread_col: str,
    window: int,
    entry_z: float,
    exit_z: float,
    cost: float,
    beta_col: str = None,
    static_beta: float = None
):
    signal_df = generate_pairs_signals(
        df, spread_col=spread_col, window=window, entry_z=entry_z, exit_z=exit_z
    )

    bt_df = run_pairs_backtest(
        signal_df,
        ticker1=ticker1,
        ticker2=ticker2,
        beta_col=beta_col,
        static_beta=static_beta,
        cost_per_turnover=cost
    )

    if bt_df.empty:
        return bt_df, {}

    metrics = compute_metrics(bt_df["strategy_net_return"])
    return bt_df, metrics


def grid_search_pairs(
    train_df: pd.DataFrame,
    ticker1: str,
    ticker2: str,
    spread_col: str,
    cost: float = 0.0005,
    beta_col: str = None,
    static_beta: float = None
):
    windows = [10, 20, 30, 40]
    entry_thresholds = [1.0, 1.5, 2.0]
    exit_thresholds = [0.25, 0.5, 1.0]

    results = []

    for window in windows:
        for entry_z in entry_thresholds:
            for exit_z in exit_thresholds:
                if exit_z >= entry_z:
                    continue

                _, metrics = evaluate_pair_strategy(
                    train_df,
                    ticker1=ticker1,
                    ticker2=ticker2,
                    spread_col=spread_col,
                    window=window,
                    entry_z=entry_z,
                    exit_z=exit_z,
                    cost=cost,
                    beta_col=beta_col,
                    static_beta=static_beta
                )

                results.append({
                    "window": window,
                    "entry_z": entry_z,
                    "exit_z": exit_z,
                    "annual_return": metrics.get("annual_return", np.nan),
                    "annual_volatility": metrics.get("annual_volatility", np.nan),
                    "sharpe_ratio": metrics.get("sharpe_ratio", np.nan),
                    "max_drawdown": metrics.get("max_drawdown", np.nan)
                })

    results_df = pd.DataFrame(results).sort_values(by="sharpe_ratio", ascending=False)
    return results_df


def save_plot(df: pd.DataFrame, ticker1: str, ticker2: str, spread_col: str, filename: str):
    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

    axes[0].plot(df.index, df[ticker1], label=ticker1)
    axes[0].plot(df.index, df[ticker2], label=ticker2)
    axes[0].set_title(f"{ticker1} and {ticker2} Prices")
    axes[0].legend()

    axes[1].plot(df.index, df[spread_col], color="purple")
    axes[1].set_title(f"{spread_col}")

    axes[2].plot(df.index, df["zscore"], color="orange")
    axes[2].axhline(0, color="black")
    axes[2].set_title("Spread Z-score")

    axes[3].plot(df.index, df["cum_strategy"], label="Strategy", color="green")
    axes[3].plot(df.index, df["cum_benchmark"], label="Benchmark", color="gray")
    axes[3].set_title("Cumulative Returns")
    axes[3].legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"Saved plot to {filename}")


def print_metrics(title: str, metrics: dict):
    print(f"\n{title}")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


def main():
    os.makedirs("outputs", exist_ok=True)

    ticker1 = "KO"
    ticker2 = "PEP"
    start = "2020-01-01"
    end = "2025-01-01"
    split_date = "2024-01-01"
    cost = 0.0005

    use_rolling_beta = True
    rolling_beta_window = 60

    df = download_pair_data(ticker1, ticker2, start, end)
    train_df, test_df = split_train_test(df, split_date)

    score, pvalue = cointegration_test(train_df, ticker1, ticker2)
    print(f"Cointegration test score: {score:.4f}")
    print(f"Cointegration p-value: {pvalue:.6f}")

    intercept, beta, model = estimate_hedge_ratio_ols(train_df, ticker1, ticker2)
    print(f"Static intercept: {intercept:.4f}")
    print(f"Static beta: {beta:.4f}")

    if use_rolling_beta:
        print(f"Using rolling hedge ratio with window={rolling_beta_window}")
        train_df = compute_rolling_hedge_ratio(train_df, ticker1, ticker2, rolling_beta_window)
        test_df = compute_rolling_hedge_ratio(test_df, ticker1, ticker2, rolling_beta_window)

        spread_col = "rolling_spread"
        beta_col = "rolling_beta"
        static_beta = None
    else:
        train_df = compute_spread(train_df, ticker1, ticker2, intercept, beta)
        test_df = compute_spread(test_df, ticker1, ticker2, intercept, beta)

        spread_col = "spread"
        beta_col = None
        static_beta = beta

    print("Running grid search on training set...")
    grid_results = grid_search_pairs(
        train_df,
        ticker1=ticker1,
        ticker2=ticker2,
        spread_col=spread_col,
        cost=cost,
        beta_col=beta_col,
        static_beta=static_beta
    )
    grid_results.to_csv("outputs/pairs_grid_search_results.csv", index=False)

    best = grid_results.iloc[0]
    best_window = int(best["window"])
    best_entry = float(best["entry_z"])
    best_exit = float(best["exit_z"])

    print("\nBest parameters on training set:")
    print(best)

    train_bt, train_metrics = evaluate_pair_strategy(
        train_df,
        ticker1=ticker1,
        ticker2=ticker2,
        spread_col=spread_col,
        window=best_window,
        entry_z=best_entry,
        exit_z=best_exit,
        cost=cost,
        beta_col=beta_col,
        static_beta=static_beta
    )

    test_bt, test_metrics = evaluate_pair_strategy(
        test_df,
        ticker1=ticker1,
        ticker2=ticker2,
        spread_col=spread_col,
        window=best_window,
        entry_z=best_entry,
        exit_z=best_exit,
        cost=cost,
        beta_col=beta_col,
        static_beta=static_beta
    )

    benchmark_metrics = compute_metrics(test_bt["benchmark_return"])

    print_metrics("Train Strategy Metrics", train_metrics)
    print_metrics("Test Strategy Metrics", test_metrics)
    print_metrics("Test Benchmark Metrics", benchmark_metrics)

    train_bt.to_csv("outputs/pairs_train_backtest.csv")
    test_bt.to_csv("outputs/pairs_test_backtest.csv")

    save_plot(train_bt, ticker1, ticker2, spread_col, "outputs/pairs_train_plot.png")
    save_plot(test_bt, ticker1, ticker2, spread_col, "outputs/pairs_test_plot.png")

    print("\nDone. Outputs saved to outputs/")


if __name__ == "__main__":
    main()
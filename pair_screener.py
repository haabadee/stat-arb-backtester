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
                pair_position[i] = 1
            elif z > entry_z:
                pair_position[i] = -1
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


def run_pairs_backtest(df, ticker1, ticker2, beta, cost_per_turnover=0.0005):
    df = df.copy()

    df[f"{ticker1}_ret"] = df[ticker1].pct_change().fillna(0)
    df[f"{ticker2}_ret"] = df[ticker2].pct_change().fillna(0)

    df["pair_position_lagged"] = df["pair_position"].shift(1).fillna(0)

    beta_abs = abs(beta)
    w1 = 1.0 / (1.0 + beta_abs)
    w2 = beta_abs / (1.0 + beta_abs)

    df["w1"] = df["pair_position_lagged"] * w1
    df["w2"] = -df["pair_position_lagged"] * np.sign(beta) * w2

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


def evaluate_pair(train_df, test_df, ticker1, ticker2, window=20, entry_z=1.5, exit_z=0.5, cost=0.0005):
    score, pvalue = cointegration_test(train_df, ticker1, ticker2)

    intercept, beta = estimate_hedge_ratio_ols(train_df, ticker1, ticker2)

    train_spread = compute_spread(train_df[[ticker1, ticker2]], ticker1, ticker2, intercept, beta)
    test_spread = compute_spread(test_df[[ticker1, ticker2]], ticker1, ticker2, intercept, beta)

    train_signal = generate_pairs_signals(train_spread, window=window, entry_z=entry_z, exit_z=exit_z)
    test_signal = generate_pairs_signals(test_spread, window=window, entry_z=entry_z, exit_z=exit_z)

    train_bt = run_pairs_backtest(train_signal, ticker1, ticker2, beta, cost_per_turnover=cost)
    test_bt = run_pairs_backtest(test_signal, ticker1, ticker2, beta, cost_per_turnover=cost)

    train_metrics = compute_metrics(train_bt["strategy_net_return"])
    test_metrics = compute_metrics(test_bt["strategy_net_return"])

    return {
        "pair": f"{ticker1}-{ticker2}",
        "ticker1": ticker1,
        "ticker2": ticker2,
        "cointegration_score": score,
        "cointegration_pvalue": pvalue,
        "intercept": intercept,
        "beta": beta,
        "train_annual_return": train_metrics.get("annual_return", np.nan),
        "train_sharpe": train_metrics.get("sharpe_ratio", np.nan),
        "train_max_drawdown": train_metrics.get("max_drawdown", np.nan),
        "test_annual_return": test_metrics.get("annual_return", np.nan),
        "test_sharpe": test_metrics.get("sharpe_ratio", np.nan),
        "test_max_drawdown": test_metrics.get("max_drawdown", np.nan),
    }, train_bt, test_bt


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
    axes[2].set_title("Spread Z-Score")

    axes[3].plot(df.index, df["cum_strategy"], label="Strategy", color="green")
    axes[3].plot(df.index, df["cum_benchmark"], label="Benchmark", color="gray")
    axes[3].legend()
    axes[3].set_title("Cumulative Returns")

    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"Saved plot to {filename}")


def main():
    os.makedirs("outputs", exist_ok=True)

    tickers = ["KO", "PEP", "XOM", "CVX", "JPM", "BAC", "V", "MA", "SPY", "IVV"]
    start = "2020-01-01"
    end = "2025-01-01"
    split_date = "2024-01-01"

    entry_z = 1.5
    exit_z = 0.5
    window = 20
    cost = 0.0005
    max_pvalue = 0.10   # screening threshold

    prices = download_universe_data(tickers, start, end)
    train_prices, test_prices = split_train_test(prices, split_date)

    pair_results = []
    saved_backtests = {}

    all_pairs = list(itertools.combinations(tickers, 2))
    print(f"Testing {len(all_pairs)} pairs...")

    for ticker1, ticker2 in all_pairs:
        try:
            result, train_bt, test_bt = evaluate_pair(
                train_prices,
                test_prices,
                ticker1,
                ticker2,
                window=window,
                entry_z=entry_z,
                exit_z=exit_z,
                cost=cost
            )

            if result["cointegration_pvalue"] <= max_pvalue:
                pair_results.append(result)
                saved_backtests[result["pair"]] = (train_bt, test_bt)
                print(
                    f"{result['pair']}: p={result['cointegration_pvalue']:.4f}, "
                    f"train_sharpe={result['train_sharpe']:.4f}, "
                    f"test_sharpe={result['test_sharpe']:.4f}"
                )

        except Exception as e:
            print(f"Skipping {ticker1}-{ticker2}: {e}")

    if not pair_results:
        print("No pairs passed the screening threshold.")
        return

    results_df = pd.DataFrame(pair_results)
    results_df = results_df.sort_values(by="test_sharpe", ascending=False)
    results_df.to_csv("outputs/pair_screen_results.csv", index=False)

    print("\nTop pairs by test Sharpe:")
    print(results_df.head(10))

    top_pair = results_df.iloc[0]["pair"]
    top_ticker1 = results_df.iloc[0]["ticker1"]
    top_ticker2 = results_df.iloc[0]["ticker2"]

    top_train_bt, top_test_bt = saved_backtests[top_pair]
    top_train_bt.to_csv(f"outputs/{top_pair}_train_backtest.csv")
    top_test_bt.to_csv(f"outputs/{top_pair}_test_backtest.csv")

    plot_pair_backtest(top_train_bt, top_ticker1, top_ticker2, f"outputs/{top_pair}_train_plot.png")
    plot_pair_backtest(top_test_bt, top_ticker1, top_ticker2, f"outputs/{top_pair}_test_plot.png")

    print(f"\nBest pair: {top_pair}")
    print("Detailed outputs saved to outputs/")


if __name__ == "__main__":
    main()
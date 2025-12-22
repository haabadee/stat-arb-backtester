import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


def download_pair_data(ticker1: str, ticker2: str, start: str, end: str) -> pd.DataFrame:
    print(f"Downloading {ticker1} and {ticker2} from {start} to {end}...")
    df = yf.download([ticker1, ticker2], start=start, end=end, auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError("No data downloaded.")

    if isinstance(df.columns, pd.MultiIndex):
        # Use Close prices only
        close = df['Close'].copy()
    else:
        raise ValueError("Expected MultiIndex columns from yfinance when downloading multiple tickers.")

    close.columns = [ticker1, ticker2]
    close = close.dropna()
    print(f"Downloaded {len(close)} rows.")
    return close


def split_train_test(df: pd.DataFrame, split_date: str):
    train = df[df.index < split_date].copy()
    test = df[df.index >= split_date].copy()

    if train.empty or test.empty:
        raise ValueError("Train/test split produced empty dataset.")

    return train, test


def estimate_hedge_ratio(train_df: pd.DataFrame, ticker1: str, ticker2: str) -> float:
    y = train_df[ticker1].values
    x = train_df[ticker2].values

    # OLS slope with intercept omitted for simplicity
    beta = np.polyfit(x, y, 1)[0]
    return beta


def compute_spread(df: pd.DataFrame, ticker1: str, ticker2: str, beta: float) -> pd.DataFrame:
    df = df.copy()
    df['spread'] = df[ticker1] - beta * df[ticker2]
    return df


def generate_pairs_signals(
    df: pd.DataFrame,
    window: int = 20,
    entry_z: float = 1.5,
    exit_z: float = 0.5
) -> pd.DataFrame:
    df = df.copy()

    df['spread_mean'] = df['spread'].rolling(window).mean()
    df['spread_std'] = df['spread'].rolling(window).std()
    df['zscore'] = (df['spread'] - df['spread_mean']) / df['spread_std']

    # pair_position:
    # +1 => long spread  => long ticker1, short beta*ticker2
    # -1 => short spread => short ticker1, long beta*ticker2
    pair_position = np.zeros(len(df))

    for i in range(1, len(df)):
        prev_pos = pair_position[i - 1]
        z = df['zscore'].iloc[i]

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

    df['pair_position'] = pair_position
    return df


def run_pairs_backtest(
    df: pd.DataFrame,
    ticker1: str,
    ticker2: str,
    beta: float,
    cost_per_turnover: float = 0.0005
) -> pd.DataFrame:
    df = df.copy()

    df[f'{ticker1}_ret'] = df[ticker1].pct_change().fillna(0)
    df[f'{ticker2}_ret'] = df[ticker2].pct_change().fillna(0)

    df['pair_position_lagged'] = df['pair_position'].shift(1).fillna(0)

    # If pair_position = +1:
    # long ticker1, short beta * ticker2
    # If pair_position = -1:
    # short ticker1, long beta * ticker2
    df['strategy_gross_return'] = (
        df['pair_position_lagged'] * df[f'{ticker1}_ret']
        - df['pair_position_lagged'] * beta * df[f'{ticker2}_ret']
    )

    df['turnover'] = df['pair_position'].diff().abs().fillna(0)
    df['transaction_cost'] = df['turnover'] * cost_per_turnover
    df['strategy_net_return'] = df['strategy_gross_return'] - df['transaction_cost']

    # simple benchmark: equal-weight long both
    df['benchmark_return'] = 0.5 * df[f'{ticker1}_ret'] + 0.5 * df[f'{ticker2}_ret']

    df['cum_strategy'] = (1 + df['strategy_net_return']).cumprod()
    df['cum_benchmark'] = (1 + df['benchmark_return']).cumprod()

    return df


def compute_metrics(returns: pd.Series) -> dict:
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

    win_rate = (returns > 0).mean()

    return {
        "annual_return": annual_return,
        "annual_volatility": annual_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate
    }


def evaluate_pair_strategy(df, ticker1, ticker2, beta, window, entry_z, exit_z, cost):
    signal_df = generate_pairs_signals(df, window=window, entry_z=entry_z, exit_z=exit_z)
    bt_df = run_pairs_backtest(signal_df, ticker1, ticker2, beta, cost_per_turnover=cost)
    metrics = compute_metrics(bt_df['strategy_net_return'])
    return bt_df, metrics


def grid_search_pairs(train_df, ticker1, ticker2, beta, cost=0.0005):
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
                    train_df, ticker1, ticker2, beta, window, entry_z, exit_z, cost
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


def save_plot(df: pd.DataFrame, ticker1: str, ticker2: str, filename: str):
    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

    axes[0].plot(df.index, df[ticker1], label=ticker1)
    axes[0].plot(df.index, df[ticker2], label=ticker2)
    axes[0].set_title(f"{ticker1} and {ticker2} Prices")
    axes[0].legend()

    axes[1].plot(df.index, df['spread'], color='purple')
    axes[1].set_title("Spread")

    axes[2].plot(df.index, df['zscore'], color='orange')
    axes[2].axhline(0, color='black')
    axes[2].set_title("Spread Z-score")

    axes[3].plot(df.index, df['cum_strategy'], label='Strategy', color='green')
    axes[3].plot(df.index, df['cum_benchmark'], label='Benchmark', color='gray')
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

    df = download_pair_data(ticker1, ticker2, start, end)
    train_df, test_df = split_train_test(df, split_date)

    beta = estimate_hedge_ratio(train_df, ticker1, ticker2)
    print(f"Estimated hedge ratio (beta): {beta:.4f}")

    train_df = compute_spread(train_df, ticker1, ticker2, beta)
    test_df = compute_spread(test_df, ticker1, ticker2, beta)

    print("Running grid search on training set...")
    grid_results = grid_search_pairs(train_df, ticker1, ticker2, beta, cost=cost)
    grid_results.to_csv("outputs/pairs_grid_search_results.csv", index=False)

    best = grid_results.iloc[0]
    best_window = int(best['window'])
    best_entry = float(best['entry_z'])
    best_exit = float(best['exit_z'])

    print("\nBest parameters:")
    print(best)

    train_bt, train_metrics = evaluate_pair_strategy(
        train_df, ticker1, ticker2, beta, best_window, best_entry, best_exit, cost
    )
    test_bt, test_metrics = evaluate_pair_strategy(
        test_df, ticker1, ticker2, beta, best_window, best_entry, best_exit, cost
    )

    benchmark_metrics = compute_metrics(test_bt['benchmark_return'])

    print_metrics("Train Strategy Metrics", train_metrics)
    print_metrics("Test Strategy Metrics", test_metrics)
    print_metrics("Test Benchmark Metrics", benchmark_metrics)

    train_bt.to_csv("outputs/pairs_train_backtest.csv")
    test_bt.to_csv("outputs/pairs_test_backtest.csv")

    save_plot(train_bt, ticker1, ticker2, "outputs/pairs_train_plot.png")
    save_plot(test_bt, ticker1, ticker2, "outputs/pairs_test_plot.png")

    print("\nDone. Outputs saved to outputs/")


if __name__ == "__main__":
    main()
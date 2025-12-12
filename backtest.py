import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


def download_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    print(f"Downloading data for {ticker} from {start} to {end}...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(f"No data downloaded for {ticker}")

    # Handle MultiIndex columns if returned
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if 'Close' not in df.columns:
        raise ValueError(f"'Close' column not found. Columns are: {df.columns.tolist()}")

    df = df[['Close']].copy()
    df.rename(columns={'Close': 'price'}, inplace=True)
    df.dropna(inplace=True)

    print(f"Downloaded {len(df)} rows.")
    return df


def generate_mean_reversion_signals(
    df: pd.DataFrame,
    window: int,
    entry_z: float,
    exit_z: float
) -> pd.DataFrame:
    df = df.copy()

    df['returns'] = df['price'].pct_change()
    df['rolling_mean'] = df['price'].rolling(window).mean()
    df['rolling_std'] = df['price'].rolling(window).std()
    df['zscore'] = (df['price'] - df['rolling_mean']) / df['rolling_std']

    position = np.zeros(len(df))

    for i in range(1, len(df)):
        prev_pos = position[i - 1]
        z = df['zscore'].iloc[i]

        if np.isnan(z):
            position[i] = prev_pos
            continue

        if prev_pos == 0:
            if z < -entry_z:
                position[i] = 1
            elif z > entry_z:
                position[i] = -1
            else:
                position[i] = 0
        elif prev_pos == 1:
            if z > -exit_z:
                position[i] = 0
            else:
                position[i] = 1
        elif prev_pos == -1:
            if z < exit_z:
                position[i] = 0
            else:
                position[i] = -1

    df['position'] = position
    return df


def run_backtest(df: pd.DataFrame, cost_per_turnover: float = 0.0005) -> pd.DataFrame:
    df = df.copy()

    df['asset_return'] = df['price'].pct_change().fillna(0)
    df['position_lagged'] = df['position'].shift(1).fillna(0)

    # strategy return before costs
    df['strategy_gross_return'] = df['position_lagged'] * df['asset_return']

    # turnover: 0->1, 1->0, -1->1, etc.
    df['turnover'] = df['position'].diff().abs().fillna(0)

    # costs
    df['transaction_cost'] = df['turnover'] * cost_per_turnover
    df['strategy_net_return'] = df['strategy_gross_return'] - df['transaction_cost']

    df['cum_asset'] = (1 + df['asset_return']).cumprod()
    df['cum_strategy'] = (1 + df['strategy_net_return']).cumprod()

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


def split_train_test(df: pd.DataFrame, split_date: str):
    train = df[df.index < split_date].copy()
    test = df[df.index >= split_date].copy()

    if train.empty or test.empty:
        raise ValueError("Train/test split produced an empty dataset. Choose another split date.")

    return train, test


def evaluate_strategy(df: pd.DataFrame, window: int, entry_z: float, exit_z: float, cost: float):
    signal_df = generate_mean_reversion_signals(df, window, entry_z, exit_z)
    bt_df = run_backtest(signal_df, cost_per_turnover=cost)
    metrics = compute_metrics(bt_df['strategy_net_return'])
    return bt_df, metrics


def grid_search(train_df: pd.DataFrame, cost: float = 0.0005):
    windows = [10, 20, 30, 40]
    entry_thresholds = [1.0, 1.5, 2.0]
    exit_thresholds = [0.25, 0.5, 1.0]

    results = []

    for window in windows:
        for entry_z in entry_thresholds:
            for exit_z in exit_thresholds:
                if exit_z >= entry_z:
                    continue

                _, metrics = evaluate_strategy(train_df, window, entry_z, exit_z, cost)

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


def save_plot(df: pd.DataFrame, ticker: str, filename: str):
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    axes[0].plot(df.index, df['price'], label='Price', color='blue')
    axes[0].set_title(f"{ticker} Price")

    axes[1].plot(df.index, df['zscore'], label='Z-score', color='purple')
    axes[1].axhline(0, color='black', linestyle='-')
    axes[1].set_title("Z-score")

    axes[2].plot(df.index, df['cum_asset'], label='Buy & Hold', color='gray')
    axes[2].plot(df.index, df['cum_strategy'], label='Strategy', color='orange')
    axes[2].set_title("Cumulative Returns")
    axes[2].legend()

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

    ticker = "SPY"
    start = "2020-01-01"
    end = "2025-01-01"
    split_date = "2024-01-01"
    cost = 0.0005

    print("Starting backtest v2...")
    df = download_data(ticker, start, end)

    train_df, test_df = split_train_test(df, split_date)
    print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")

    print("\nRunning parameter search on training set...")
    grid_results = grid_search(train_df, cost=cost)
    grid_results.to_csv("outputs/grid_search_results.csv", index=False)

    best = grid_results.iloc[0]
    best_window = int(best['window'])
    best_entry = float(best['entry_z'])
    best_exit = float(best['exit_z'])

    print("\nBest parameters found on training set:")
    print(best)

    # Evaluate on train
    train_bt, train_metrics = evaluate_strategy(
        train_df, best_window, best_entry, best_exit, cost
    )

    # Evaluate on test
    test_bt, test_metrics = evaluate_strategy(
        test_df, best_window, best_entry, best_exit, cost
    )

    # Buy and hold on test
    bh_test_metrics = compute_metrics(test_bt['asset_return'])

    print_metrics("Train Strategy Metrics", train_metrics)
    print_metrics("Test Strategy Metrics", test_metrics)
    print_metrics("Test Buy & Hold Metrics", bh_test_metrics)

    train_bt.to_csv("outputs/train_backtest.csv")
    test_bt.to_csv("outputs/test_backtest.csv")

    save_plot(train_bt, ticker, "outputs/train_plot.png")
    save_plot(test_bt, ticker, "outputs/test_plot.png")

    print("\nDone.")
    print("Files written to outputs/")


if __name__ == "__main__":
    main()
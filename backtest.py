import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


def download_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(f"No data downloaded for {ticker}")

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if 'Close' not in df.columns:
        raise ValueError(f"'Close' column not found. Columns are: {df.columns.tolist()}")

    df = df[['Close']].copy()
    df.columns = ['price']
    return df


def generate_mean_reversion_signals(
    df: pd.DataFrame,
    window: int = 20,
    entry_z: float = 1.5,
    exit_z: float = 0.5
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

        # Entry rules
        if prev_pos == 0:
            if z < -entry_z:
                position[i] = 1   # long
            elif z > entry_z:
                position[i] = -1  # short
            else:
                position[i] = 0

        # Exit rules
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


def run_backtest(df: pd.DataFrame, cost_per_unit: float = 0.0005) -> pd.DataFrame:
    df = df.copy()

    # shift position to avoid lookahead bias
    df['position_lagged'] = df['position'].shift(1).fillna(0)

    df['asset_return'] = df['price'].pct_change().fillna(0)
    df['strategy_gross_return'] = df['position_lagged'] * df['asset_return']

    # transaction cost based on turnover
    df['turnover'] = df['position'].diff().abs().fillna(0)
    df['transaction_cost'] = df['turnover'] * cost_per_unit

    df['strategy_net_return'] = df['strategy_gross_return'] - df['transaction_cost']

    df['cum_asset'] = (1 + df['asset_return']).cumprod()
    df['cum_strategy'] = (1 + df['strategy_net_return']).cumprod()

    return df


def compute_metrics(returns: pd.Series) -> dict:
    returns = returns.dropna()

    if len(returns) == 0:
        return {}

    annual_return = (1 + returns.mean()) ** 252 - 1
    annual_vol = returns.std() * np.sqrt(252)

    sharpe = annual_return / annual_vol if annual_vol != 0 else np.nan

    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = cumulative / rolling_max - 1
    max_drawdown = drawdown.min()

    win_rate = (returns > 0).mean()

    return {
        "Annual Return": annual_return,
        "Annual Volatility": annual_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_drawdown,
        "Win Rate": win_rate
    }


def plot_results(df: pd.DataFrame, ticker: str):
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # Price and signals
    axes[0].plot(df.index, df['price'], label='Price', color='blue')
    buy_signals = df[(df['position'] == 1) & (df['position'].shift(1) != 1)]
    sell_signals = df[(df['position'] == -1) & (df['position'].shift(1) != -1)]
    exit_signals = df[(df['position'] == 0) & (df['position'].shift(1) != 0)]

    axes[0].scatter(buy_signals.index, buy_signals['price'], marker='^', color='green', label='Long Entry')
    axes[0].scatter(sell_signals.index, sell_signals['price'], marker='v', color='red', label='Short Entry')
    axes[0].scatter(exit_signals.index, exit_signals['price'], marker='o', color='black', label='Exit', s=20)
    axes[0].set_title(f"{ticker} Price and Trading Signals")
    axes[0].legend()

    # Z-score
    axes[1].plot(df.index, df['zscore'], label='Z-score', color='purple')
    axes[1].axhline(1.5, linestyle='--', color='red')
    axes[1].axhline(-1.5, linestyle='--', color='green')
    axes[1].axhline(0, linestyle='-', color='black')
    axes[1].set_title("Z-score")

    # Cumulative returns
    axes[2].plot(df.index, df['cum_asset'], label='Buy & Hold', color='gray')
    axes[2].plot(df.index, df['cum_strategy'], label='Strategy', color='orange')
    axes[2].set_title("Cumulative Returns")
    axes[2].legend()

    plt.tight_layout()
    plt.show()


def main():
    ticker = "SPY"
    start = "2020-01-01"
    end = "2025-01-01"

    df = download_data(ticker, start, end)
    df = generate_mean_reversion_signals(df, window=20, entry_z=1.5, exit_z=0.5)
    df = run_backtest(df, cost_per_unit=0.0005)

    metrics = compute_metrics(df['strategy_net_return'])
    benchmark_metrics = compute_metrics(df['asset_return'])

    print("\nStrategy Metrics")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    print("\nBuy & Hold Metrics")
    for k, v in benchmark_metrics.items():
        print(f"{k}: {v:.4f}")

    plot_results(df, ticker)


if __name__ == "__main__":
    main()
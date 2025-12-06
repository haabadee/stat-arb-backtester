# stat-arb-backtester

A Python backtesting engine for evaluating mean reversion strategies on historical equity data. The project includes parameter tuning on a training set, out-of-sample testing, transaction cost modeling, and common risk/performance metrics.

## Features
- Historical market data download using Yahoo Finance
- Z-score based mean reversion signal generation
- Transaction cost modeling
- Train/test split for out-of-sample evaluation
- Parameter grid search
- Performance metrics including Sharpe ratio and max drawdown
- Plot generation and CSV export

## Strategy
The strategy uses a rolling mean and rolling standard deviation to compute a z-score:

$$
z_t = \frac{P_t - \mu_t}{\sigma_t}
$$

- Enter long when the z-score is sufficiently negative
- Enter short when the z-score is sufficiently positive
- Exit when the z-score reverts toward zero

Positions are lagged by one period before computing returns to avoid lookahead bias.

## Installation
```bash
pip install -r requirements.txt
python backtest.py
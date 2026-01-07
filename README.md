# stat-arb-backtester

A Python backtesting project for evaluating quantitative trading strategies on historical market data. The repository currently supports:

- single-asset mean reversion
- pairs trading / statistical arbitrage
- parameter search on a training set
- out-of-sample evaluation on a test set
- transaction cost modeling
- performance analytics and plot generation

## Strategies Implemented

### 1. Single-Asset Mean Reversion
This strategy assumes that an asset's price tends to revert toward its recent average.

For a chosen asset:
- compute a rolling mean and rolling standard deviation
- calculate the z-score of price relative to that rolling window
- enter long when price is significantly below the rolling mean
- enter short when price is significantly above the rolling mean
- exit when the z-score reverts toward zero

### 2. Pairs Trading / Statistical Arbitrage
This strategy trades the relative relationship between two historically related assets.

For a selected pair:
- estimate a hedge ratio between the two assets
- define the spread as:

$$
\text{spread}_t = P_{1,t} - \beta P_{2,t}
$$

- compute a rolling z-score of the spread
- enter a long spread position when the spread is abnormally low
- enter a short spread position when the spread is abnormally high
- exit when the spread reverts toward its recent average

The pairs framework aims to isolate relative mispricing instead of outright market direction.

---

## Key Features

- Historical price download using Yahoo Finance
- Mean reversion signal generation
- Pairs trading signal generation
- Hedge ratio estimation
- Optional rolling hedge ratio support
- Parameter grid search
- Train/test split for out-of-sample evaluation
- Transaction cost modeling
- CSV export of results
- Plot generation for price, spread, z-score, and cumulative returns
- Risk and performance metrics:
  - annual return
  - annual volatility
  - Sharpe ratio
  - max drawdown
  - Calmar ratio
  - win rate

---

## Repository Structure

```bash
quant-backtester/
  backtest.py
  pairs_backtest.py
  requirements.txt
  README.md
  outputs/
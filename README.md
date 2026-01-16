# stat-arb-backtester

A Python backtesting and research project for evaluating quantitative trading strategies on historical market data. The repository currently supports:

- single-asset mean reversion
- pairs trading / statistical arbitrage
- multi-pair screening across a candidate universe
- parameter search on a training set
- out-of-sample evaluation on a test set
- transaction cost modeling
- performance analytics and plot generation

This project is intended as a lightweight research framework for exploring mean reversion and relative value strategies, with an emphasis on avoiding lookahead bias and evaluating performance out-of-sample.

---

## Strategies Implemented

### 1. Single-Asset Mean Reversion
This strategy assumes that an asset's price tends to revert toward its recent average.

For a chosen asset:
- compute a rolling mean and rolling standard deviation
- calculate the z-score of price relative to that rolling window
- enter long when price is significantly below the rolling mean
- enter short when price is significantly above the rolling mean
- exit when the z-score reverts toward zero

This provides a simple baseline framework for testing time-series mean reversion.

### 2. Pairs Trading / Statistical Arbitrage
This strategy trades the relative relationship between two historically related assets.

For a selected pair:
- estimate a hedge ratio between the two assets
- define the spread as:


$$
\text{spread}_t = P_{1,t} - (\alpha + \beta P_{2,t})
$$


where:
- $P_{1,t}$ is the price of asset 1
- $P_{2,t}$ is the price of asset 2
- $\alpha$ is the intercept
- $\beta$ is the hedge ratio estimated by OLS

Then:
- compute a rolling z-score of the spread
- enter a long spread position when the spread is abnormally low
- enter a short spread position when the spread is abnormally high
- exit when the spread reverts toward its recent average

The pairs framework aims to isolate relative mispricing instead of outright market direction.

### 3. Multi-Pair Screening
The repository also includes a pair screening workflow that:
- downloads a universe of related tickers
- generates all possible ticker pairs
- applies a cointegration test on the training set
- estimates hedge ratios using OLS
- optionally uses rolling hedge ratios
- performs parameter search per pair on the training set
- backtests spread-based mean reversion on valid pairs
- ranks pairs by out-of-sample performance

This is a more systematic research workflow than testing a single hand-picked pair.

---

## Key Features

- Historical price download using Yahoo Finance
- Mean reversion signal generation
- Pairs trading signal generation
- OLS hedge ratio estimation
- Optional rolling hedge ratio support
- Cointegration testing for pair screening
- Parameter grid search
- Train/test split for out-of-sample evaluation
- Transaction cost modeling
- CSV export of results
- Plot generation for:
  - price series
  - spread
  - z-score
  - cumulative returns
- Risk and performance metrics:
  - annual return
  - annual volatility
  - Sharpe ratio
  - max drawdown
  - Calmar ratio
  - win rate

---

## Repository Structure

~~~bash
stat-arb-backtester/
  backtest.py
  pairs_backtest.py
  pair_screener.py
  requirements.txt
  README.md
  outputs/
~~~

### File Overview

#### `backtest.py`
Single-asset mean reversion backtester.

Implements:
- rolling mean / standard deviation
- z-score signal generation
- transaction cost modeling
- parameter search on training data
- out-of-sample evaluation on test data

#### `pairs_backtest.py`
Detailed backtester for a single selected pair.

Implements:
- hedge ratio estimation
- spread construction
- z-score trading logic
- optional rolling hedge ratio
- transaction costs
- train/test evaluation
- plot and CSV generation for a selected pair

#### `pair_screener.py`
Multi-pair research pipeline.

Implements:
- candidate universe download
- all-pairs generation
- cointegration screening
- per-pair parameter search
- out-of-sample ranking of candidate pairs
- result export for the top-ranked pair(s)

---

## Installation

Create a virtual environment if desired, then install dependencies:

~~~bash
pip install -r requirements.txt
~~~

Typical dependencies include:
- numpy
- pandas
- matplotlib
- yfinance
- statsmodels

---

## Running the Backtests

### Single-Asset Mean Reversion
~~~bash
python backtest.py
~~~

### Single-Pair Statistical Arbitrage
~~~bash
python pairs_backtest.py
~~~

### Multi-Pair Screening Workflow
~~~bash
python pair_screener.py
~~~

Generated results are typically written to the `outputs/` directory.

---

## Methodology

### Train/Test Split
Model parameters are selected on a training set and then evaluated on a separate test set.

This helps reduce overfitting and provides a more realistic estimate of out-of-sample performance.

Example:
- training period: 2020-01-01 to 2023-12-31
- test period: 2024-01-01 to 2025-01-01

### Parameter Search
The backtester supports grid search over:
- rolling window length
- entry z-score threshold
- exit z-score threshold

For pair screening, parameter search is performed separately for each pair on the training set.

### Lookahead Bias Prevention
Positions are lagged by one period before applying returns:

- signal generated on day $t$
- trade assumed to affect returns beginning on day $t+1$

This prevents the strategy from using future information when computing performance.

### Transaction Costs
The framework includes a simple transaction cost model based on turnover.

While simplified, this makes the backtests more realistic than ignoring execution costs entirely.

---

## Pairs Trading Methodology

Pairs trading depends heavily on the quality of the pair relationship.

The pairs framework may include:
- cointegration testing
- static OLS hedge ratio estimation
- optional rolling hedge ratio estimation
- spread construction from residual relationships
- z-score entry and exit thresholds
- normalized leg weighting

The strategy attempts to profit when a spread deviates significantly from its historical relationship and later reverts.

---

## Multi-Pair Screening Workflow

The screening pipeline follows these steps:

1. Select a universe of candidate tickers  
2. Download daily adjusted close prices  
3. Split data into train and test windows  
4. Generate all possible ticker pairs  
5. Run a cointegration test on the training set for each pair  
6. Discard pairs that fail the screening threshold  
7. Estimate hedge ratios for valid pairs  
8. Run per-pair parameter search on the training set  
9. Evaluate selected parameters out-of-sample on the test set  
10. Rank pairs by out-of-sample performance metrics such as Sharpe ratio  

This workflow is more robust than manually selecting a pair and visually inspecting results.

---

## Example Outputs

The project may generate files such as:

- `grid_search_results.csv`
- `train_backtest.csv`
- `test_backtest.csv`
- `pairs_train_backtest.csv`
- `pairs_test_backtest.csv`
- `pair_screen_results.csv`
- `{PAIR}_grid_results.csv`
- `{PAIR}_train_backtest.csv`
- `{PAIR}_test_backtest.csv`
- `train_plot.png`
- `test_plot.png`
- `pairs_train_plot.png`
- `pairs_test_plot.png`
- `{PAIR}_train_plot.png`
- `{PAIR}_test_plot.png`

Plots generally include:
- asset prices
- spread and z-score
- strategy cumulative returns vs benchmark

---

## Metrics

The backtester computes common performance metrics:

- **Annual Return**: annualized return estimate from daily returns
- **Annual Volatility**: annualized standard deviation of returns
- **Sharpe Ratio**: annualized return divided by annualized volatility
- **Max Drawdown**: worst peak-to-trough decline in cumulative returns
- **Calmar Ratio**: annual return divided by absolute max drawdown
- **Win Rate**: fraction of periods with positive returns

These metrics should be interpreted together rather than in isolation.

---

## Limitations

This project is intended as a research and educational framework, not production trading infrastructure.

Current limitations include:
- daily data only
- simplified transaction cost model
- no detailed slippage or execution model
- no borrow fees or financing costs for short positions
- no portfolio-level optimization
- no intraday execution logic
- no live trading connectivity
- reliance on Yahoo Finance data quality
- strategy performance may be sensitive to regime changes and parameter choices

For pairs trading specifically:
- cointegration relationships may not remain stable over time
- rolling beta estimates can be noisy
- pair selection is limited by the chosen ticker universe
- ranking by historical Sharpe does not guarantee future performance

---

## Future Improvements

Potential future work:
- automated sector-aware pair universe construction
- stronger pair stability filters
- walk-forward optimization
- portfolio-level multi-pair allocation
- volatility targeting
- more realistic slippage and execution modeling
- support for intraday data
- factor-based and cross-sectional strategies
- richer benchmarking and reporting
- notebook-based research reports

---

## Notes

This repository is designed as an iterative research project. The codebase may continue to evolve as new strategies, screening methods, and evaluation tools are added.
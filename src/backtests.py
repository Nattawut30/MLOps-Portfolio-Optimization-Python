"""
Walk-forward portfolio backtesting.

Input must be a wide Parquet file of adjusted daily close prices:
- DatetimeIndex
- one column per asset
- no missing values
- at least lookback_days + 2 observations

This module does not download data, call an API, or modify the existing
pipeline. It compares risk parity with equal weight out of sample.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS_PER_YEAR = 252
Strategy = Literal["risk_parity", "equal_weight"]


@dataclass(frozen=True)
class BacktestConfig:
    lookback_days: int = 252
    rebalance_frequency_days: int = 21
    transaction_cost_bps: float = 5.0
    annual_risk_free_rate: float = 0.045
    strategy: Strategy = "risk_parity"

    def __post_init__(self) -> None:
        if self.lookback_days < 30:
            raise ValueError("lookback_days must be at least 30.")
        if self.rebalance_frequency_days < 1:
            raise ValueError("rebalance_frequency_days must be at least 1.")
        if self.transaction_cost_bps < 0:
            raise ValueError("transaction_cost_bps cannot be negative.")


@dataclass
class BacktestResult:
    daily: pd.DataFrame
    weights: pd.DataFrame
    metrics: pd.DataFrame


def _validate_prices(prices: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise ValueError("prices must be a non-empty pandas DataFrame.")
    if prices.shape[1] < 2:
        raise ValueError("prices must contain at least two assets.")

    clean = prices.copy()
    clean.index = pd.to_datetime(clean.index, errors="raise")
    clean = clean.sort_index()
    clean = clean.apply(pd.to_numeric, errors="raise")

    if clean.index.has_duplicates:
        raise ValueError("prices contain duplicate dates.")
    if clean.isna().any().any():
        raise ValueError(
            "prices contain missing values. Use a documented common trading "
            "calendar; do not silently forward-fill a backtest."
        )
    if not np.isfinite(clean.to_numpy()).all() or (clean <= 0).any().any():
        raise ValueError("prices must be finite and strictly positive.")

    return clean


def _risk_parity_weights(window_returns: pd.DataFrame) -> pd.Series:
    covariance = window_returns.cov().to_numpy() * TRADING_DAYS_PER_YEAR
    asset_count = covariance.shape[0]

    def objective(weights: np.ndarray) -> float:
        portfolio_variance = float(weights @ covariance @ weights)
        if portfolio_variance <= 0:
            return float("inf")

        risk_contribution = weights * (covariance @ weights)
        contribution_share = risk_contribution / portfolio_variance
        target_share = 1.0 / asset_count
        return float(np.sum((contribution_share - target_share) ** 2))

    result = minimize(
        objective,
        x0=np.full(asset_count, 1.0 / asset_count),
        method="SLSQP",
        bounds=[(1e-8, 1.0)] * asset_count,
        constraints=[{"type": "eq", "fun": lambda weights: weights.sum() - 1.0}],
        options={"ftol": 1e-12, "maxiter": 1000},
    )

    if not result.success:
        raise RuntimeError(f"Risk-parity optimisation failed: {result.message}")

    return pd.Series(result.x, index=window_returns.columns)


def _target_weights(window_returns: pd.DataFrame, strategy: Strategy) -> pd.Series:
    if strategy == "risk_parity":
        return _risk_parity_weights(window_returns)

    if strategy == "equal_weight":
        return pd.Series(
            1.0 / window_returns.shape[1],
            index=window_returns.columns,
        )

    raise ValueError(f"Unsupported strategy: {strategy}")


def _metrics(
    net_returns: pd.Series,
    gross_returns: pd.Series,
    turnover: pd.Series,
    config: BacktestConfig,
) -> pd.DataFrame:
    if net_returns.empty:
        raise ValueError("No active trading days available.")

    years = len(net_returns) / TRADING_DAYS_PER_YEAR
    net_equity = (1.0 + net_returns).cumprod()
    gross_equity = (1.0 + gross_returns).cumprod()

    annualized_return = net_equity.iloc[-1] ** (1.0 / years) - 1.0
    annualized_volatility = net_returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)

    daily_risk_free_rate = (1.0 + config.annual_risk_free_rate) ** (
        1.0 / TRADING_DAYS_PER_YEAR
    ) - 1.0
    daily_std = net_returns.std(ddof=1)
    sharpe_ratio = (
        np.nan
        if daily_std == 0
        else (net_returns.mean() - daily_risk_free_rate)
        / daily_std
        * np.sqrt(TRADING_DAYS_PER_YEAR)
    )

    drawdown = net_equity / net_equity.cummax() - 1.0

    return pd.DataFrame(
        {
            "strategy": [config.strategy],
            "start": [net_returns.index.min().date().isoformat()],
            "end": [net_returns.index.max().date().isoformat()],
            "trading_days": [len(net_returns)],
            "annualized_return": [annualized_return],
            "annualized_volatility": [annualized_volatility],
            "sharpe_ratio": [sharpe_ratio],
            "max_drawdown": [float(drawdown.min())],
            "total_turnover": [float(turnover.sum())],
            "transaction_cost_bps": [config.transaction_cost_bps],
            "terminal_wealth_net": [float(net_equity.iloc[-1])],
            "transaction_cost_drag": [
                float(gross_equity.iloc[-1] - net_equity.iloc[-1])
            ],
        }
    )


def run_backtest(prices: pd.DataFrame, config: BacktestConfig) -> BacktestResult:
    """
    Strict walk-forward timing:

    A target is calculated only from returns ending yesterday and applied
    today. Holdings then drift with realised asset returns until the next
    rebalance. This is a true periodic-rebalance backtest, not cost-free
    daily rebalancing disguised as monthly rebalancing.
    """
    prices = _validate_prices(prices)
    returns = prices.pct_change(fill_method=None).iloc[1:]

    if len(returns) <= config.lookback_days:
        raise ValueError(
            f"Need more than {config.lookback_days + 1} price observations."
        )

    weights = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    turnover = pd.Series(0.0, index=returns.index, name="turnover")
    gross_returns = pd.Series(0.0, index=returns.index, name="gross_return")
    current_weights = pd.Series(0.0, index=returns.columns)

    for position, date in enumerate(returns.index):
        rebalance_today = position >= config.lookback_days and (
            (position - config.lookback_days) % config.rebalance_frequency_days == 0
        )

        if rebalance_today:
            trailing_returns = returns.iloc[position - config.lookback_days : position]
            target = _target_weights(trailing_returns, config.strategy)
            turnover.loc[date] = float((target - current_weights).abs().sum())
            current_weights = target

        held_weights = current_weights.copy()
        weights.loc[date] = held_weights
        daily_asset_returns = returns.loc[date]
        gross_returns.loc[date] = float(held_weights @ daily_asset_returns)

        if held_weights.sum() > 0:
            post_return_values = held_weights * (1.0 + daily_asset_returns)
            current_weights = post_return_values / post_return_values.sum()

    transaction_cost = turnover * config.transaction_cost_bps / 10000.0
    net_returns = gross_returns - transaction_cost
    active_days = weights.sum(axis=1) > 0

    daily = pd.DataFrame(
        {
            "gross_return": gross_returns,
            "transaction_cost": transaction_cost,
            "net_return": net_returns,
            "turnover": turnover,
            "equity_curve": (1.0 + net_returns).cumprod(),
        }
    )

    metrics = _metrics(
        net_returns.loc[active_days],
        gross_returns.loc[active_days],
        turnover.loc[active_days],
        config,
    )

    return BacktestResult(daily=daily, weights=weights, metrics=metrics)


def write_report(result: BacktestResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    result.daily.to_parquet(output_dir / "daily_results.parquet")
    result.weights.to_parquet(output_dir / "weights.parquet")
    result.metrics.to_csv(output_dir / "metrics.csv", index=False)

    metric = result.metrics.iloc[0]

    report = f"""# Walk-forward backtest report

Strategy: {metric["strategy"]}
Period: {metric["start"]} to {metric["end"]}
Trading days: {metric["trading_days"]}
Annualised return: {metric["annualized_return"]:.2%}
Annualised volatility: {metric["annualized_volatility"]:.2%}
Sharpe ratio: {metric["sharpe_ratio"]:.3f}
Maximum drawdown: {metric["max_drawdown"]:.2%}
Total turnover: {metric["total_turnover"]:.3f}x
Transaction cost assumption: {metric["transaction_cost_bps"]:.2f} bps
Net terminal wealth: {metric["terminal_wealth_net"]:.4f}

This is historical research, not investment advice. It does not include
taxes, bid-ask spreads, market impact, execution delay, or survivorship bias.
"""

    (output_dir / "report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a walk-forward portfolio backtest."
    )
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--strategy",
        choices=["risk_parity", "equal_weight", "all"],
        default="all",
    )
    parser.add_argument("--lookback-days", type=int, default=252)
    parser.add_argument("--rebalance-frequency-days", type=int, default=21)
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    parser.add_argument("--annual-risk-free-rate", type=float, default=0.045)
    args = parser.parse_args()

    prices = pd.read_parquet(args.prices)

    strategies: list[Strategy]
    if args.strategy == "all":
        strategies = ["risk_parity", "equal_weight"]
    else:
        strategies = [args.strategy]

    for strategy in strategies:
        config = BacktestConfig(
            lookback_days=args.lookback_days,
            rebalance_frequency_days=args.rebalance_frequency_days,
            transaction_cost_bps=args.transaction_cost_bps,
            annual_risk_free_rate=args.annual_risk_free_rate,
            strategy=strategy,
        )
        result = run_backtest(prices, config)
        report_dir = args.output_dir / strategy
        write_report(result, report_dir)
        print(f"Wrote {strategy} report: {report_dir}")


if __name__ == "__main__":
    main()

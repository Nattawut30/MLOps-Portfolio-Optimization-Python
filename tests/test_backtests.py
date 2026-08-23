import numpy as np
import pandas as pd
import pytest

from src.backtests import (
    BacktestConfig,
    _risk_parity_weights,
    run_backtest,
    write_report,
)


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    rng = np.random.default_rng(17)
    returns = rng.normal(
        loc=[0.0003, 0.0004, 0.0002],
        scale=[0.008, 0.016, 0.012],
        size=(360, 3),
    )
    prices = 100.0 * np.exp(np.cumsum(returns, axis=0))

    return pd.DataFrame(
        prices,
        index=pd.bdate_range("2023-01-02", periods=len(prices)),
        columns=["CALM", "VOLATILE", "MIDDLE"],
    )


def test_stays_in_cash_until_lookback_is_complete(sample_prices):
    result = run_backtest(
        sample_prices,
        BacktestConfig(
            strategy="equal_weight",
            lookback_days=60,
            rebalance_frequency_days=21,
        ),
    )

    assert (result.weights.iloc[:60].sum(axis=1) == 0.0).all()
    np.testing.assert_allclose(result.weights.iloc[60].to_numpy(), 1.0 / 3.0)


def test_initial_transaction_cost_is_charged_once():
    prices = pd.DataFrame(
        {"A": np.full(90, 100.0), "B": np.full(90, 200.0)},
        index=pd.bdate_range("2024-01-02", periods=90),
    )

    result = run_backtest(
        prices,
        BacktestConfig(
            strategy="equal_weight",
            lookback_days=30,
            rebalance_frequency_days=1,
            transaction_cost_bps=10.0,
        ),
    )

    assert result.daily["turnover"].sum() == pytest.approx(1.0)
    assert result.daily["transaction_cost"].sum() == pytest.approx(0.001)
    assert result.metrics.loc[0, "terminal_wealth_net"] == pytest.approx(0.999)


def test_risk_parity_allocates_more_to_lower_volatility_asset():
    rng = np.random.default_rng(3)
    returns = pd.DataFrame(
        {
            "CALM": rng.normal(0.0002, 0.005, 100),
            "VOLATILE": rng.normal(0.0002, 0.020, 100),
        }
    )

    weights = _risk_parity_weights(returns)

    assert weights.sum() == pytest.approx(1.0)
    assert weights["CALM"] > weights["VOLATILE"]


def test_report_writes_all_artifacts(sample_prices, tmp_path):
    result = run_backtest(
        sample_prices,
        BacktestConfig(strategy="equal_weight", lookback_days=60),
    )

    write_report(result, tmp_path)

    assert (tmp_path / "daily_results.parquet").exists()
    assert (tmp_path / "weights.parquet").exists()
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "report.md").exists()


def test_weights_drift_between_monthly_rebalances():
    dates = pd.bdate_range("2024-01-02", periods=40)
    prices = pd.DataFrame(
        {
            "A": [100.0] * 31 + [110.0] * 9,
            "B": [100.0] * 40,
        },
        index=dates,
    )

    result = run_backtest(
        prices,
        BacktestConfig(
            strategy="equal_weight",
            lookback_days=30,
            rebalance_frequency_days=10,
            transaction_cost_bps=0.0,
        ),
    )

    np.testing.assert_allclose(
        result.weights.iloc[31].to_numpy(),
        np.array([110.0 / 210.0, 100.0 / 210.0]),
    )

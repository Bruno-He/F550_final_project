from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from backtester import EventBacktester, BacktestConfig


@dataclass
class StaticDecision:
    action: str
    confidence: float
    score: float
    thesis: str


class StaticDecisionAgent:
    """
    Agent wrapper that returns precomputed decisions on event dates.
    """

    def __init__(self, decisions_by_date):
        self.decisions_by_date = decisions_by_date

    def decide(self, vin):
        if hasattr(vin, "asof"):
            dt = pd.Timestamp(vin.asof)
        elif hasattr(vin, "evaluation_date"):
            dt = pd.Timestamp(vin.evaluation_date)
        else:
            raise AttributeError("Input object must have either 'asof' or 'evaluation_date'.")

        d = self.decisions_by_date.get(dt)

        if d is None:
            return StaticDecision(
                action="hold",
                confidence=0.0,
                score=0.0,
                thesis="No decision available.",
            )

        return d


def annualized_sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Compute annualized Sharpe ratio from daily return series.
    """
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if len(r) < 2:
        return np.nan

    std = r.std(ddof=1)
    if std is None or std == 0 or np.isnan(std):
        return np.nan

    return float(np.sqrt(periods_per_year) * r.mean() / std)


def build_decision_mapping(decision_df: pd.DataFrame) -> Dict[pd.Timestamp, StaticDecision]:
    """
    Build date -> StaticDecision mapping from a decision DataFrame.
    """
    out = {}

    for _, row in decision_df.iterrows():
        dt = pd.Timestamp(row["evaluation_date"])
        out[dt] = StaticDecision(
            action=row["action"],
            confidence=float(row["confidence"]),
            score=float(row["score"]),
            thesis=str(row["thesis"]),
        )

    return out


from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class ScoreBacktestConfig:
    initial_cash: float = 100_000.0
    max_units: float = 10.0
    transaction_cost_bps: float = 10.0


def build_score_mapping(decision_df: pd.DataFrame) -> Dict[pd.Timestamp, float]:
    """
    Build date -> score mapping from a decision DataFrame.
    """
    out = {}
    for _, row in decision_df.iterrows():
        out[pd.Timestamp(row["evaluation_date"])] = float(row["score"])
    return out


def run_score_based_backtest(
    prices: pd.Series,
    score_by_date: Dict[pd.Timestamp, float],
    cfg: ScoreBacktestConfig,
    ticker: str = "NVDA",
) -> pd.DataFrame:
    """
    Run a simple score-based event backtest.

    On each event date:
    - update target position = score * max_units

    Between events:
    - hold the position constant
    - mark portfolio value to market daily

    Parameters
    ----------
    prices : pd.Series
        Daily close prices indexed by date.
    score_by_date : dict
        Mapping from event date to agent score in [-1, 1].
    cfg : ScoreBacktestConfig
        Backtest configuration.
    ticker : str
        Ticker label.

    Returns
    -------
    pd.DataFrame
        Daily backtest results.
    """
    px = prices.copy()
    px.index = pd.to_datetime(px.index)
    px = px.sort_index().astype(float)

    cash = float(cfg.initial_cash)
    position = 0.0
    prev_portfolio_value = float(cfg.initial_cash)

    rows = []

    for dt, price in px.items():
        action = "hold"
        decision_score = np.nan

        if dt in score_by_date:
            target_position = float(score_by_date[dt]) * float(cfg.max_units)
            trade_units = target_position - position

            # transaction cost on traded notional
            traded_notional = abs(trade_units) * float(price)
            cost = traded_notional * cfg.transaction_cost_bps / 10_000.0

            # if trade_units > 0, buying uses cash
            # if trade_units < 0, selling/shorting adds cash
            cash -= trade_units * float(price)
            cash -= cost
            position = target_position

            decision_score = score_by_date[dt]

            if target_position > 0:
                action = "buy"
            elif target_position < 0:
                action = "sell"
            else:
                action = "hold"

        portfolio_value = cash + position * float(price)
        daily_return = (portfolio_value / prev_portfolio_value - 1.0) if prev_portfolio_value != 0 else 0.0

        rows.append(
            {
                "date": dt,
                "ticker": ticker,
                "price": float(price),
                "action": action,
                "position": float(position),
                "cash": float(cash),
                "portfolio_value": float(portfolio_value),
                "decision_score": decision_score,
                "returns": float(daily_return),
            }
        )

        prev_portfolio_value = portfolio_value

    out = pd.DataFrame(rows).set_index("date")
    return out
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import ast
import pandas as pd
from pathlib import Path

from sec_fundamentals import ttm_from_quarters


@dataclass
class EventInputRow:
    event_label: str
    evaluation_date: pd.Timestamp
    price: float
    recent_5d_prices: List[float]
    ttm_revenue: Optional[float]
    ttm_net_income: Optional[float]
    ttm_fcf: Optional[float]
    ps: Optional[float]
    pe: Optional[float]
    p_fcf: Optional[float]
    sentiment_score_finbert: Optional[float]
    sentiment_score_lexicon: Optional[float]


def parse_recent_prices(x: Any) -> List[float]:
    """
    Parse a recent-price object into a list of floats.

    Accepts:
    - Python list
    - string representation of a list
    - pandas Series / numpy array-like

    Returns
    -------
    list[float]
    """
    if x is None:
        return []

    if isinstance(x, list):
        return [float(v) for v in x]

    if isinstance(x, str):
        x = x.strip()
        if not x:
            return []
        try:
            parsed = ast.literal_eval(x)
            if isinstance(parsed, list):
                return [float(v) for v in parsed]
        except Exception:
            return []

    try:
        return [float(v) for v in list(x)]
    except Exception:
        return []


def get_recent_5d_prices(
    prices: pd.Series,
    evaluation_date: pd.Timestamp,
) -> List[float]:
    """
    Get the last five available trading-day close prices up to and including
    the evaluation date.

    Parameters
    ----------
    prices : pd.Series
        Daily close price series indexed by date.
    evaluation_date : pd.Timestamp
        Target evaluation date.

    Returns
    -------
    list[float]
        Up to five closing prices.
    """
    s = prices.copy()
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()

    subset = s.loc[s.index <= pd.Timestamp(evaluation_date)]
    if subset.empty:
        return []

    return [float(v) for v in subset.tail(5).tolist()]


def safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """
    Safe division helper.
    """
    if a is None or b is None:
        return None
    if abs(b) < 1e-12:
        return None
    return float(a / b)


def get_point_in_time_ttm_metrics(
    quarter_table: pd.DataFrame,
    evaluation_date: pd.Timestamp,
    price: float,
) -> Dict[str, Optional[float]]:
    """
    Compute point-in-time TTM valuation metrics using the latest filing
    available on or before the evaluation date.

    Parameters
    ----------
    quarter_table : pd.DataFrame
        Quarterly fundamentals table.
    evaluation_date : pd.Timestamp
        Evaluation date.
    price : float
        Stock price at evaluation date.

    Returns
    -------
    dict
        TTM metrics and valuation multiples.
    """
    q = quarter_table.copy()
    q["filed"] = pd.to_datetime(q["filed"], errors="coerce")
    q["end"] = pd.to_datetime(q["end"], errors="coerce")

    eligible = q.loc[q["filed"] <= pd.Timestamp(evaluation_date)].copy()
    if eligible.empty:
        return {
            "ttm_revenue": None,
            "ttm_net_income": None,
            "ttm_fcf": None,
            "shares_outstanding": None,
            "market_cap": None,
            "ps": None,
            "pe": None,
            "p_fcf": None,
        }

    last_end = eligible["end"].max()
    ttm = ttm_from_quarters(q, last_end)

    shares = ttm.get("shares_outstanding")
    market_cap = None
    if shares is not None:
        market_cap = float(price) * float(shares)

    ps = safe_div(market_cap, ttm.get("ttm_revenue"))
    pe = safe_div(market_cap, ttm.get("ttm_net_income"))
    p_fcf = safe_div(market_cap, ttm.get("ttm_fcf"))

    return {
        "ttm_revenue": ttm.get("ttm_revenue"),
        "ttm_net_income": ttm.get("ttm_net_income"),
        "ttm_fcf": ttm.get("ttm_fcf"),
        "shares_outstanding": shares,
        "market_cap": market_cap,
        "ps": ps,
        "pe": pe,
        "p_fcf": p_fcf,
    }


def get_latest_sentiment_before_date(
    call_df: pd.DataFrame,
    evaluation_date: pd.Timestamp,
    score_col: str,
    ticker_col: str = "symbol",
    date_col: str = "date",
    ticker: str = "NVDA",
) -> Optional[float]:
    """
    Get the latest available call-level sentiment score for a ticker on or before
    the evaluation date.

    Parameters
    ----------
    call_df : pd.DataFrame
        Call-level sentiment table.
    evaluation_date : pd.Timestamp
        Evaluation date.
    score_col : str
        Name of sentiment score column.
    ticker_col : str
        Ticker column name.
    date_col : str
        Date column name.
    ticker : str
        Ticker to filter.

    Returns
    -------
    float or None
    """
    df = call_df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    subset = df.loc[
        (df[ticker_col].astype(str).str.upper() == ticker.upper())
        & (df[date_col] <= pd.Timestamp(evaluation_date))
    ].sort_values(date_col)

    if subset.empty:
        return None

    return float(subset.iloc[-1][score_col])


def build_nvda_event_inputs(
    prices: pd.Series,
    quarter_table: pd.DataFrame,
    call_finbert: pd.DataFrame,
    call_lex: pd.DataFrame,
    event_dates: Dict[str, str],
) -> pd.DataFrame:
    """
    Build Exercise 2 event input table for NVDA.

    Parameters
    ----------
    prices : pd.Series
        NVDA daily close prices.
    quarter_table : pd.DataFrame
        NVDA quarterly fundamentals table.
    call_finbert : pd.DataFrame
        Call-level FinBERT sentiment table.
    call_lex : pd.DataFrame
        Call-level LM lexicon sentiment table.
    event_dates : dict
        Mapping of event labels to evaluation-date strings.

    Returns
    -------
    pd.DataFrame
        Event input table.
    """
    rows = []

    price_series = prices.copy()
    price_series.index = pd.to_datetime(price_series.index)
    price_series = price_series.sort_index().astype(float)

    for event_label, dt_str in event_dates.items():
        evaluation_date = pd.Timestamp(dt_str)

        px_subset = price_series.loc[price_series.index <= evaluation_date]
        if px_subset.empty:
            continue

        price = float(px_subset.iloc[-1])
        actual_trade_date = pd.Timestamp(px_subset.index[-1])

        recent_5d = get_recent_5d_prices(price_series, actual_trade_date)
        ttm_metrics = get_point_in_time_ttm_metrics(
            quarter_table=quarter_table,
            evaluation_date=actual_trade_date,
            price=price,
        )

        sentiment_finbert = get_latest_sentiment_before_date(
            call_df=call_finbert,
            evaluation_date=actual_trade_date,
            score_col="call_finbert_score",
            ticker="NVDA",
        )

        sentiment_lex = get_latest_sentiment_before_date(
            call_df=call_lex,
            evaluation_date=actual_trade_date,
            score_col="call_lexicon_score",
            ticker="NVDA",
        )

        rows.append(
            {
                "event_label": event_label,
                "evaluation_date": actual_trade_date,
                "price": price,
                "recent_5d_prices": recent_5d,
                "ttm_revenue": ttm_metrics["ttm_revenue"],
                "ttm_net_income": ttm_metrics["ttm_net_income"],
                "ttm_fcf": ttm_metrics["ttm_fcf"],
                "ps": ttm_metrics["ps"],
                "pe": ttm_metrics["pe"],
                "p_fcf": ttm_metrics["p_fcf"],
                "sentiment_score_finbert": sentiment_finbert,
                "sentiment_score_lexicon": sentiment_lex,
            }
        )

    return pd.DataFrame(rows)


def load_local_price_series(csv_path: str | Path) -> pd.Series:
    """
    Load a local daily price CSV exported from Yahoo Finance or Investing.com.

    Supported formats:
    - Yahoo Finance: Date + Close
    - Investing.com: Date + Price

    Returns
    -------
    pd.Series
        Daily close-like price series indexed by Date.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    # Parse date
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).copy()
    df = df.sort_values("Date")

    # Use Close if available; otherwise fall back to Price
    if "Close" in df.columns:
        price_col = "Close"
    elif "Price" in df.columns:
        price_col = "Price"
    else:
        raise ValueError("Local price file must contain either 'Close' or 'Price'.")

    # Clean numeric strings if needed
    df[price_col] = (
        df[price_col]
        .astype(str)
        .str.replace(",", "", regex=False)
    )

    s = pd.Series(pd.to_numeric(df[price_col], errors="coerce").values, index=df["Date"])
    s = s.dropna()
    s.name = "Close"

    return s

def build_nvda_event_inputs_from_manual_fundamentals(
    prices: pd.Series,
    call_finbert: pd.DataFrame,
    call_lex: pd.DataFrame,
    manual_fundamentals: pd.DataFrame,
    event_dates: dict,
) -> pd.DataFrame:
    """
    Build NVDA event input table using manually supplied TTM fundamentals.

    Parameters
    ----------
    prices : pd.Series
        NVDA daily close prices.
    call_finbert : pd.DataFrame
        Call-level FinBERT sentiment table.
    call_lex : pd.DataFrame
        Call-level LM lexicon sentiment table.
    manual_fundamentals : pd.DataFrame
        DataFrame with columns:
        ['event_label', 'evaluation_date', 'ttm_revenue',
         'ttm_net_income', 'ttm_fcf', 'shares_outstanding',
         'market_cap', 'ps', 'pe', 'p_fcf']
    event_dates : dict
        Mapping of event labels to evaluation-date strings.

    Returns
    -------
    pd.DataFrame
        Event input table.
    """
    rows = []

    price_series = prices.copy()
    price_series.index = pd.to_datetime(price_series.index)
    price_series = price_series.sort_index().astype(float)

    mf = manual_fundamentals.copy()
    mf["evaluation_date"] = pd.to_datetime(mf["evaluation_date"], errors="coerce")

    for event_label, dt_str in event_dates.items():
        evaluation_date = pd.Timestamp(dt_str)

        px_subset = price_series.loc[price_series.index <= evaluation_date]
        if px_subset.empty:
            continue

        price = float(px_subset.iloc[-1])
        actual_trade_date = pd.Timestamp(px_subset.index[-1])

        recent_5d = get_recent_5d_prices(price_series, actual_trade_date)

        f_row = mf.loc[mf["event_label"] == event_label].copy()
        if f_row.empty:
            ttm_revenue = None
            ttm_net_income = None
            ttm_fcf = None
            shares_outstanding = None
            market_cap = None
            ps = None
            pe = None
            p_fcf = None
        else:
            f_row = f_row.iloc[0]
            ttm_revenue = f_row.get("ttm_revenue")
            ttm_net_income = f_row.get("ttm_net_income")
            ttm_fcf = f_row.get("ttm_fcf")
            shares_outstanding = f_row.get("shares_outstanding")
            market_cap = f_row.get("market_cap")
            ps = f_row.get("ps")
            pe = f_row.get("pe")
            p_fcf = f_row.get("p_fcf")

        sentiment_finbert = get_latest_sentiment_before_date(
            call_df=call_finbert,
            evaluation_date=actual_trade_date,
            score_col="call_finbert_score",
            ticker="NVDA",
        )

        sentiment_lex = get_latest_sentiment_before_date(
            call_df=call_lex,
            evaluation_date=actual_trade_date,
            score_col="call_lexicon_score",
            ticker="NVDA",
        )

        # Point-in-time safe fallback for missing pre-event sentiment
        if sentiment_finbert is None:
            sentiment_finbert = 0.0
        if sentiment_lex is None:
            sentiment_lex = 0.0

        rows.append(
            {
                "event_label": event_label,
                "evaluation_date": actual_trade_date,
                "price": price,
                "recent_5d_prices": recent_5d,
                "ttm_revenue": ttm_revenue,
                "ttm_net_income": ttm_net_income,
                "ttm_fcf": ttm_fcf,
                "shares_outstanding": shares_outstanding,
                "market_cap": market_cap,
                "ps": ps,
                "pe": pe,
                "p_fcf": p_fcf,
                "sentiment_score_finbert": sentiment_finbert,
                "sentiment_score_lexicon": sentiment_lex,
            }
        )

    return pd.DataFrame(rows)
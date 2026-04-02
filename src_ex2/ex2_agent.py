from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

import pandas as pd


@dataclass
class Ex2ValuationInput:
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
    sentiment_score: Optional[float] = None


@dataclass
class Ex2Decision:
    event_label: str
    evaluation_date: pd.Timestamp
    action: str
    confidence: float
    score: float
    thesis: str
    agent_type: str


class Ex2ValuationAgent:
    """
    Lightweight Exercise 2 valuation agent wrapper.

    This agent supports two modes:
    - baseline: valuation metrics + recent prices
    - sentiment: valuation metrics + recent prices + sentiment score
    """

    def __init__(self, llm_backend, agent_type: str = "baseline"):
        if agent_type not in {"baseline", "sentiment"}:
            raise ValueError("agent_type must be 'baseline' or 'sentiment'.")

        self.llm = llm_backend
        self.agent_type = agent_type

    @staticmethod
    def _format_recent_prices(prices: List[float]) -> str:
        if not prices:
            return "[]"
        return "[" + ", ".join(f"{float(x):.2f}" for x in prices) + "]"

    def build_prompt(self, vin: Ex2ValuationInput) -> str:
        base_prompt = f"""
You are a valuation-focused equity analyst.

Ticker: NVDA
Event label: {vin.event_label}
Evaluation date: {pd.Timestamp(vin.evaluation_date).date()}
Current price: {vin.price:.2f}
Recent 5 trading day prices: {self._format_recent_prices(vin.recent_5d_prices)}

Point-in-time TTM fundamentals:
- TTM Revenue: {vin.ttm_revenue}
- TTM Net Income: {vin.ttm_net_income}
- TTM Free Cash Flow: {vin.ttm_fcf}

Point-in-time valuation multiples:
- P/S: {vin.ps}
- P/E: {vin.pe}
- P/FCF: {vin.p_fcf}
""".strip()

        if self.agent_type == "sentiment":
            base_prompt += f"""

Latest available NVDA earnings-call sentiment score:
- Sentiment score: {vin.sentiment_score}
""".rstrip()

        base_prompt += """

Task:
Assess whether NVDA should be rated Buy, Sell, or Hold based on the valuation inputs above.
Be concise, numerate, and conservative.

Output exactly in the following format:
ACTION: <buy|sell|hold>
CONFIDENCE: <0 to 1>
SCORE: <-1 to 1>
THESIS: <2-4 sentences>
""".rstrip()

        return base_prompt

    @staticmethod
    def _safe_float(x: str, default: float) -> float:
        try:
            return float(x)
        except Exception:
            return default

    @staticmethod
    def _parse_response(text: str) -> dict:
        out = {
            "ACTION": "",
            "CONFIDENCE": "",
            "SCORE": "",
            "THESIS": "",
        }

        for line in str(text).splitlines():
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            k = k.strip().upper()
            if k in out:
                out[k] = v.strip()

        return out

    def decide(self, vin: Ex2ValuationInput) -> Ex2Decision:
        prompt = self.build_prompt(vin)

        resp = self.llm.chat(
            [
                {"role": "system", "content": "Be precise, conservative, and investment-focused."},
                {"role": "user", "content": prompt},
            ]
        )

        raw_text = resp.get("content", "")
        parsed = self._parse_response(raw_text)

        action = parsed["ACTION"].lower().strip()
        if action not in {"buy", "sell", "hold"}:
            action = "hold"

        confidence = min(max(self._safe_float(parsed["CONFIDENCE"], 0.5), 0.0), 1.0)
        score = min(max(self._safe_float(parsed["SCORE"], 0.0), -1.0), 1.0)
        thesis = parsed["THESIS"] or "No thesis returned."

        return Ex2Decision(
            event_label=vin.event_label,
            evaluation_date=pd.Timestamp(vin.evaluation_date),
            action=action,
            confidence=confidence,
            score=score,
            thesis=thesis,
            agent_type=self.agent_type,
        )
    

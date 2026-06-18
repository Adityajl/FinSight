"""
FinSight Market Data
---------------------
Real-time market data ingestion via Finnhub API.
Pulls: current quote, company profile, basic financials,
       recent candlestick data, and analyst recommendations.

- All raw API responses are normalized into typed dataclasses
- The trading agent never sees raw JSON — only MarketSnapshot objects
- Candlestick data is fetched via yfinance as a fallback (more history)
- We cache results per session to avoid redundant API calls
"""

import os
import time
from dataclasses import dataclass, field
from typing import Optional
import finnhub
import yfinance as yf
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


@dataclass
class QuoteData:
    ticker:         str
    current_price:  float
    open_price:     float
    high_price:     float
    low_price:      float
    prev_close:     float
    price_change:   float       # absolute change
    pct_change:     float       # % change
    timestamp:      float = field(default_factory=time.time)

    def summary(self) -> str:
        direction = "▲" if self.price_change >= 0 else "▼"
        return (
            f"{self.ticker} @ ${self.current_price:.2f} "
            f"{direction} {self.pct_change:+.2f}% "
            f"(Open: ${self.open_price:.2f}, "
            f"H: ${self.high_price:.2f}, L: ${self.low_price:.2f})"
        )


@dataclass
class CompanyProfile:
    ticker:      str
    name:        str
    industry:    str
    sector:      str
    market_cap:  float
    description: str = ""

    def summary(self) -> str:
        mc_b = self.market_cap / 1e9
        return (
            f"{self.name} ({self.ticker}) | "
            f"{self.sector} / {self.industry} | "
            f"Market Cap: ${mc_b:.1f}B"
        )


@dataclass
class MarketSnapshot:
    """
    Complete picture of a ticker at a point in time.
    This is what gets fed into the RAG pipeline and LLM.
    """
    ticker:          str
    quote:           QuoteData
    profile:         CompanyProfile
    analyst_signal:  str            # "buy", "hold", "sell"
    analyst_target:  float          # price target
    pe_ratio:        float
    eps:             float
    week_52_high:    float
    week_52_low:     float
    candlestick_df:  Optional[pd.DataFrame] = None

    def to_context_string(self) -> str:
        """
        Converts snapshot to a string that gets embedded
        into the RAG pipeline as context for the LLM.
        """
        upside = 0.0
        if self.quote.current_price > 0 and self.analyst_target > 0:
            upside = ((self.analyst_target - self.quote.current_price)
                      / self.quote.current_price * 100)

        return f"""
MARKET DATA FOR {self.ticker} — {time.strftime('%Y-%m-%d %H:%M UTC')}

Price Action:
  {self.quote.summary()}
  52-Week Range: ${self.week_52_low:.2f} — ${self.week_52_high:.2f}

Company:
  {self.profile.summary()}

Valuation:
  P/E Ratio : {self.pe_ratio:.2f}
  EPS       : ${self.eps:.2f}

Analyst Consensus:
  Rating      : {self.analyst_signal.upper()}
  Price Target: ${self.analyst_target:.2f}
  Upside      : {upside:+.1f}%
""".strip()


class MarketDataClient:
    """
    Unified client for all market data.
    Uses Finnhub for real-time data, yfinance for historical OHLCV.
    """

    def __init__(self):
        api_key = os.getenv("FINNHUB_API_KEY")
        if not api_key:
            raise ValueError("FINNHUB_API_KEY not set in .env")
        self.client = finnhub.Client(api_key=api_key)
        self._cache: dict[str, MarketSnapshot] = {}
        self._cache_ttl = 60  # seconds

    def _is_cache_valid(self, ticker: str) -> bool:
        if ticker not in self._cache:
            return False
        age = time.time() - self._cache[ticker].quote.timestamp
        return age < self._cache_ttl

    def get_quote(self, ticker: str) -> QuoteData:
        raw = self.client.quote(ticker)
        if not raw or raw.get("c", 0) == 0:
            raise ValueError(f"No quote data returned for {ticker}")

        current   = float(raw["c"])
        prev      = float(raw["pc"])
        change    = current - prev
        pct       = (change / prev * 100) if prev > 0 else 0.0

        return QuoteData(
            ticker=ticker,
            current_price=current,
            open_price=float(raw.get("o", 0)),
            high_price=float(raw.get("h", 0)),
            low_price=float(raw.get("l", 0)),
            prev_close=prev,
            price_change=change,
            pct_change=pct,
        )

    def get_profile(self, ticker: str) -> CompanyProfile:
        raw = self.client.company_profile2(symbol=ticker)
        return CompanyProfile(
            ticker=ticker,
            name=raw.get("name", ticker),
            industry=raw.get("finnhubIndustry", "Unknown"),
            sector=raw.get("ggroup", "Unknown"),
            market_cap=float(raw.get("marketCapitalization", 0)) * 1e6,
        )

    def get_analyst_recommendation(self, ticker: str) -> tuple[str, float]:
        """Returns (signal, price_target)."""
        try:
            recs = self.client.recommendation_trends(ticker)
            if recs:
                latest = recs[0]
                buy    = latest.get("buy", 0) + latest.get("strongBuy", 0)
                sell   = latest.get("sell", 0) + latest.get("strongSell", 0)
                hold   = latest.get("hold", 0)
                total  = buy + sell + hold
                if total == 0:
                    signal = "hold"
                elif buy / total > 0.5:
                    signal = "buy"
                elif sell / total > 0.5:
                    signal = "sell"
                else:
                    signal = "hold"
            else:
                signal = "hold"
        except Exception:
            signal = "hold"

        try:
            targets = self.client.price_target(ticker)
            target  = float(targets.get("targetMean", 0))
        except Exception:
            target = 0.0

        return signal, target

    def get_basic_financials(self, ticker: str) -> tuple[float, float, float, float]:
        """Returns (pe_ratio, eps, week_52_high, week_52_low)."""
        try:
            fin = self.client.company_basic_financials(ticker, "all")
            m   = fin.get("metric", {})
            return (
                float(m.get("peTTM", 0) or 0),
                float(m.get("epsTTM", 0) or 0),
                float(m.get("52WeekHigh", 0) or 0),
                float(m.get("52WeekLow", 0) or 0),
            )
        except Exception:
            return 0.0, 0.0, 0.0, 0.0

    def get_candlestick(self, ticker: str, period: str = "1mo") -> pd.DataFrame:
        """Uses yfinance for OHLCV history — more reliable than Finnhub free tier."""
        try:
            df = yf.download(ticker, period=period, progress=False)
            return df
        except Exception:
            return pd.DataFrame()

    def get_snapshot(self, ticker: str) -> MarketSnapshot:
        """
        Main entry point. Returns a complete MarketSnapshot.
        Results are cached for self._cache_ttl seconds.
        """
        ticker = ticker.upper().strip()

        if self._is_cache_valid(ticker):
            return self._cache[ticker]

        quote             = self.get_quote(ticker)
        profile           = self.get_profile(ticker)
        analyst, target   = self.get_analyst_recommendation(ticker)
        pe, eps, h52, l52 = self.get_basic_financials(ticker)
        candles           = self.get_candlestick(ticker)

        snapshot = MarketSnapshot(
            ticker=ticker,
            quote=quote,
            profile=profile,
            analyst_signal=analyst,
            analyst_target=target,
            pe_ratio=pe,
            eps=eps,
            week_52_high=h52,
            week_52_low=l52,
            candlestick_df=candles,
        )

        self._cache[ticker] = snapshot
        return snapshot
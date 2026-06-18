"""
FinSight Trading Agent
-----------------------
Groq (Llama3-70b) powered reasoning core.

Pipeline per query:
  1. Fetch market snapshot (price, profile, fundamentals)
  2. Retrieve relevant news context from FAISS
  3. Build structured prompt with both data sources
  4. Call Llama3-70b via Groq with JSON output
  5. Parse response into typed TradingSignal
  6. Record in ConfidenceLedger with composite score

Interview explanation:
- Groq runs Llama3-70b on LPU hardware — sub-200ms inference
- We enforce JSON output via prompt engineering (Groq has no JSON mode)
- LLM self-reports certainty (0-1) which feeds into confidence score
- The agent never makes a decision alone — it always cites retrieved sources
"""

import os
import json
import time
from groq import Groq
from dotenv import load_dotenv

from core.market_data import MarketDataClient, MarketSnapshot
from core.news_fetcher import NewsFetcher
from core.rag_pipeline import RAGPipeline
from core.inference_engine import InferenceEngine
from core.confidence_ledger import (
    ConfidenceLedger,
    TradingSignal,
    SignalStrength,
)

load_dotenv()

SYSTEM_PROMPT = """You are FinSight, a quantitative financial analyst AI.

Your job is to analyze a stock using:
1. Real-time market data (price, fundamentals, analyst ratings)
2. Retrieved news context (provided below)

You must respond ONLY with a valid JSON object.
Do NOT include any markdown, code fences, or explanation outside the JSON.
Start your response with { and end with }

JSON format:
{
  "signal": "STRONG BUY" or "BUY" or "HOLD" or "SELL" or "STRONG SELL",
  "reasoning": "2-3 sentence explanation citing specific data points and news sources",
  "key_catalysts": ["catalyst 1", "catalyst 2", "catalyst 3"],
  "key_risks": ["risk 1", "risk 2"],
  "certainty": 0.85,
  "time_horizon": "short-term (days)" or "medium-term (weeks)" or "long-term (months)"
}

Rules:
- certainty is a float between 0.0 and 1.0
- reasoning must reference at least one specific news source or data point
- be conservative: when data conflicts, lean toward HOLD
- never fabricate data not present in the context
- respond with raw JSON only, no other text
"""


class TradingAgent:
    """
    End-to-end trading signal pipeline.
    Orchestrates: data fetch → RAG retrieval → LLM reasoning → ledger recording
    """

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in .env")

        self.llm          = Groq(api_key=api_key)
        self.market       = MarketDataClient()
        self.news         = NewsFetcher()
        self.engine       = InferenceEngine()
        self.rag          = RAGPipeline(self.engine)
        self.ledger       = ConfidenceLedger()
        self._initialized = False

    def initialize(self, tickers: list[str]) -> None:
        """
        Build or load the RAG index for the given tickers.
        Called once at startup before any queries.
        """
        if self.rag.load():
            print("[TradingAgent] Loaded existing RAG index.")
            self._initialized = True
            return

        print(f"[TradingAgent] Fetching news for: {', '.join(tickers)}")

        profiles = {}
        for ticker in tickers:
            try:
                profile = self.market.get_profile(ticker)
                profiles[ticker] = profile.name
            except Exception:
                profiles[ticker] = ticker

        docs = self.news.fetch_all(tickers, profiles)
        self.rag.build(docs)
        self._initialized = True

    def _build_prompt(
        self,
        snapshot: MarketSnapshot,
        news_context: str,
    ) -> str:
        return f"""
MARKET DATA:
{snapshot.to_context_string()}

RETRIEVED NEWS CONTEXT:
{news_context}

Based on the above market data and news context, provide your trading signal for {snapshot.ticker}.
Respond ONLY with the JSON object. No other text.
""".strip()

    def _parse_signal(self, raw: str) -> dict:
        """
        Parse Llama3 JSON response into a validated dict.
        Groq has no JSON mode so we handle edge cases manually.
        """
        raw = raw.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw   = "\n".join(lines[1:-1]).strip()

        # Find the JSON object boundaries
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start != -1 and end > start:
            raw = raw[start:end]

        data = json.loads(raw)

        # Validate required fields
        for field in ["signal", "reasoning", "certainty"]:
            if field not in data:
                raise ValueError(f"Missing field in LLM response: {field}")

        # Clamp certainty to [0, 1]
        data["certainty"] = min(max(float(data["certainty"]), 0.0), 1.0)

        return data

    def _str_to_signal_strength(self, s: str) -> SignalStrength:
        mapping = {
            "STRONG BUY":  SignalStrength.STRONG_BUY,
            "BUY":         SignalStrength.BUY,
            "HOLD":        SignalStrength.HOLD,
            "SELL":        SignalStrength.SELL,
            "STRONG SELL": SignalStrength.STRONG_SELL,
        }
        return mapping.get(s.upper().strip(), SignalStrength.HOLD)

    def analyze(self, ticker: str) -> TradingSignal:
        """
        Main entry point. Analyze a ticker and return a TradingSignal.

        Full pipeline:
          market data → RAG retrieval → Llama3 → confidence ledger
        """
        if not self._initialized:
            raise RuntimeError("Call initialize() before analyze()")

        ticker = ticker.upper().strip()
        print(f"\n[TradingAgent] Analyzing {ticker}...")

        # Step 1: Fetch market snapshot
        t0       = time.perf_counter()
        snapshot = self.market.get_snapshot(ticker)
        t_market = (time.perf_counter() - t0) * 1000
        print(f"[TradingAgent] Market data fetched in {t_market:.1f}ms")

        # Step 2: RAG retrieval
        t0    = time.perf_counter()
        query = (
            f"{ticker} {snapshot.profile.name} stock analysis "
            f"earnings revenue outlook"
        )
        news_context, retrieval_score = self.rag.retrieve_context_string(
            query=query,
            ticker=ticker,
            top_k=5,
        )
        t_rag = (time.perf_counter() - t0) * 1000
        print(f"[TradingAgent] RAG retrieval in {t_rag:.1f}ms "
              f"(top score: {retrieval_score:.3f})")

        # Step 3: Build prompt and call Llama3 via Groq
        prompt = self._build_prompt(snapshot, news_context)

        t0 = time.perf_counter()
        response = self.llm.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.1,   # very low — we need consistent JSON
            max_tokens=600,
        )
        t_llm = (time.perf_counter() - t0) * 1000
        print(f"[TradingAgent] Llama3 response in {t_llm:.1f}ms")

        raw_response = response.choices[0].message.content

        # Step 4: Parse response
        parsed = self._parse_signal(raw_response)

        signal_strength = self._str_to_signal_strength(parsed["signal"])
        llm_certainty   = parsed["certainty"]

        # Step 5: Record in confidence ledger
        trading_signal = self.ledger.record(
            ticker=ticker,
            signal=signal_strength,
            reasoning=parsed.get("reasoning", ""),
            retrieved_context=news_context,
            retrieval_score=retrieval_score,
            llm_certainty=llm_certainty,
        )

        # Attach extra parsed fields for UI display
        trading_signal.key_catalysts  = parsed.get("key_catalysts", [])
        trading_signal.key_risks      = parsed.get("key_risks", [])
        trading_signal.time_horizon   = parsed.get("time_horizon", "medium-term")
        trading_signal.latency_market = t_market
        trading_signal.latency_rag    = t_rag
        trading_signal.latency_llm    = t_llm
        trading_signal.latency_total  = t_market + t_rag + t_llm

        print(f"[TradingAgent] Signal: {trading_signal.signal.value} | "
              f"Confidence: {trading_signal.confidence_score:.3f} "
              f"({trading_signal.confidence_tier.value}) | "
              f"Position: {trading_signal.position_size_pct * 100:.0f}%")

        return trading_signal

    def refresh_index(self, tickers: list[str]) -> None:
        """Force rebuild the RAG index with fresh news."""
        import shutil
        from core.rag_pipeline import INDEX_DIR
        if INDEX_DIR.exists():
            shutil.rmtree(INDEX_DIR)
        self._initialized = False
        self.initialize(tickers)

    def get_ledger_stats(self) -> dict:
        return self.ledger.get_stats()

    def get_ledger(self) -> list[dict]:
        return self.ledger.get_ledger()
"""
FinSight Confidence Ledger
--------------------------
Adapted from CLARA's Epistemic Confidence Head.
Tracks confidence scores per trading signal and gates
position sizing based on epistemic certainty.

Interview explanation:
- Every trading signal gets a confidence score (0.0 - 1.0)
- Score is derived from: RAG retrieval quality + LLM self-assessment
- Confidence gates position size: high confidence = full position,
  low confidence = reduced or no position
- All signals are logged to the ledger for auditability
"""

import time
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class SignalStrength(Enum):
    STRONG_BUY  = "STRONG BUY"
    BUY         = "BUY"
    HOLD        = "HOLD"
    SELL        = "SELL"
    STRONG_SELL = "STRONG SELL"


class ConfidenceTier(Enum):
    """
    Three-tier gating system from CLARA.
    HIGH   → full position, auto-execute
    MEDIUM → reduced position, flag for review
    LOW    → no position, human queue
    """
    HIGH   = "HIGH"
    MEDIUM = "MEDIUM"
    LOW    = "LOW"


@dataclass
class TradingSignal:
    ticker:            str
    signal:            SignalStrength
    confidence_score:  float          # 0.0 - 1.0
    confidence_tier:   ConfidenceTier
    reasoning:         str
    retrieved_context: str
    position_size_pct: float          # % of portfolio to allocate
    timestamp:         float = field(default_factory=time.time)
    retrieval_score:   float = 0.0    # FAISS similarity score
    llm_certainty:     float = 0.0    # LLM self-reported certainty

    def to_dict(self) -> dict:
        return {
            "ticker":            self.ticker,
            "signal":            self.signal.value,
            "confidence_score":  round(self.confidence_score, 4),
            "confidence_tier":   self.confidence_tier.value,
            "reasoning":         self.reasoning,
            "position_size_pct": round(self.position_size_pct, 2),
            "timestamp":         self.timestamp,
            "retrieval_score":   round(self.retrieval_score, 4),
            "llm_certainty":     round(self.llm_certainty, 4),
        }


class ConfidenceLedger:
    """
    Central ledger that scores, gates, and logs all trading signals.

    Confidence score = weighted combination of:
      - retrieval_score : how relevant the RAG context was (0-1)
      - llm_certainty   : how certain the LLM says it is (0-1)
      - signal_strength : penalty for extreme signals (need more evidence)
    """

    # Thresholds (tunable — good interview talking point)
    HIGH_THRESHOLD   = 0.72
    MEDIUM_THRESHOLD = 0.45

    # Position sizing by tier
    POSITION_SIZE = {
        ConfidenceTier.HIGH:   1.0,   # 100% of intended allocation
        ConfidenceTier.MEDIUM: 0.5,   # 50% of intended allocation
        ConfidenceTier.LOW:    0.0,   # No position
    }

    # Weights for composite score
    RETRIEVAL_WEIGHT  = 0.40
    LLM_WEIGHT        = 0.45
    STRENGTH_WEIGHT   = 0.15

    def __init__(self):
        self.ledger: list[TradingSignal] = []

    def _signal_strength_score(self, signal: SignalStrength) -> float:
        """
        Extreme signals (STRONG BUY/SELL) require more evidence,
        so they get a slight confidence penalty unless other scores
        are very high. HOLD is neutral.
        """
        return {
            SignalStrength.STRONG_BUY:  0.7,
            SignalStrength.BUY:         0.85,
            SignalStrength.HOLD:        1.0,
            SignalStrength.SELL:        0.85,
            SignalStrength.STRONG_SELL: 0.7,
        }[signal]

    def _compute_confidence(
        self,
        retrieval_score: float,
        llm_certainty: float,
        signal: SignalStrength,
    ) -> float:
        """
        Weighted composite confidence score.
        All inputs normalized to [0, 1].
        """
        strength_score = self._signal_strength_score(signal)
        score = (
            self.RETRIEVAL_WEIGHT * retrieval_score +
            self.LLM_WEIGHT       * llm_certainty   +
            self.STRENGTH_WEIGHT  * strength_score
        )
        return min(max(score, 0.0), 1.0)  # clamp to [0,1]

    def _get_tier(self, score: float) -> ConfidenceTier:
        if score >= self.HIGH_THRESHOLD:
            return ConfidenceTier.HIGH
        elif score >= self.MEDIUM_THRESHOLD:
            return ConfidenceTier.MEDIUM
        else:
            return ConfidenceTier.LOW

    def record(
        self,
        ticker:            str,
        signal:            SignalStrength,
        reasoning:         str,
        retrieved_context: str,
        retrieval_score:   float,
        llm_certainty:     float,
    ) -> TradingSignal:
        """
        Score a signal, assign a tier, compute position size,
        and log it to the ledger.
        """
        confidence_score = self._compute_confidence(
            retrieval_score, llm_certainty, signal
        )
        tier          = self._get_tier(confidence_score)
        position_size = self.POSITION_SIZE[tier]

        trading_signal = TradingSignal(
            ticker=ticker,
            signal=signal,
            confidence_score=confidence_score,
            confidence_tier=tier,
            reasoning=reasoning,
            retrieved_context=retrieved_context,
            position_size_pct=position_size,
            retrieval_score=retrieval_score,
            llm_certainty=llm_certainty,
        )

        self.ledger.append(trading_signal)
        return trading_signal

    def get_ledger(self) -> list[dict]:
        return [s.to_dict() for s in self.ledger]

    def get_stats(self) -> dict:
        if not self.ledger:
            return {}
        scores = [s.confidence_score for s in self.ledger]
        tiers  = [s.confidence_tier.value for s in self.ledger]
        return {
            "total_signals":  len(self.ledger),
            "avg_confidence": round(sum(scores) / len(scores), 4),
            "high_count":     tiers.count("HIGH"),
            "medium_count":   tiers.count("MEDIUM"),
            "low_count":      tiers.count("LOW"),
        }
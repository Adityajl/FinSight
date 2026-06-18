"""
FinSight Latency Benchmark
---------------------------
Two benchmarks in one file:

1. INFERENCE BENCHMARK
   CPU (HuggingFace baseline) vs CUDA optimized (torch.compile + autocast)
   Measures: P50, P99, mean latency for embedding workload

2. PIPELINE BENCHMARK
   End-to-end latency breakdown per component:
   Market fetch | RAG retrieval | LLM call | Total

This is the file that produces the numbers for your LinkedIn post
and NVIDIA recruiter conversations.

Run directly:
  python -m benchmarks.latency_benchmark
"""

import time
import statistics
import json
from pathlib import Path

from core.inference_engine import InferenceEngine
from core.trading_agent import TradingAgent


# ── Config ────────────────────────────────────────────────────────────────────

BENCHMARK_TICKERS   = ["AAPL", "NVDA", "MSFT"]
PIPELINE_RUNS       = 3     # full pipeline runs per ticker (costly — uses API)
INFERENCE_RUNS      = 20    # embedding-only runs (free)
RESULTS_FILE        = Path("benchmark_results.json")

SAMPLE_TEXTS = [
    "Apple reports record quarterly revenue beating Wall Street estimates",
    "NVIDIA data center revenue surges driven by AI chip demand",
    "Federal Reserve signals pause in rate hikes amid cooling inflation",
    "Microsoft Azure cloud revenue grows 29 percent year over year",
    "Tesla misses delivery estimates as EV competition intensifies",
    "Amazon AWS operating income rises sharply in Q3 earnings report",
    "Meta AI investments accelerate despite rising capital expenditure",
    "Goldman Sachs upgrades semiconductor sector on AI tailwind thesis",
]


# ── Inference Benchmark ───────────────────────────────────────────────────────

def run_inference_benchmark() -> dict:
    """
    Benchmark embedding latency: CPU baseline vs CUDA optimized.
    This is pure inference — no API calls, fully reproducible.
    """
    print("\n" + "="*60)
    print("  BENCHMARK 1: INFERENCE LATENCY (CPU vs CUDA)")
    print("="*60)

    engine = InferenceEngine()
    results = engine.benchmark(SAMPLE_TEXTS)

    print(f"\n  Device        : {results['device'].upper()}")
    print(f"  CUDA Available: {results['cuda_available']}")
    print(f"  torch.compile : {results['torch_compiled']}")
    print(f"\n  {'Metric':<20} {'Baseline (CPU)':<20} {'Optimized':<20}")
    print(f"  {'-'*58}")

    for metric in ["p50_ms", "p99_ms", "mean_ms", "min_ms", "max_ms"]:
        label = metric.replace("_ms", "").upper()
        b_val = results["baseline"][metric]
        o_val = results["optimized"][metric]
        print(f"  {label:<20} {b_val:<20} {o_val:<20}")

    print(f"  {'-'*58}")
    print(f"  {'P50 Speedup':<20} {results['speedup_p50']}x")
    print("="*60)

    return results


# ── Pipeline Benchmark ────────────────────────────────────────────────────────

def run_pipeline_benchmark(agent: TradingAgent) -> dict:
    """
    Benchmark end-to-end pipeline latency per component.
    Runs PIPELINE_RUNS times per ticker and averages results.
    Note: LLM latency includes network round-trip to OpenAI.
    """
    print("\n" + "="*60)
    print("  BENCHMARK 2: PIPELINE LATENCY BREAKDOWN")
    print("="*60)

    all_results = {}

    for ticker in BENCHMARK_TICKERS:
        print(f"\n  Running {PIPELINE_RUNS} analyses for {ticker}...")
        runs = []

        for i in range(PIPELINE_RUNS):
            try:
                # Clear market cache to get real fetch latency
                agent.market._cache.clear()

                signal = agent.analyze(ticker)

                runs.append({
                    "market_ms": signal.latency_market,
                    "rag_ms":    signal.latency_rag,
                    "llm_ms":    signal.latency_llm,
                    "total_ms":  signal.latency_total,
                    "signal":    signal.signal.value,
                    "confidence": signal.confidence_score,
                })

                time.sleep(1)  # avoid rate limits

            except Exception as e:
                print(f"  [!] Run {i+1} failed: {e}")
                continue

        if not runs:
            continue

        # Compute stats across runs
        ticker_stats = {}
        for component in ["market_ms", "rag_ms", "llm_ms", "total_ms"]:
            vals = [r[component] for r in runs]
            ticker_stats[component] = {
                "p50_ms":  round(statistics.median(vals), 1),
                "mean_ms": round(statistics.mean(vals), 1),
                "min_ms":  round(min(vals), 1),
                "max_ms":  round(max(vals), 1),
            }

        ticker_stats["runs"]             = len(runs)
        ticker_stats["last_signal"]      = runs[-1]["signal"]
        ticker_stats["last_confidence"]  = round(runs[-1]["confidence"], 3)
        all_results[ticker]              = ticker_stats

        # Print ticker summary
        print(f"\n  {ticker} — {len(runs)} runs")
        print(f"  {'Component':<20} {'P50 (ms)':<15} {'Mean (ms)':<15}")
        print(f"  {'-'*48}")
        for comp, label in [
            ("market_ms", "Market Fetch"),
            ("rag_ms",    "RAG Retrieval"),
            ("llm_ms",    "GPT-4o Call"),
            ("total_ms",  "TOTAL"),
        ]:
            s = ticker_stats[comp]
            print(f"  {label:<20} {s['p50_ms']:<15} {s['mean_ms']:<15}")

        print(f"  Last Signal    : {ticker_stats['last_signal']}")
        print(f"  Last Confidence: {ticker_stats['last_confidence']}")

    print("\n" + "="*60)
    return all_results


# ── Summary Report ────────────────────────────────────────────────────────────

def print_summary(inference: dict, pipeline: dict) -> None:
    print("\n" + "="*60)
    print("  FINSIGHT BENCHMARK SUMMARY")
    print("="*60)

    # Inference summary
    print(f"\n  INFERENCE (Embedding {len(SAMPLE_TEXTS)} texts)")
    print(f"  Baseline P50 : {inference['baseline']['p50_ms']} ms")
    print(f"  Optimized P50: {inference['optimized']['p50_ms']} ms")
    print(f"  Speedup      : {inference['speedup_p50']}x")

    # Pipeline summary
    if pipeline:
        print(f"\n  PIPELINE (End-to-end per ticker)")
        totals = [
            pipeline[t]["total_ms"]["p50_ms"]
            for t in pipeline
            if "total_ms" in pipeline[t]
        ]
        if totals:
            avg_total = round(sum(totals) / len(totals), 1)
            print(f"  Avg Total P50: {avg_total} ms")
            print(f"  Breakdown (avg across tickers):")

            for comp, label in [
                ("market_ms", "  Market Fetch "),
                ("rag_ms",    "  RAG Retrieval"),
                ("llm_ms",    "  GPT-4o Call  "),
            ]:
                vals = [
                    pipeline[t][comp]["p50_ms"]
                    for t in pipeline
                    if comp in pipeline[t]
                ]
                if vals:
                    avg = round(sum(vals) / len(vals), 1)
                    pct = round(avg / avg_total * 100, 1) if avg_total else 0
                    print(f"  {label}: {avg} ms ({pct}%)")

    print("\n  → Share these numbers in your LinkedIn post and interviews.")
    print("="*60 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n  FinSight Latency Benchmark")
    print("  " + "-"*30)

    # Benchmark 1: Pure inference (no API needed)
    inference_results = run_inference_benchmark()

    # Benchmark 2: Full pipeline (uses OpenAI + Finnhub + NewsAPI)
    print("\n  Initializing trading agent for pipeline benchmark...")
    agent = TradingAgent()
    agent.initialize(BENCHMARK_TICKERS)

    pipeline_results = run_pipeline_benchmark(agent)

    # Summary
    print_summary(inference_results, pipeline_results)

    # Save results
    all_results = {
        "inference": inference_results,
        "pipeline":  pipeline_results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M UTC"),
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"  Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
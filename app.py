"""
FinSight — Streamlit App
-------------------------
Main UI. Four tabs:
  1. Analyze    — enter a ticker, get a trading signal
  2. Benchmark  — run inference benchmark, see CPU vs CUDA
  3. Ledger     — full history of all signals this session
  4. About      — architecture explanation for demos/interviews
"""

import time
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from core.trading_agent import TradingAgent
from core.confidence_ledger import ConfidenceTier
from core.inference_engine import InferenceEngine

# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="FinSight",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .signal-strong-buy  { color: #00ff88; font-size: 2rem; font-weight: 800; }
    .signal-buy         { color: #00cc66; font-size: 2rem; font-weight: 800; }
    .signal-hold        { color: #ffcc00; font-size: 2rem; font-weight: 800; }
    .signal-sell        { color: #ff6644; font-size: 2rem; font-weight: 800; }
    .signal-strong-sell { color: #ff2222; font-size: 2rem; font-weight: 800; }
    .tier-high   { background: #003322; border-left: 4px solid #00ff88;
                   padding: 8px 12px; border-radius: 4px; }
    .tier-medium { background: #332200; border-left: 4px solid #ffcc00;
                   padding: 8px 12px; border-radius: 4px; }
    .tier-low    { background: #330011; border-left: 4px solid #ff2222;
                   padding: 8px 12px; border-radius: 4px; }
    .metric-card { background: #1a1d27; padding: 16px; border-radius: 8px;
                   border: 1px solid #2a2d3a; }
    .source-card { background: #1a1d27; padding: 12px; border-radius: 6px;
                   border: 1px solid #2a2d3a; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_agent():
    """Load agent once and cache across reruns."""
    agent = TradingAgent()
    return agent

def get_signal_css_class(signal: str) -> str:
    mapping = {
        "STRONG BUY":  "signal-strong-buy",
        "BUY":         "signal-buy",
        "HOLD":        "signal-hold",
        "SELL":        "signal-sell",
        "STRONG SELL": "signal-strong-sell",
    }
    return mapping.get(signal.upper(), "signal-hold")

def get_tier_css_class(tier: str) -> str:
    return {
        "HIGH":   "tier-high",
        "MEDIUM": "tier-medium",
        "LOW":    "tier-low",
    }.get(tier.upper(), "tier-medium")

def signal_color(signal: str) -> str:
    colors = {
        "STRONG BUY":  "#00ff88",
        "BUY":         "#00cc66",
        "HOLD":        "#ffcc00",
        "SELL":        "#ff6644",
        "STRONG SELL": "#ff2222",
    }
    return colors.get(signal.upper(), "#ffcc00")

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📈 FinSight")
    st.markdown("*CUDA-Accelerated Financial RAG Agent*")
    st.divider()

    st.markdown("### Watchlist")
    default_tickers = ["AAPL", "NVDA", "MSFT", "GOOGL", "TSLA"]
    watchlist = st.multiselect(
        "Pre-load tickers",
        options=["AAPL", "NVDA", "MSFT", "GOOGL", "TSLA",
                 "AMZN", "META", "NFLX", "AMD", "INTC"],
        default=default_tickers[:3],
    )

    if st.button("🔄 Initialize / Refresh Index", use_container_width=True):
        with st.spinner("Fetching news and building FAISS index..."):
            agent = load_agent()
            agent.refresh_index(watchlist)
        st.success(f"Index ready for {', '.join(watchlist)}")

    st.divider()
    st.markdown("### About")
    st.markdown("""
    **FinSight** combines:
    - 🔵 GPT-4o reasoning
    - 🟢 FAISS vector retrieval
    - 🟡 CUDA inference optimization
    - 🔴 CLARA confidence gating
    """)
    st.divider()
    st.caption("Built by Aditya | CLARA Architecture")

# ── Main Tabs ─────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Analyze",
    "⚡ Benchmark",
    "📋 Ledger",
    "🏗️ Architecture",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ANALYZE
# ═══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown("## Stock Analysis")
    st.markdown("Enter a ticker to get a RAG-grounded, confidence-gated trading signal.")

    col1, col2 = st.columns([3, 1])
    with col1:
        ticker_input = st.text_input(
            "Ticker Symbol",
            placeholder="e.g. AAPL, NVDA, MSFT",
            label_visibility="collapsed",
        ).upper().strip()
    with col2:
        analyze_btn = st.button("Analyze →", use_container_width=True, type="primary")

    if analyze_btn and ticker_input:
        agent = load_agent()

        # Auto-initialize if not done
        if not agent._initialized:
            with st.spinner("Building RAG index... (first run takes ~30s)"):
                agent.initialize(watchlist + [ticker_input])

        with st.spinner(f"Analyzing {ticker_input}..."):
            try:
                signal = agent.analyze(ticker_input)
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.stop()

        # ── Signal Header ──────────────────────────────────────────────────
        st.divider()
        css_class  = get_signal_css_class(signal.signal.value)
        tier_class = get_tier_css_class(signal.confidence_tier.value)

        col_sig, col_conf, col_pos = st.columns(3)

        with col_sig:
            st.markdown(f"""
            <div class="metric-card">
                <div style="color:#888; font-size:0.8rem;">SIGNAL</div>
                <div class="{css_class}">{signal.signal.value}</div>
            </div>
            """, unsafe_allow_html=True)

        with col_conf:
            tier_color = {"HIGH": "#00ff88", "MEDIUM": "#ffcc00", "LOW": "#ff2222"}
            tc = tier_color.get(signal.confidence_tier.value, "#ffcc00")
            st.markdown(f"""
            <div class="metric-card">
                <div style="color:#888; font-size:0.8rem;">CONFIDENCE</div>
                <div style="color:{tc}; font-size:2rem; font-weight:800;">
                    {signal.confidence_score:.0%}
                </div>
                <div style="color:{tc}; font-size:0.85rem;">{signal.confidence_tier.value} TIER</div>
            </div>
            """, unsafe_allow_html=True)

        with col_pos:
            pos_pct = signal.position_size_pct * 100
            pos_color = "#00ff88" if pos_pct > 50 else "#ffcc00" if pos_pct > 0 else "#ff2222"
            st.markdown(f"""
            <div class="metric-card">
                <div style="color:#888; font-size:0.8rem;">POSITION SIZE</div>
                <div style="color:{pos_color}; font-size:2rem; font-weight:800;">
                    {pos_pct:.0f}%
                </div>
                <div style="color:#888; font-size:0.85rem;">of intended allocation</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Confidence Gauge ───────────────────────────────────────────────
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=signal.confidence_score * 100,
            title={"text": "Confidence Score", "font": {"color": "white"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "white"},
                "bar":  {"color": signal_color(signal.signal.value)},
                "bgcolor": "#1a1d27",
                "steps": [
                    {"range": [0,  45], "color": "#330011"},
                    {"range": [45, 72], "color": "#332200"},
                    {"range": [72, 100],"color": "#003322"},
                ],
                "threshold": {
                    "line":  {"color": "white", "width": 2},
                    "thickness": 0.75,
                    "value": 72,
                },
            },
            number={"suffix": "%", "font": {"color": "white"}},
        ))
        fig_gauge.update_layout(
            height=250,
            paper_bgcolor="#0e1117",
            font_color="white",
            margin=dict(t=40, b=10, l=30, r=30),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # ── Reasoning + Catalysts / Risks ──────────────────────────────────
        col_r, col_cr = st.columns(2)

        with col_r:
            st.markdown("#### 🧠 Reasoning")
            st.info(signal.reasoning)

            horizon = getattr(signal, "time_horizon", "medium-term")
            st.markdown(f"**Time Horizon:** `{horizon}`")

        with col_cr:
            catalysts = getattr(signal, "key_catalysts", [])
            risks     = getattr(signal, "key_risks", [])

            if catalysts:
                st.markdown("#### 🚀 Key Catalysts")
                for c in catalysts:
                    st.markdown(f"✅ {c}")

            if risks:
                st.markdown("#### ⚠️ Key Risks")
                for r in risks:
                    st.markdown(f"🔴 {r}")

        # ── Component Score Breakdown ──────────────────────────────────────
        st.markdown("#### 🔬 Confidence Breakdown (CLARA Ledger)")
        col_s1, col_s2, col_s3 = st.columns(3)
        col_s1.metric("Retrieval Score",  f"{signal.retrieval_score:.3f}",  "FAISS similarity")
        col_s2.metric("LLM Certainty",    f"{signal.llm_certainty:.3f}",    "GPT-4o self-report")
        col_s3.metric("Composite Score",  f"{signal.confidence_score:.3f}", "Weighted final")

        # ── Latency Breakdown ──────────────────────────────────────────────
        st.markdown("#### ⚡ Latency Breakdown")
        lat_data = {
            "Component": ["Market Fetch", "RAG Retrieval", "GPT-4o Call"],
            "Latency (ms)": [
                round(signal.latency_market, 1),
                round(signal.latency_rag, 1),
                round(signal.latency_llm, 1),
            ],
        }
        fig_lat = px.bar(
            lat_data,
            x="Component",
            y="Latency (ms)",
            color="Component",
            color_discrete_sequence=["#00aaff", "#00ff88", "#ffcc00"],
            text="Latency (ms)",
        )
        fig_lat.update_layout(
            paper_bgcolor="#0e1117",
            plot_bgcolor="#1a1d27",
            font_color="white",
            showlegend=False,
            height=280,
            margin=dict(t=20, b=20),
        )
        fig_lat.update_traces(textposition="outside")
        st.plotly_chart(fig_lat, use_container_width=True)

        total_ms = getattr(signal, "latency_total", 0)
        st.caption(f"Total end-to-end latency: **{total_ms:.0f} ms**")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("## ⚡ Inference Benchmark")
    st.markdown("Compares CPU baseline vs CUDA-optimized embedding path.")
    st.markdown("*This is the number you show NVIDIA recruiters.*")

    if st.button("▶ Run Inference Benchmark", type="primary"):
        with st.spinner("Warming up and benchmarking... (~30 seconds)"):
            engine  = InferenceEngine()
            results = engine.benchmark()

        # Results table
        baseline  = results["baseline"]
        optimized = results["optimized"]

        st.markdown("### Results")

        col_b, col_o = st.columns(2)
        with col_b:
            st.markdown("**Baseline (CPU)**")
            for k, v in baseline.items():
                st.metric(k.replace("_", " ").upper(), f"{v} ms")

        with col_o:
            label = "Optimized (CUDA)" if results["cuda_available"] else "Optimized (CPU compiled)"
            st.markdown(f"**{label}**")
            for k, v in optimized.items():
                delta = round(baseline[k] - v, 2)
                st.metric(k.replace("_", " ").upper(), f"{v} ms", f"-{delta} ms")

        # Speedup callout
        speedup = results["speedup_p50"]
        color   = "#00ff88" if speedup > 2 else "#ffcc00"
        st.markdown(f"""
        <div style="background:#1a1d27; padding:20px; border-radius:8px;
                    border:2px solid {color}; text-align:center; margin:16px 0;">
            <div style="color:#888;">P50 Speedup</div>
            <div style="color:{color}; font-size:3rem; font-weight:800;">{speedup}x</div>
            <div style="color:#888;">CUDA optimized vs CPU baseline</div>
        </div>
        """, unsafe_allow_html=True)

        # Bar chart comparison
        metrics   = ["p50_ms", "p99_ms", "mean_ms"]
        fig_bench = go.Figure()
        fig_bench.add_trace(go.Bar(
            name="Baseline (CPU)",
            x=[m.replace("_ms","").upper() for m in metrics],
            y=[baseline[m] for m in metrics],
            marker_color="#ff6644",
            text=[f"{baseline[m]}ms" for m in metrics],
            textposition="outside",
        ))
        fig_bench.add_trace(go.Bar(
            name="Optimized",
            x=[m.replace("_ms","").upper() for m in metrics],
            y=[optimized[m] for m in metrics],
            marker_color="#00ff88",
            text=[f"{optimized[m]}ms" for m in metrics],
            textposition="outside",
        ))
        fig_bench.update_layout(
            barmode="group",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#1a1d27",
            font_color="white",
            height=350,
            title="Embedding Latency: CPU vs CUDA",
            margin=dict(t=50, b=20),
        )
        st.plotly_chart(fig_bench, use_container_width=True)

        st.info(
            f"Device: **{results['device'].upper()}** | "
            f"torch.compile: **{results['torch_compiled']}** | "
            f"CUDA: **{results['cuda_available']}**"
        )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — LEDGER
# ═══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("## 📋 Signal Ledger")
    st.markdown("Full history of all signals generated this session.")

    agent = load_agent()
    ledger_data = agent.get_ledger()

    if not ledger_data:
        st.info("No signals yet. Run an analysis in the Analyze tab.")
    else:
        stats = agent.get_ledger_stats()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Signals",   stats.get("total_signals", 0))
        col2.metric("Avg Confidence",  f"{stats.get('avg_confidence', 0):.1%}")
        col3.metric("High Tier",       stats.get("high_count", 0))
        col4.metric("Low Tier (held)", stats.get("low_count", 0))

        df = pd.DataFrame(ledger_data)
        df = df[[
            "ticker", "signal", "confidence_score",
            "confidence_tier", "position_size_pct",
            "retrieval_score", "llm_certainty",
        ]]
        df.columns = [
            "Ticker", "Signal", "Confidence",
            "Tier", "Position %", "Retrieval", "LLM Certainty",
        ]
        df["Position %"]  = df["Position %"].apply(lambda x: f"{x*100:.0f}%")
        df["Confidence"]  = df["Confidence"].apply(lambda x: f"{x:.1%}")
        df["Retrieval"]   = df["Retrieval"].apply(lambda x: f"{x:.3f}")
        df["LLM Certainty"] = df["LLM Certainty"].apply(lambda x: f"{x:.3f}")

        st.dataframe(df, use_container_width=True)

        # Tier distribution pie
        tier_counts = {
            "HIGH":   stats.get("high_count", 0),
            "MEDIUM": stats.get("medium_count", 0),
            "LOW":    stats.get("low_count", 0),
        }
        fig_pie = px.pie(
            names=list(tier_counts.keys()),
            values=list(tier_counts.values()),
            color=list(tier_counts.keys()),
            color_discrete_map={
                "HIGH":   "#00ff88",
                "MEDIUM": "#ffcc00",
                "LOW":    "#ff2222",
            },
            title="Confidence Tier Distribution",
        )
        fig_pie.update_layout(
            paper_bgcolor="#0e1117",
            font_color="white",
            height=300,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown("## 🏗️ Architecture")

    st.markdown("""
    ### FinSight Pipeline

    ```
    User Query (ticker)
         │
         ▼
    ┌─────────────────┐
    │  Market Data    │  ← Finnhub API (real-time quote, financials)
    │  (market_data)  │
    └────────┬────────┘
             │ MarketSnapshot
             ▼
    ┌─────────────────┐     ┌──────────────────┐
    │  News Fetcher   │────▶│  RAG Pipeline    │
    │  (NewsAPI)      │     │  (FAISS + CUDA)  │
    └─────────────────┘     └────────┬─────────┘
                                     │ top-k docs + retrieval_score
                                     ▼
                            ┌─────────────────┐
                            │  Trading Agent  │  ← GPT-4o (JSON mode)
                            │  (GPT-4o)       │
                            └────────┬────────┘
                                     │ signal + llm_certainty
                                     ▼
                            ┌─────────────────────┐
                            │  Confidence Ledger  │  ← CLARA Architecture
                            │  (CLARA-derived)    │
                            └────────┬────────────┘
                                     │ TradingSignal (typed)
                                     ▼
                            ┌─────────────────┐
                            │  Streamlit UI   │
                            └─────────────────┘
    ```

    ### Confidence Gating (CLARA)

    | Score Range | Tier   | Position Size | Action          |
    |-------------|--------|---------------|-----------------|
    | 0.72 – 1.00 | HIGH   | 100%          | Full allocation |
    | 0.45 – 0.72 | MEDIUM | 50%           | Reduced size    |
    | 0.00 – 0.45 | LOW    | 0%            | Human queue     |

    ### Confidence Score Formula
    ```
    score = 0.40 × retrieval_score   (FAISS cosine similarity)
          + 0.45 × llm_certainty     (GPT-4o self-report)
          + 0.15 × strength_penalty  (extreme signals penalized)
    ```

    ### Inference Optimization Stack
    - **Embedding model:** all-MiniLM-L6-v2 (384-dim, 80MB)
    - **CUDA path:** torch.compile (reduce-overhead) + autocast float16
    - **Baseline path:** CPU, plain HuggingFace SentenceTransformer
    - **Index:** FAISS IndexFlatIP (exact cosine search after L2-norm)
    """)

    st.markdown("---")
    st.markdown("""
    ### Interview Talking Points
    - **Why FAISS over a hosted vector DB?** Zero latency overhead, runs in-process,
      no network hop. For a latency-sensitive trading agent this matters.
    - **Why JSON mode for GPT-4o?** Guaranteed parseable output.
      No regex, no exception handling on malformed responses.
    - **Why confidence gating?** Reduces false positive signals.
      An unconfident BUY is worse than a HOLD — it wastes allocation.
    - **Why torch.compile reduce-overhead?** Optimized for repeated short inputs
      (embedding queries), which is exactly our workload.
    """)
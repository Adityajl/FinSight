"""
FinSight News Fetcher
----------------------
Fetches financial news articles from NewsAPI and structures
them as RAG documents for FAISS indexing.

- Each article becomes a Document with content + metadata
- Metadata (source, date, url) allows the LLM to cite sources
- We deduplicate by URL to avoid redundant embeddings
- Text is cleaned and truncated to fit embedding model context window
"""

import os
import time
from dataclasses import dataclass, field
from typing import Optional
from newsapi import NewsApiClient
from dotenv import load_dotenv

load_dotenv()

MAX_ARTICLE_CHARS = 1500   # truncate long articles
MIN_ARTICLE_CHARS = 100    # skip articles with no content


@dataclass
class NewsDocument:
    """
    A single news article formatted as a RAG document.
    content  → gets embedded and stored in FAISS
    metadata → returned alongside retrieval results for citation
    """
    content:    str
    source:     str
    title:      str
    url:        str
    published:  str
    ticker:     str
    doc_id:     str = field(default_factory=lambda: str(time.time()))

    def to_context_string(self) -> str:
        return (
            f"[{self.published}] {self.source}: {self.title}\n"
            f"{self.content}"
        )


class NewsFetcher:
    """
    Fetches and structures financial news for RAG ingestion.
    """

    def __init__(self):
        api_key = os.getenv("NEWS_API_KEY")
        if not api_key:
            raise ValueError("NEWS_API_KEY not set in .env")
        self.client   = NewsApiClient(api_key=api_key)
        self._seen_urls: set[str] = set()

    def _clean_text(self, text: Optional[str]) -> str:
        if not text:
            return ""
        # Remove [+N chars] truncation notice from NewsAPI free tier
        text = text.split("[+")[0].strip()
        # Collapse whitespace
        text = " ".join(text.split())
        return text[:MAX_ARTICLE_CHARS]

    def _article_to_doc(self, article: dict, ticker: str) -> Optional[NewsDocument]:
        url = article.get("url", "")
        if url in self._seen_urls:
            return None

        title       = self._clean_text(article.get("title", ""))
        description = self._clean_text(article.get("description", ""))
        content     = self._clean_text(article.get("content", ""))

        # Build best available content: prefer content > description > title
        body = content or description or title
        if len(body) < MIN_ARTICLE_CHARS:
            return None

        self._seen_urls.add(url)

        return NewsDocument(
            content=body,
            source=article.get("source", {}).get("name", "Unknown"),
            title=title,
            url=url,
            published=article.get("publishedAt", "")[:10],  # YYYY-MM-DD
            ticker=ticker,
        )

    def fetch_for_ticker(
        self,
        ticker: str,
        company_name: str = "",
        max_articles: int = 20,
    ) -> list[NewsDocument]:
        """
        Fetch recent news for a ticker.
        Queries both the ticker symbol and company name for better coverage.
        """
        query = f"{ticker} stock"
        if company_name:
            query = f"{company_name} OR {ticker} stock market"

        try:
            response = self.client.get_everything(
                q=query,
                language="en",
                sort_by="publishedAt",
                page_size=min(max_articles, 100),
            )
        except Exception as e:
            print(f"[NewsFetcher] API error for {ticker}: {e}")
            return []

        articles = response.get("articles", [])
        docs     = []

        for article in articles:
            doc = self._article_to_doc(article, ticker)
            if doc:
                docs.append(doc)

        print(f"[NewsFetcher] {ticker}: fetched {len(docs)} articles")
        return docs

    def fetch_market_headlines(self, max_articles: int = 30) -> list[NewsDocument]:
        """
        Fetch general market headlines (not ticker-specific).
        Used to populate the base RAG knowledge with macro context.
        """
        try:
            response = self.client.get_top_headlines(
                category="business",
                language="en",
                page_size=min(max_articles, 100),
            )
        except Exception as e:
            print(f"[NewsFetcher] Headlines error: {e}")
            return []

        docs = []
        for article in response.get("articles", []):
            doc = self._article_to_doc(article, "MARKET")
            if doc:
                docs.append(doc)

        print(f"[NewsFetcher] Market headlines: fetched {len(docs)} articles")
        return docs

    def fetch_all(
        self,
        tickers: list[str],
        profiles: dict[str, str] = None,
    ) -> list[NewsDocument]:
        """
        Fetch news for multiple tickers + general headlines.
        profiles: dict of {ticker: company_name} for better queries.
        """
        all_docs = self.fetch_market_headlines()

        for ticker in tickers:
            company_name = (profiles or {}).get(ticker, "")
            docs = self.fetch_for_ticker(ticker, company_name)
            all_docs.extend(docs)
            time.sleep(0.2)  # be polite to the API

        print(f"[NewsFetcher] Total documents: {len(all_docs)}")
        return all_docs
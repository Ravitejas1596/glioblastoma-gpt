"""
PubMed E-utilities Fetcher
Fetches GBM-related abstracts from NCBI PubMed using the E-utilities API.
Rate limited to 3 req/sec (10/sec with API key).
"""
from __future__ import annotations

import asyncio
import time
from typing import AsyncGenerator

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from gbm_copilot.config import NCBI_EMAIL, MAX_PUBMED_RESULTS, QUICK_PUBMED_RESULTS, INGEST_QUICK_MODE

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
GBM_QUERIES = [
    "glioblastoma[MeSH Terms]",
    "glioblastoma multiforme treatment",
    "GBM IDH MGMT methylation",
    "temozolomide radiotherapy glioblastoma",
    "bevacizumab glioblastoma recurrent",
    "glioblastoma clinical trial phase II III",
    "glioblastoma immunotherapy checkpoint",
    "glioblastoma tumor treating fields Optune",
    "glioblastoma surgery resection outcomes",
    "glioblastoma prognosis survival",
    "glioblastoma stem cells",
    "EGFR amplification glioblastoma",
    "TERT PTEN glioblastoma molecular",
    "glioblastoma WHO classification 2021",
    "glioblastoma quality of life caregiver",
]

BATCH_SIZE = 200
RATE_LIMIT_DELAY = 0.34  # ~3 requests/second


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def _esearch(client: httpx.AsyncClient, query: str, max_results: int) -> list[str]:
    """Search PubMed and return PMIDs."""
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "email": NCBI_EMAIL,
        "usehistory": "y",
    }
    resp = await client.get(f"{EUTILS_BASE}/esearch.fcgi", params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("esearchresult", {}).get("idlist", [])


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def _efetch_batch(client: httpx.AsyncClient, pmids: list[str]) -> list[dict]:
    """Fetch abstract data for a batch of PMIDs."""
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
        "email": NCBI_EMAIL,
    }
    resp = await client.get(f"{EUTILS_BASE}/efetch.fcgi", params=params, timeout=60)
    resp.raise_for_status()

    from bs4 import BeautifulSoup
    soup = BeautifulSoup(resp.text, "lxml-xml")
    articles = []

    for article in soup.find_all("PubmedArticle"):
        pmid_tag = article.find("PMID")
        title_tag = article.find("ArticleTitle")
        abstract_tag = article.find("AbstractText")
        journal_tag = article.find("Journal")
        year_tag = article.find("PubDate")

        if not pmid_tag or not abstract_tag:
            continue

        journal_name = ""
        if journal_tag:
            title = journal_tag.find("Title")
            journal_name = title.get_text() if title else ""

        pub_year = ""
        if year_tag:
            year = year_tag.find("Year")
            pub_year = year.get_text() if year else ""

        articles.append({
            "pmid": pmid_tag.get_text(),
            "title": title_tag.get_text() if title_tag else "",
            "abstract": abstract_tag.get_text(),
            "journal": journal_name,
            "year": pub_year,
            "source": "pubmed",
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid_tag.get_text()}/",
        })

    return articles


async def fetch_pubmed_abstracts(
    max_results: int | None = None,
) -> AsyncGenerator[dict, None]:
    """
    Async generator yielding PubMed article dicts.
    Fetches from GBM_QUERIES and deduplicates by PMID.
    """
    if max_results is None:
        max_results = QUICK_PUBMED_RESULTS if INGEST_QUICK_MODE else MAX_PUBMED_RESULTS

    per_query = max(50, max_results // len(GBM_QUERIES))
    seen_pmids: set[str] = set()

    async with httpx.AsyncClient(
        headers={"User-Agent": f"GlioblastomaGPT/1.0 ({NCBI_EMAIL})"}
    ) as client:
        for query in GBM_QUERIES:
            pmids = await _esearch(client, query, per_query)
            new_pmids = [p for p in pmids if p not in seen_pmids]
            seen_pmids.update(new_pmids)

            for i in range(0, len(new_pmids), BATCH_SIZE):
                batch = new_pmids[i : i + BATCH_SIZE]
                await asyncio.sleep(RATE_LIMIT_DELAY)
                articles = await _efetch_batch(client, batch)
                for article in articles:
                    yield article

            if len(seen_pmids) >= max_results:
                break

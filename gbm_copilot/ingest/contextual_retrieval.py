"""
Contextual Retrieval (Anthropic-style)
Prepends a context summary to each chunk before embedding.
This dramatically improves retrieval by "situating" each chunk within its document.
"""
from __future__ import annotations

from openai import AsyncOpenAI

from gbm_copilot.config import OPENAI_API_KEY, OPENAI_MODEL

_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

CONTEXT_PROMPT = """\
Here is the full document:
<document>
{document}
</document>

Here is the chunk from the document which we wish to add context to:
<chunk>
{chunk}
</chunk>

Please give a short, succinct context (1-2 sentences) that situates this chunk within the overall document.
The context should:
1. State the document's subject (paper title, drug name, trial info, etc.)
2. Note the specific aspect this chunk covers (method, result, side effects, etc.)
3. Include any GBM-specific acronyms or gene names relevant to this chunk
Only output the context sentence(s), nothing else."""


async def add_context(
    chunk_text: str,
    document_text: str,
    max_doc_tokens: int = 4000,
) -> str:
    """
    Prepend Anthropic-style contextual summary to a chunk.
    Truncates document if too long.
    """
    # Truncate document to avoid excessive token usage
    doc_excerpt = document_text[:max_doc_tokens * 4]  # ~4 chars/token

    prompt = CONTEXT_PROMPT.format(document=doc_excerpt, chunk=chunk_text)

    response = await _client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0,
    )

    context = response.choices[0].message.content.strip()
    return f"{context}\n\n{chunk_text}"


async def add_context_batch(
    chunks: list[dict],
    document_text: str,
    concurrency: int = 5,
) -> list[dict]:
    """
    Add contextual retrieval to a batch of chunks.
    Uses limited concurrency to avoid rate limits.
    """
    import asyncio
    semaphore = asyncio.Semaphore(concurrency)

    async def process_chunk(chunk: dict) -> dict:
        async with semaphore:
            try:
                enriched_text = await add_context(chunk["text"], document_text)
                return {**chunk, "text": enriched_text, "has_context": True}
            except Exception:
                # Fallback: use original text without context
                return {**chunk, "has_context": False}

    tasks = [process_chunk(c) for c in chunks]
    return await asyncio.gather(*tasks)

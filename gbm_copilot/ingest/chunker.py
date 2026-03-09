"""
Semantic Chunker for GBM Documents
Splits documents into chunks of ~256 tokens with 20% overlap.
Preserves sentence boundaries for better semantic coherence.
"""
from __future__ import annotations

import re
from typing import Any

import tiktoken

from gbm_copilot.config import CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_PCT

_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text))


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences, preserving medical abbreviations."""
    # Protect common medical abbreviations from splitting
    protected = text
    abbrevs = [
        "Dr.", "vs.", "et al.", "Fig.", "No.", "Vol.", "ca.", "approx.",
        "e.g.", "i.e.", "i.v.", "p.o.", "b.i.d.", "t.i.d.", "q.d.",
        "mg/m2", "mg/kg",
    ]
    for abbrev in abbrevs:
        protected = protected.replace(abbrev, abbrev.replace(".", "<<DOT>>"))

    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', protected)

    # Restore protected dots
    return [s.replace("<<DOT>>", ".") for s in sentences]


def chunk_text(
    text: str,
    metadata: dict[str, Any],
    chunk_size: int = CHUNK_SIZE_TOKENS,
    overlap_pct: float = CHUNK_OVERLAP_PCT,
) -> list[dict[str, Any]]:
    """
    Split text into overlapping chunks.
    Returns list of chunk dicts with text + metadata.
    """
    if not text or not text.strip():
        return []

    overlap_tokens = int(chunk_size * overlap_pct)
    sentences = split_into_sentences(text)

    chunks: list[dict[str, Any]] = []
    current_sentences: list[str] = []
    current_tokens = 0
    chunk_index = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        sentence_tokens = count_tokens(sentence)

        # If this sentence alone exceeds chunk size, split it by words
        if sentence_tokens > chunk_size:
            if current_sentences:
                chunk_text_str = " ".join(current_sentences)
                chunks.append(_make_chunk(chunk_text_str, metadata, chunk_index))
                chunk_index += 1
                current_sentences = []
                current_tokens = 0

            # Hard split the long sentence
            words = sentence.split()
            window: list[str] = []
            window_tokens = 0
            for word in words:
                word_tokens = count_tokens(word)
                if window_tokens + word_tokens > chunk_size and window:
                    chunk_text_str = " ".join(window)
                    chunks.append(_make_chunk(chunk_text_str, metadata, chunk_index))
                    chunk_index += 1

                    # Overlap: keep last overlap_tokens worth of words
                    overlap_window: list[str] = []
                    overlap_count = 0
                    for w in reversed(window):
                        wt = count_tokens(w)
                        if overlap_count + wt <= overlap_tokens:
                            overlap_window.insert(0, w)
                            overlap_count += wt
                        else:
                            break
                    window = overlap_window
                    window_tokens = overlap_count

                window.append(word)
                window_tokens += word_tokens

            if window:
                current_sentences = [" ".join(window)]
                current_tokens = window_tokens
            continue

        # Normal case: add sentence to current chunk
        if current_tokens + sentence_tokens > chunk_size and current_sentences:
            chunk_text_str = " ".join(current_sentences)
            chunks.append(_make_chunk(chunk_text_str, metadata, chunk_index))
            chunk_index += 1

            # Roll back to overlap window
            overlap_sentences: list[str] = []
            overlap_count = 0
            for s in reversed(current_sentences):
                st = count_tokens(s)
                if overlap_count + st <= overlap_tokens:
                    overlap_sentences.insert(0, s)
                    overlap_count += st
                else:
                    break
            current_sentences = overlap_sentences
            current_tokens = overlap_count

        current_sentences.append(sentence)
        current_tokens += sentence_tokens

    # Final chunk
    if current_sentences:
        chunk_text_str = " ".join(current_sentences)
        chunks.append(_make_chunk(chunk_text_str, metadata, chunk_index))

    return chunks


def _make_chunk(text: str, metadata: dict[str, Any], index: int) -> dict[str, Any]:
    """Create a chunk dict with text and metadata."""
    return {
        "text": text,
        "chunk_index": index,
        "token_count": count_tokens(text),
        **metadata,
    }

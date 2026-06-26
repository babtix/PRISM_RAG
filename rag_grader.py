"""
Corrective-RAG relevance grader for MedCore.

After the hybrid retrieval step (BM25 + vector → RRF) returns a ranked list of
candidate parent chunks, this module scores each chunk for relevance to the
user's query using a fast, non-streaming Flash model call via OpenRouter.

Design decisions
────────────────
* One grader call per chunk, all fired in parallel with asyncio.gather.
* The grader prompt is intentionally minimal: the model must reply with exactly
  one word — "relevant" or "irrelevant".  This keeps latency and cost minimal
  (≈ 1–3 output tokens per chunk).
* On any failure (network error, unexpected output, timeout) the grader
  defaults to KEEP the chunk (fail-open) — we prefer a slightly noisy context
  over silently dropping a possibly useful chunk.
* The grader uses the cheapest available Flash model to avoid spending Pro
  quota for an internal quality gate.
"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Tuple

import httpx

logger = logging.getLogger(__name__)

# Maximum seconds to wait for a single grader call.
_GRADER_TIMEOUT = 8.0

# Model used for grading — always Flash regardless of user's model preference.
_GRADER_MODEL = "deepseek/deepseek-chat-v3-0324:free"

# Minimum fraction of chunks that must pass grading before we declare
# "no relevant source found".  If 0 pass, we still let the LLM answer from
# general training data but we flag the call as source_not_found=True.
# (This threshold is not used for filtering — ALL passing chunks are returned.)

_GRADER_SYSTEM = (
    "You are a strict medical-content relevance grader. "
    "You will receive a QUESTION and a TEXT CHUNK from a medical textbook. "
    "Your sole job is to decide whether the chunk contains information that "
    "would help answer the question. "
    "Reply with exactly one word: 'relevant' or 'irrelevant'. "
    "Do not add any explanation, punctuation, or other text."
)


async def _grade_single_chunk(
    query: str,
    chunk: str,
    client: httpx.AsyncClient,
    api_key: str,
    base_url: str,
) -> bool:
    """
    Ask the grader LLM whether *chunk* is relevant to *query*.

    Returns True  → keep chunk (relevant or grader failed → fail-open)
    Returns False → drop chunk (irrelevant)
    """
    user_content = f"QUESTION: {query}\n\nTEXT CHUNK:\n{chunk[:2000]}"

    payload = {
        "model": _GRADER_MODEL,
        "max_tokens": 5,
        "temperature": 0.0,
        "stream": False,
        "messages": [
            {"role": "system", "content": _GRADER_SYSTEM},
            {"role": "user", "content": user_content},
        ],
    }

    try:
        resp = await client.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://medcore.app",
                "X-Title": "MedCore",
            },
            json=payload,
            timeout=_GRADER_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        verdict = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
            .lower()
        )
        is_relevant = verdict.startswith("relevant")
        logger.debug(
            "Grader verdict='%s' relevant=%s chunk_preview='%.60s'",
            verdict,
            is_relevant,
            chunk,
        )
        return is_relevant
    except Exception as exc:
        # Fail-open: keep the chunk if the grader is unavailable
        logger.warning("Grader call failed (keeping chunk by default): %s", exc)
        return True


async def grade_chunks(
    query: str,
    chunks: List[str],
    api_key: str,
    base_url: str = "https://openrouter.ai/api/v1",
) -> Tuple[List[str], bool]:
    """
    Grade a list of candidate chunks for relevance to *query*.

    All chunks are graded in parallel.

    Returns
    ───────
    (relevant_chunks, source_found)

    relevant_chunks : list of chunks that passed the grader
                      (may be empty if none passed)
    source_found    : False if zero chunks were relevant — the caller should
                      modify the system prompt to warn the LLM that no
                      textbook source was found for this query
    """
    if not chunks:
        return [], False

    async with httpx.AsyncClient() as client:
        tasks = [
            _grade_single_chunk(query, chunk, client, api_key, base_url)
            for chunk in chunks
        ]
        results = await asyncio.gather(*tasks)

    relevant = [chunk for chunk, keep in zip(chunks, results) if keep]
    source_found = len(relevant) > 0

    logger.info(
        "Grader: %d/%d chunks passed for query='%.80s'",
        len(relevant),
        len(chunks),
        query,
    )
    return relevant, source_found

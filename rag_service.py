"""
RAG (Retrieval-Augmented Generation) service for MedBot.

Provides hybrid retrieval via:
  1. BM25 (Atlas Search)  — precise keyword matching for medical terms
  2. Vector Search         — semantic similarity matching
  3. RRF fusion            — Reciprocal Rank Fusion (k=60) to merge ranked lists
  4. Parent expansion      — resolve child chunks → parent chunks for richer LLM context
  5. Relevance grading     — Corrective RAG: filter irrelevant chunks via Flash model

Caching
───────
Context results are cached in Redis (key: ``rag:<module_id>:<query_hash>``, TTL 5 min).
Uses ``get_redis_or_none()`` so it degrades gracefully when Redis is unavailable.

Fallback behaviour
──────────────────
* If Atlas Search index is absent (M0/M2/M5 shared cluster), BM25 returns an
  empty list and the pipeline degrades gracefully to vector-only retrieval.
* If both vector search and BM25 fail, the recency-sort fallback is used.
* If the relevance grader fails, all retrieved chunks pass through (fail-open).

Public API
──────────
  retrieve_context(query, module_id, top_k) -> tuple[List[str], bool]
  build_rag_context(chunks, max_tokens)     -> str    (unchanged)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from beanie import PydanticObjectId
from pymongo import DESCENDING

from app.models.rag import TextEmbedding

logger = logging.getLogger(__name__)

_MAX_CONTEXT_TOKENS = 6_000
_RRF_K = 60              # Standard RRF constant
_CACHE_TTL = 300         # seconds — 5 minutes


# ── Token estimation ─────────────────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~1 token per 4 characters."""
    return len(text) // 4


# ── Cache helpers ────────────────────────────────────────────────────────────

def _cache_key(query: str, module_id: Optional[str]) -> str:
    """Deterministic Redis key for a (query, module_id) pair."""
    digest = hashlib.sha256(query.encode()).hexdigest()[:24]
    scope = module_id or "global"
    return f"rag:{scope}:{digest}"


async def _cache_get(query: str, module_id: Optional[str]) -> Optional[Tuple[List[str], bool]]:
    """Return cached (chunks, source_found) or None on miss/error."""
    try:
        from app.core.redis import get_redis_or_none
        from app.core.cache import cache_get

        redis = get_redis_or_none()
        if redis is None:
            return None

        raw = await cache_get(redis, _cache_key(query, module_id))
        if raw is None:
            return None

        data = json.loads(raw)
        return data["chunks"], data["source_found"]
    except Exception as exc:
        logger.debug("RAG cache read failed (non-critical): %s", exc)
        return None


async def _cache_set(
    query: str,
    module_id: Optional[str],
    chunks: List[str],
    source_found: bool,
) -> None:
    """Store (chunks, source_found) in Redis. Silently skips on error."""
    try:
        from app.core.redis import get_redis_or_none
        from app.core.cache import cache_set

        redis = get_redis_or_none()
        if redis is None:
            return

        payload = json.dumps({"chunks": chunks, "source_found": source_found})
        await cache_set(redis, _cache_key(query, module_id), payload, _CACHE_TTL)
    except Exception as exc:
        logger.debug("RAG cache write failed (non-critical): %s", exc)


# ── Query embedding ──────────────────────────────────────────────────────────

async def _get_query_embedding(query: str) -> List[float]:
    """Generate a 768-dim embedding for *query* via OpenRouter.

    Returns a zero vector on failure so the pipeline can degrade gracefully.
    """
    try:
        from app.core.config import settings
        from app.services.medbot import _get_http_client

        client = _get_http_client()
        response = await client.post(
            "https://openrouter.ai/api/v1/embeddings",
            headers={
                "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "google/gemini-embedding-2",
                "input": query,
            },
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
        return data["data"][0]["embedding"]

    except Exception as exc:
        logger.warning("Query embedding generation failed: %s", exc)
        return [0.0] * 768


# ── BM25 (Atlas Search) retrieval ────────────────────────────────────────────

async def _bm25_search(
    query: str,
    module_id: Optional[str],
    top_k: int,
) -> List[Dict[str, Any]]:
    """Full-text BM25 search via Atlas Search on the ``text`` field.

    Only searches child chunks (embeddings exist and are more precisely scoped).
    Degrades gracefully to an empty list if the Atlas Search index does not
    exist (shared cluster tiers M0/M2/M5).

    Returns a list of raw dicts with at minimum ``_id``, ``text``,
    ``chunk_index``, ``pdf_id``, and optionally ``parent_chunk_index``.
    """
    compound_must: List[Dict[str, Any]] = [
        {"equals": {"path": "chunk_type", "value": "child"}}
    ]
    if module_id is not None:
        compound_must.append(
            {"equals": {"path": "module_id", "value": PydanticObjectId(module_id)}}
        )

    pipeline: List[Dict[str, Any]] = [
        {
            "$search": {
                "index": "text_search_index",
                "compound": {
                    "must": compound_must,
                    "should": [
                        {
                            "text": {
                                "query": query,
                                "path": "text",
                                "fuzzy": {"maxEdits": 1},
                            }
                        }
                    ],
                    "minimumShouldMatch": 1,
                },
            }
        },
        {"$limit": top_k},
        {
            "$project": {
                "_id": 1,
                "text": 1,
                "chunk_index": 1,
                "chunk_type": 1,
                "parent_chunk_index": 1,
                "pdf_id": 1,
                "module_id": 1,
                "semester_id": 1,
                "lesson_id": 1,
                "score": {"$meta": "searchScore"},
            }
        },
    ]

    try:
        results = await TextEmbedding.aggregate(pipeline).to_list()
        logger.debug("BM25 search returned %d hits", len(results))
        return results
    except Exception as exc:
        error_str = str(exc).lower()
        if "index not found" in error_str or "no such index" in error_str or "search" in error_str:
            logger.info("Atlas Search unavailable — falling back to local keyword search: %s", exc)
        else:
            logger.warning("BM25 search failed, falling back to local keyword search: %s", exc)
        return await _local_keyword_search(query, module_id, top_k)


async def _local_keyword_search(
    query: str,
    module_id: Optional[str],
    top_k: int,
) -> List[Dict[str, Any]]:
    """Simple regex keyword search — local fallback when Atlas Search is unavailable.

    Extracts meaningful words (≥ 4 chars) from the query and performs a
    case-insensitive OR regex match on the ``text`` field.  Works on any
    MongoDB instance with no extra indexes required.

    This is a low-precision substitute for BM25 — good enough for local
    development/testing.  On M10+ Atlas, the real ``$search`` BM25 pipeline
    runs instead and this function is never called.
    """
    import re

    # Extract words ≥ 4 chars as search tokens (skip common stopwords)
    _STOPWORDS = {"with", "this", "that", "from", "have", "what", "when", "where", "which"}
    tokens = [
        w for w in re.findall(r"[a-zA-Z]{4,}", query)
        if w.lower() not in _STOPWORDS
    ]
    if not tokens:
        return []

    # Build a single regex that matches ANY of the tokens (case-insensitive)
    pattern = "|".join(re.escape(t) for t in tokens[:8])  # cap at 8 tokens

    filter_query: Dict[str, Any] = {
        "chunk_type": "child",
        "text": {"$regex": pattern, "$options": "i"},
    }
    if module_id is not None:
        filter_query["module_id"] = PydanticObjectId(module_id)

    try:
        docs = await TextEmbedding.find(filter_query).limit(top_k).to_list()
    except Exception as exc:
        logger.warning("Local keyword search also failed: %s", exc)
        return []

    logger.debug("Local keyword search: %d hits for pattern '%s'", len(docs), pattern[:60])

    return [
        {
            "_id": doc.id,
            "text": doc.text,
            "chunk_index": doc.chunk_index,
            "chunk_type": doc.chunk_type,
            "parent_chunk_index": doc.parent_chunk_index,
            "pdf_id": doc.pdf_id,
            "module_id": doc.module_id,
            "semester_id": doc.semester_id,
            "lesson_id": doc.lesson_id,
        }
        for doc in docs
        if doc.text
    ]




# ── Local cosine-similarity fallback (for non-Atlas / M0 environments) ──────

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Pure-Python cosine similarity. No external dependencies."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


async def _local_vector_search(
    embedding: List[float],
    module_id: Optional[str],
    top_k: int,
) -> List[Dict[str, Any]]:
    """In-memory cosine similarity search — works on any MongoDB (local, M0, M10+).

    Used as an automatic fallback when $vectorSearch is unavailable (no Atlas
    Vector Search index, local development environment, or M0 free cluster).

    Fetches all child-chunk embeddings from MongoDB using a regular ``find``
    query, scores them in Python, and returns the top-*top_k* results in the
    same dict format as the Atlas pipeline so the rest of the pipeline
    (RRF fusion, parent expansion, grading, caching) is unaffected.

    Performance note: this is O(N * D) where N = number of child chunks and
    D = embedding dimension (768).  Fine for local testing with one PDF
    (~50-200 child chunks); not suitable for large production datasets.
    """
    filter_query: Dict[str, Any] = {
        "chunk_type": "child",
        # Only consider docs that actually have an embedding stored
        "embedding.0": {"$exists": True},
    }
    if module_id is not None:
        filter_query["module_id"] = PydanticObjectId(module_id)

    try:
        all_chunks = await TextEmbedding.find(filter_query).to_list()
    except Exception as exc:
        logger.warning("Local vector search: DB fetch failed: %s", exc)
        return []

    if not all_chunks:
        return []

    # Score every chunk and sort descending
    scored = [
        (_cosine_similarity(embedding, chunk.embedding), chunk)
        for chunk in all_chunks
        if chunk.embedding
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)

    top = scored[:top_k]
    logger.debug(
        "Local cosine search: scored %d chunks, returning top %d",
        len(scored),
        len(top),
    )

    return [
        {
            "_id": chunk.id,
            "text": chunk.text,
            "chunk_index": chunk.chunk_index,
            "chunk_type": chunk.chunk_type,
            "parent_chunk_index": chunk.parent_chunk_index,
            "pdf_id": chunk.pdf_id,
            "module_id": chunk.module_id,
            "semester_id": chunk.semester_id,
            "lesson_id": chunk.lesson_id,
        }
        for _score, chunk in top
        if _score > 0.0
    ]


# ── Vector Search retrieval ──────────────────────────────────────────────────

async def _vector_search(
    embedding: List[float],
    module_id: Optional[str],
    top_k: int,
) -> List[Dict[str, Any]]:
    """Semantic similarity search.

    Tries Atlas Vector Search (``$vectorSearch``) first.  If the index is
    unavailable (local MongoDB, M0 free cluster, missing index), automatically
    falls back to ``_local_vector_search`` — an in-memory cosine similarity
    scan.  The fallback produces identical output format so the rest of the
    pipeline (RRF, parent expansion, grading) runs without modification.

    On Atlas M10+: Atlas handles ANN efficiently with the vector index.
    Locally / M0: Python cosine scan runs over all stored child embeddings.

    IMPORTANT: On Atlas, the vector index must declare ``chunk_type`` as a
    filterable field (see ``scripts/update_vector_search_index.py``).
    """
    vector_filter: Dict[str, Any] = {"chunk_type": {"$eq": "child"}}
    if module_id is not None:
        vector_filter["module_id"] = {"$eq": PydanticObjectId(module_id)}

    pipeline: List[Dict[str, Any]] = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": embedding,
                "numCandidates": max(top_k * 10, 100),
                "limit": top_k,
                "filter": vector_filter,
            }
        },
        {
            "$project": {
                "_id": 1,
                "text": 1,
                "chunk_index": 1,
                "chunk_type": 1,
                "parent_chunk_index": 1,
                "pdf_id": 1,
                "module_id": 1,
                "semester_id": 1,
                "lesson_id": 1,
            }
        },
    ]

    try:
        results = await TextEmbedding.aggregate(pipeline).to_list()
        logger.debug("Atlas vector search returned %d hits", len(results))
        return results
    except Exception as exc:
        logger.info(
            "$vectorSearch unavailable (%s) — falling back to local cosine search", exc
        )
        return await _local_vector_search(embedding, module_id, top_k)




# ── Reciprocal Rank Fusion ───────────────────────────────────────────────────

def _rrf_fuse(
    bm25_docs: List[Dict[str, Any]],
    vec_docs: List[Dict[str, Any]],
    top_k: int,
    k: int = _RRF_K,
) -> List[Dict[str, Any]]:
    """Fuse two ranked lists using Reciprocal Rank Fusion.

    RRF score = Σ  1 / (k + rank)   for each list the document appears in.

    Returns the top *top_k* deduplicated documents, sorted by RRF score desc.
    Documents are identified by their MongoDB ``_id``.
    """
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Dict[str, Any]] = {}

    for rank, doc in enumerate(bm25_docs, start=1):
        doc_id = str(doc["_id"])
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        doc_map[doc_id] = doc

    for rank, doc in enumerate(vec_docs, start=1):
        doc_id = str(doc["_id"])
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        doc_map[doc_id] = doc

    sorted_ids = sorted(scores, key=lambda did: scores[did], reverse=True)[:top_k]
    fused = [doc_map[did] for did in sorted_ids]

    logger.debug(
        "RRF fusion: %d BM25 + %d vec → %d fused (top_k=%d)",
        len(bm25_docs),
        len(vec_docs),
        len(fused),
        top_k,
    )
    return fused


# ── Parent chunk expansion (batched) ─────────────────────────────────────────

async def _expand_to_parents(
    child_docs: List[Dict[str, Any]],
) -> List[str]:
    """For each child chunk, fetch its parent document text.

    Uses a **single batched query** (``$or`` with all parent references) instead
    of one ``find_one`` per child — eliminates the N+1 round-trip problem.

    Deduplicates parents: if multiple children share the same parent, the
    parent text appears only once in the output (preserving RRF rank order).

    Falls back to child text for any child that is missing a parent_chunk_index
    or whose parent document cannot be found.
    """
    if not child_docs:
        return []

    # ── Build the ordered list of unique (pdf_id, parent_chunk_index) refs ──
    seen_keys: set[str] = set()
    ordered_keys: List[Tuple[Any, int]] = []  # (pdf_id, parent_chunk_index)
    legacy_texts: List[str] = []              # child text for docs without a parent ref

    for child in child_docs:
        parent_chunk_idx: Optional[int] = child.get("parent_chunk_index")
        pdf_id = child.get("pdf_id")

        if parent_chunk_idx is None or pdf_id is None:
            # Legacy or malformed doc — use child text
            text = child.get("text", "")
            if text and text[:64] not in seen_keys:
                seen_keys.add(text[:64])
                legacy_texts.append(text)
            continue

        key = f"{pdf_id}:{parent_chunk_idx}"
        if key not in seen_keys:
            seen_keys.add(key)
            ordered_keys.append((pdf_id, parent_chunk_idx))

    if not ordered_keys and not legacy_texts:
        return []

    # ── Single batched DB query for all parent docs ──────────────────────────
    parent_text_map: Dict[str, str] = {}

    if ordered_keys:
        or_conditions = [
            {
                "pdf_id": pdf_id,
                "chunk_type": "parent",
                "chunk_index": idx,
            }
            for pdf_id, idx in ordered_keys
        ]
        try:
            parent_docs = await TextEmbedding.find({"$or": or_conditions}).to_list()
            for doc in parent_docs:
                key = f"{doc.pdf_id}:{doc.chunk_index}"
                if doc.text:
                    parent_text_map[key] = doc.text
        except Exception as exc:
            logger.warning("Batched parent fetch failed: %s — falling back to child texts", exc)

    # ── Assemble output in RRF-ranked order ──────────────────────────────────
    parent_texts: List[str] = []

    for child in child_docs:
        parent_chunk_idx = child.get("parent_chunk_index")
        pdf_id = child.get("pdf_id")

        if parent_chunk_idx is None or pdf_id is None:
            continue  # handled in legacy_texts above

        key = f"{pdf_id}:{parent_chunk_idx}"
        text = parent_text_map.get(key) or child.get("text", "")
        if text and key + ":out" not in seen_keys:
            seen_keys.add(key + ":out")
            parent_texts.append(text)

    # Prepend any legacy child texts (maintain deterministic order)
    result = legacy_texts + parent_texts
    logger.debug(
        "Parent expansion: %d child docs → %d unique parent texts (%d batch query)",
        len(child_docs),
        len(result),
        len(ordered_keys),
    )
    return result


# ── Recency fallback ─────────────────────────────────────────────────────────

async def _recency_fallback(
    module_id: Optional[str],
    top_k: int,
) -> List[str]:
    """Return the most recently stored child chunks as a last-resort fallback.

    WARNING: this degrades MedBot response quality significantly.  It should
    only fire if both BM25 and vector search raise exceptions.
    """
    filter_query: Dict[str, Any] = {"chunk_type": "child"}
    if module_id is not None:
        filter_query["module_id"] = PydanticObjectId(module_id)

    embeddings = (
        await TextEmbedding.find(filter_query)
        .sort([("created_at", DESCENDING)])
        .limit(top_k)
        .to_list()
    )
    return [emb.text for emb in embeddings if emb.text]


# ── Public API ───────────────────────────────────────────────────────────────

async def retrieve_context(
    query: str,
    module_id: Optional[str] = None,
    top_k: int = 5,
) -> Tuple[List[str], bool]:
    """Retrieve relevant parent-chunk text via hybrid BM25 + vector search.

    Pipeline
    ────────
    1. Check Redis cache — return immediately on hit
    2. Embed the query (async)
    3. Run BM25 and vector search in parallel
    4. Fuse with Reciprocal Rank Fusion (k=60)
    5. Expand child matches → parent chunks (single batched DB query)
    6. Grade relevance (Corrective RAG); irrelevant chunks are filtered out
    7. Store result in Redis cache

    Parameters
    ──────────
    query     : raw user query string
    module_id : optional ObjectId string to scope search to one RAG module
    top_k     : number of chunks to return after fusion (before grading)

    Returns
    ───────
    (context_chunks, source_found)

    context_chunks  : list of parent-chunk text strings ready to join into the
                      system prompt context block
    source_found    : False when the relevance grader determined that none of
                      the retrieved chunks are relevant to the query
    """
    # ── Step 0: cache hit ────────────────────────────────────────────────────
    cached = await _cache_get(query, module_id)
    if cached is not None:
        logger.debug("RAG cache hit for query='%.60s'", query)
        return cached

    try:
        # ── Step 1: embed query ──────────────────────────────────────────────
        embedding = await _get_query_embedding(query)

        # ── Step 2: parallel BM25 + vector retrieval ─────────────────────────
        bm25_results, vec_results = await asyncio.gather(
            _bm25_search(query, module_id, top_k * 2),
            _vector_search(embedding, module_id, top_k * 2),
        )

        # ── Step 3: RRF fusion ───────────────────────────────────────────────
        fused = _rrf_fuse(bm25_results, vec_results, top_k=top_k)

        if not fused:
            logger.warning(
                "Both BM25 and vector search returned 0 results — using recency fallback"
            )
            fallback_texts = await _recency_fallback(module_id, top_k)
            result = (fallback_texts, bool(fallback_texts))
            await _cache_set(query, module_id, *result)
            return result

        # ── Step 4: expand child → parent (single batched query) ─────────────
        parent_texts = await _expand_to_parents(fused)

        if not parent_texts:
            result = ([], False)
            await _cache_set(query, module_id, *result)
            return result

        # ── Step 5: relevance grading (Corrective RAG) ───────────────────────
        try:
            from app.core.config import settings
            from app.services.rag_grader import grade_chunks

            relevant_chunks, source_found = await grade_chunks(
                query=query,
                chunks=parent_texts,
                api_key=settings.OPENROUTER_API_KEY,
            )
            result = (relevant_chunks, source_found)

        except Exception as exc:
            logger.warning("Relevance grader unavailable: %s — skipping grading", exc)
            result = (parent_texts, True)

        # ── Step 6: cache and return ─────────────────────────────────────────
        await _cache_set(query, module_id, *result)
        return result

    except Exception as exc:
        logger.warning(
            "Hybrid retrieval pipeline failed, falling back to recency sort: %s", exc
        )
        fallback_texts = await _recency_fallback(module_id, top_k)
        return fallback_texts, bool(fallback_texts)


# ── Context builder (unchanged) ───────────────────────────────────────────────

def build_rag_context(chunks: List[str], max_tokens: int = _MAX_CONTEXT_TOKENS) -> str:
    """Join context chunks into a single string, trimming to stay within token limit."""
    if not chunks:
        return "No additional context available."

    selected: List[str] = []
    total = 0

    for chunk in chunks:
        chunk_tokens = _estimate_tokens(chunk)
        if total + chunk_tokens > max_tokens:
            break
        selected.append(chunk)
        total += chunk_tokens

    return "\n---\n".join(selected) if selected else "No additional context available."

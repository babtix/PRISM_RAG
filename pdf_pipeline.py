"""
PDF processing pipeline for RAG modules.

Provides end-to-end: PDF upload → text extraction → parent-child chunking
→ embedding (child chunks only) → storage.

Parent-child strategy
─────────────────────
Each section of a medical PDF is represented as:

  • One *parent* chunk  (~600 tokens, ~2 400 chars) — stored text-only with no
    embedding vector.  Sent to the LLM after retrieval to supply broad context.

  • N *child*  chunks  (~150 tokens, ~600 chars)  — carry a 768-dim embedding
    and are the targets for both Atlas Vector Search and Atlas Search (BM25).
    Each child stores the ``parent_chunk_index`` of its sibling parent.

This lets retrieval remain precise (small semantic windows) while the LLM
receives richer, more coherent context (the parent window).
"""

import logging
import math
from typing import Any, List, Optional, Tuple

from beanie import PydanticObjectId

from app.models.rag import PDFDocument, TextEmbedding, RAGModule

logger = logging.getLogger(__name__)

# ── Chunking Config ──────────────────────────────────────────────────────────

_PARENT_CHUNK_TOKENS: int = 600    # target parent window  (~2 400 chars)
_CHILD_CHUNK_TOKENS: int  = 150    # target child window   (~600 chars)
_CHILD_OVERLAP_TOKENS: int = 20    # child overlap         (~80 chars)
_CHARS_PER_TOKEN: int      = 4     # rough character-per-token approximation


# ── Text splitting helpers ───────────────────────────────────────────────────

def _split_at_boundary(
    text: str,
    target_chars: int,
    search_window: int = 200,
) -> List[str]:
    """Split *text* into chunks of approximately *target_chars* characters.

    Prefers splitting at paragraph (``\\n\\n``) then sentence boundaries to
    avoid cutting mid-sentence.  Returns a list of non-empty stripped strings.
    """
    if len(text) <= target_chars:
        stripped = text.strip()
        return [stripped] if stripped else []

    segments: List[str] = []
    start = 0

    while start < len(text):
        end = start + target_chars

        if end < len(text):
            # 1. Prefer a paragraph break
            para_break = text.rfind(
                "\n\n", start + target_chars // 2, end + search_window
            )
            if para_break > start:
                end = para_break
            else:
                # 2. Fall back to sentence boundary
                for sep in [". ", ".\n", "? ", "! "]:
                    sent_break = text.rfind(
                        sep, start + target_chars // 2, end + search_window // 2
                    )
                    if sent_break > start:
                        end = sent_break + len(sep)
                        break

        segment = text[start:end].strip()
        if segment:
            segments.append(segment)

        # Advance; no overlap at the parent level
        start = max(start + 1, end)

    return segments


def _chunk_parent_child(
    text: str,
) -> List[Tuple[str, str, int, int]]:
    """Split *text* into parent-child chunk pairs.

    Returns a list of ``(parent_text, child_text, parent_idx, child_idx)``.

    Algorithm
    ─────────
    1. Split the full document into *parent* segments (~600 tokens).
    2. For each parent, produce N *child* sub-segments (~150 tokens, with
       overlap) so that child chunks together cover the parent text.
    3. Assign a monotonically increasing ``child_idx`` across all parents.

    The caller is responsible for assigning ``chunk_index`` values to the
    stored ``TextEmbedding`` documents.
    """
    parent_char_size  = _PARENT_CHUNK_TOKENS  * _CHARS_PER_TOKEN
    child_char_size   = _CHILD_CHUNK_TOKENS   * _CHARS_PER_TOKEN
    child_char_overlap = _CHILD_OVERLAP_TOKENS * _CHARS_PER_TOKEN

    parent_segments = _split_at_boundary(text, parent_char_size)

    pairs: List[Tuple[str, str, int, int]] = []
    child_idx = 0

    for parent_idx, parent_text in enumerate(parent_segments):
        # Produce child chunks from this parent's text
        if len(parent_text) <= child_char_size:
            # Short parent → single child equals the parent text
            pairs.append((parent_text, parent_text, parent_idx, child_idx))
            child_idx += 1
        else:
            child_start = 0
            while child_start < len(parent_text):
                child_end = child_start + child_char_size

                if child_end < len(parent_text):
                    # Try to break at a sentence boundary
                    for sep in [". ", ".\n", "? ", "! "]:
                        sb = parent_text.rfind(
                            sep,
                            child_start + child_char_size // 2,
                            child_end + 80,
                        )
                        if sb > child_start:
                            child_end = sb + len(sep)
                            break

                child_text = parent_text[child_start:child_end].strip()
                if child_text:
                    pairs.append((parent_text, child_text, parent_idx, child_idx))
                    child_idx += 1

                # Advance with overlap
                child_start = max(child_start + 1, child_end - child_char_overlap)

    return pairs


# ── Embedding generation ─────────────────────────────────────────────────────

_EMBED_BATCH_SIZE: int = 96   # stay well under OpenRouter's payload limit


async def _embed_batch(texts: List[str], client: Any) -> List[List[float]]:
    """Send one batch of texts to the embeddings API. Returns one vector per text."""
    import httpx
    from app.core.config import settings

    response = await client.post(
        "https://openrouter.ai/api/v1/embeddings",
        headers={
            "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "google/gemini-embedding-2",
            "input": texts,
        },
        timeout=120.0,
    )
    response.raise_for_status()
    data = response.json()

    embeddings: List[Optional[List[float]]] = [None] * len(texts)
    for item in data.get("data", []):
        idx = item.get("index")
        if idx is not None and idx < len(texts):
            embeddings[idx] = item.get("embedding", [0.0] * 768)

    return [emb if emb is not None else [0.0] * 768 for emb in embeddings]


async def _generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embedding vectors for a list of text chunks via OpenRouter.

    Chunks the request into batches of ``_EMBED_BATCH_SIZE`` (96) to avoid
    hitting API payload size limits on large PDFs.  Batches are processed
    sequentially with a short back-off between calls.

    Uses google/gemini-embedding-2. Returns zero vectors on failure.
    """
    if not texts:
        return []

    import asyncio
    import httpx

    logger.info("Generating embeddings for %d chunks via OpenRouter", len(texts))

    all_embeddings: List[List[float]] = []

    try:
        async with httpx.AsyncClient() as client:
            for batch_start in range(0, len(texts), _EMBED_BATCH_SIZE):
                batch = texts[batch_start : batch_start + _EMBED_BATCH_SIZE]
                batch_num = batch_start // _EMBED_BATCH_SIZE + 1
                total_batches = math.ceil(len(texts) / _EMBED_BATCH_SIZE)

                logger.info(
                    "Embedding batch %d/%d (%d texts)",
                    batch_num,
                    total_batches,
                    len(batch),
                )
                try:
                    batch_embeddings = await _embed_batch(batch, client)
                    all_embeddings.extend(batch_embeddings)
                except Exception as exc:
                    logger.error("Embedding batch %d failed: %s — using zero vectors", batch_num, exc)
                    all_embeddings.extend([[0.0] * 768 for _ in batch])

                # Brief back-off between batches to avoid rate-limit bursts
                if batch_start + _EMBED_BATCH_SIZE < len(texts):
                    await asyncio.sleep(0.5)

    except Exception as exc:
        logger.error("Embedding generation failed: %s", exc)
        return [[0.0] * 768 for _ in texts]

    return all_embeddings


# ── PDF text extraction ──────────────────────────────────────────────────────

def _extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF bytes using PyPDF2."""
    try:
        from PyPDF2 import PdfReader
        import io

        reader = PdfReader(io.BytesIO(file_bytes))
        pages: List[str] = []

        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)

        return "\n\n".join(pages)

    except ImportError:
        logger.error(
            "PyPDF2 not installed. Cannot extract PDF text. "
            "Install with: pip install PyPDF2"
        )
        raise RuntimeError("PyPDF2 is required for PDF processing")

    except Exception as exc:
        logger.error("PDF text extraction failed: %s", exc)
        raise


# ── Main pipeline ─────────────────────────────────────────────────────────────

async def process_pdf(
    file_bytes: bytes,
    pdf_doc: PDFDocument,
    module_id: PydanticObjectId,
) -> None:
    """Full PDF processing pipeline: extract → chunk → embed → store.

    Updates the PDFDocument status as it progresses.

    Storage layout
    ──────────────
    For each parent-child pair the pipeline stores TWO ``TextEmbedding`` docs:
      • The parent  (chunk_type="parent", embedding=[])
      • The child   (chunk_type="child",  embedding=<768-dim vector>,
                     parent_chunk_index=<parent chunk_index>)

    Child chunk_index values are assigned as monotonically increasing integers
    starting from 0.  Parent chunk_index values start directly after the last
    child index so that every document in the collection has a unique
    (pdf_id, chunk_index) pair.
    """
    try:
        # ── 1. Extract text ──────────────────────────────────────────────
        logger.info("Processing PDF: %s (module=%s)", pdf_doc.filename, module_id)
        raw_text = _extract_text_from_pdf(file_bytes)

        if not raw_text.strip():
            pdf_doc.status = "failed"
            pdf_doc.error_message = "No text could be extracted from this PDF"
            await pdf_doc.save()
            return

        # Count pages (approximate from PyPDF2)
        try:
            from PyPDF2 import PdfReader
            import io
            reader = PdfReader(io.BytesIO(file_bytes))
            pdf_doc.page_count = len(reader.pages)
        except Exception:
            pdf_doc.page_count = None

        # ── 2. Parent-child chunking ──────────────────────────────────────
        pairs = _chunk_parent_child(raw_text)
        if not pairs:
            pdf_doc.status = "failed"
            pdf_doc.error_message = "Text extraction produced no usable chunks"
            await pdf_doc.save()
            return

        # Collect unique (parent_idx → parent_text) and all child texts/indices
        # Use a dict to deduplicate parents (multiple children share one parent).
        parent_map: dict[int, str] = {}          # parent_idx → parent_text
        child_records: List[Tuple[str, int, int]] = []  # (child_text, child_idx, parent_idx)

        for parent_text, child_text, parent_idx, child_idx in pairs:
            parent_map[parent_idx] = parent_text
            child_records.append((child_text, child_idx, parent_idx))

        child_texts = [cr[0] for cr in child_records]
        logger.info(
            "PDF %s: %d parent chunks, %d child chunks",
            pdf_doc.filename,
            len(parent_map),
            len(child_texts),
        )

        # ── 3. Generate embeddings for child chunks only ──────────────────
        child_embeddings = await _generate_embeddings(child_texts)

        # ── 4. Assign global chunk_index values ───────────────────────────
        # Children occupy indices 0 … N_children-1.
        # Parents occupy indices N_children … N_children + N_parents - 1.
        # This guarantees uniqueness on the (pdf_id, chunk_index) compound key.
        n_children = len(child_records)
        # Map parent_idx → global chunk_index for the parent document
        parent_global_idx: dict[int, int] = {
            p_idx: n_children + i
            for i, p_idx in enumerate(sorted(parent_map.keys()))
        }

        # ── 5. Build TextEmbedding documents ─────────────────────────────
        embedding_docs: List[TextEmbedding] = []

        # Child documents
        for (child_text, child_idx, parent_idx), emb in zip(child_records, child_embeddings):
            embedding_docs.append(
                TextEmbedding(
                    pdf_id=pdf_doc.id,
                    semester_id=pdf_doc.semester_id,
                    module_id=pdf_doc.module_id,
                    lesson_id=pdf_doc.lesson_id,
                    chunk_index=child_idx,
                    chunk_type="child",
                    parent_chunk_index=parent_global_idx[parent_idx],
                    text=child_text,
                    embedding=emb,
                )
            )

        # Parent documents (text-only, no embedding)
        for parent_idx, parent_text in sorted(parent_map.items()):
            embedding_docs.append(
                TextEmbedding(
                    pdf_id=pdf_doc.id,
                    semester_id=pdf_doc.semester_id,
                    module_id=pdf_doc.module_id,
                    lesson_id=pdf_doc.lesson_id,
                    chunk_index=parent_global_idx[parent_idx],
                    chunk_type="parent",
                    parent_chunk_index=None,
                    text=parent_text,
                    embedding=[],
                )
            )

        # Batch insert for performance
        if embedding_docs:
            await TextEmbedding.insert_many(embedding_docs)

        # ── 6. Update document status ─────────────────────────────────────
        pdf_doc.status = "ready"
        pdf_doc.child_chunk_count = len(child_records)
        pdf_doc.parent_chunk_count = len(parent_map)
        # Legacy field: total chunks stored
        pdf_doc.chunk_count = len(embedding_docs)
        await pdf_doc.save()

        logger.info(
            "PDF %s processed successfully: %d child chunks, %d parent chunks, %d pages",
            pdf_doc.filename,
            len(child_records),
            len(parent_map),
            pdf_doc.page_count or 0,
        )

    except Exception as exc:
        logger.error("PDF processing failed for %s: %s", pdf_doc.filename, exc)
        pdf_doc.status = "failed"
        pdf_doc.error_message = str(exc)[:500]
        await pdf_doc.save()

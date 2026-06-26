# PRISM RAG

**Precision Retrieval with Intelligent Source Matching**

PRISM is a production-grade RAG (Retrieval-Augmented Generation) architecture designed for domains where retrieval accuracy and answer honesty are critical. It combines three layered patterns — Hybrid Retrieval, Corrective RAG, and Parent-Child Chunking — into a unified pipeline.

---

## Why PRISM

Most RAG pipelines fail not because the language model is weak, but because the retrieval layer is imprecise. A naive vector search returns semantically similar chunks that may miss exact terminology. A naive keyword search misses conceptual queries. Neither alone validates whether the retrieved content is actually useful before passing it to the model.

PRISM addresses all three failure modes.

---

## System Architecture & Workflow

### Layer 1 — Hybrid Retrieval

Combines dense and sparse retrieval, fused with Reciprocal Rank Fusion (RRF).

- **BM25** handles exact term matching (technical terms, proper nouns, codes)
- **Vector search** handles semantic similarity (concept-level queries)
- **RRF** merges ranked lists without requiring score normalization

### Layer 2 — Corrective RAG

A relevance grader sits between retrieval and generation.

- Each retrieved chunk is scored for relevance to the query
- Chunks below a confidence threshold are discarded
- If no chunks pass the gate, the system returns a structured "source not found" response rather than generating a weak or hallucinated answer
- Optionally triggers query expansion or fallback search on low-confidence retrievals

### Layer 3 — Parent-Child Chunking

Decouples retrieval granularity from generation context.

- **Child chunks** (small, precise) are embedded and indexed for retrieval
- **Parent chunks** (full sections) are stored separately and fetched when a child chunk matches
- The LLM receives the full parent context, not just the matched fragment

---

## Stack Compatibility

PRISM is stack-agnostic at the retrieval layer. The reference implementation uses:

| Component | Technology |
|---|---|
| Vector + BM25 Search | MongoDB Atlas (native hybrid search) |
| Embedding Model | Configurable via OpenRouter |
| LLM | DeepSeek / OpenRouter |
| Backend | FastAPI |
| Cache | Redis |

---

## Pipeline Flow

```
User Query
    |
    v
[Query Processing] — rewrite / expand if needed
    |
    v
[Hybrid Retrieval] — BM25 + Vector → RRF fusion
    |
    v
[Corrective Gate] — relevance scoring per chunk
    |
   / \
  /   \
Pass  Fail → "Source not found" or query expansion
  |
  v
[Parent Fetch] — retrieve full parent section
  |
  v
[LLM Generation] — grounded, context-rich answer
```

---

## Roadmap

- [x] Hybrid retrieval with RRF
- [x] Corrective relevance grader
- [x] Parent-child chunking
- [ ] Agentic loop for multi-step queries
- [ ] RAG-Fusion (multi-query variant)
- [ ] Evaluation benchmarks (RAGAS integration)

---

## Getting Started

```bash
git clone https://github.com/babtix/PRISM_RAG.git
cd PRISM_RAG
pip install -r requirements.txt
```

Configuration is handled via environment variables. Copy `.env.example` to `.env` and fill in your credentials.

---

## License

MIT

---

## Author

Built by [@babtix](https://github.com/babtix) while shipping a SaaS product in a domain where retrieval accuracy is non-negotiable.
"""
RAG data models for MedCore.

Collections:
  - rag_modules      : named knowledge domains (Cardiology, Pharmacology, …)
  - pdf_documents    : uploaded source PDFs with processing status
  - text_embeddings  : child chunks (with vector embeddings) AND parent chunks
                       (text-only, no embedding vector).  Parent chunks supply
                       richer context to the LLM after a child chunk is matched.
"""

from datetime import datetime, timezone
from typing import List, Literal, Optional

from beanie import Document, Indexed, PydanticObjectId
from pydantic import Field
from pymongo import ASCENDING, IndexModel


# ── RAG Module ────────────────────────────────────────────────────────────────


class RAGModule(Document):
    """
    A named knowledge domain that scopes vector searches and provides a
    specialised system prompt (e.g. Cardiology, General Medicine).
    """

    name: str
    slug: str
    description: Optional[str] = None
    icon: str = "Stethoscope"
    color: str = "#3b82f6"
    is_active: bool = True
    system_prompt: Optional[str] = None
    created_by: Optional[PydanticObjectId] = None
    document_count: int = 0
    chunk_count: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Settings:
        name = "rag_modules"
        indexes = [
            IndexModel([("slug", ASCENDING)], unique=True),
            [("is_active", ASCENDING)],
        ]


# ── PDF Document ──────────────────────────────────────────────────────────────


class PDFDocument(Document):
    """
    Tracks an uploaded PDF through the ingestion pipeline.

    status values:
      "processing" → being chunked / embedded
      "ready"      → all chunks stored, available for retrieval
      "failed"     → pipeline error (see error_message)
    """

    filename: str
    r2_key: str
    uploaded_by: PydanticObjectId

    # Optional hierarchical scoping
    semester_id: Optional[PydanticObjectId] = None
    module_id: Optional[PydanticObjectId] = None
    lesson_id: Optional[PydanticObjectId] = None

    status: str = "processing"
    page_count: Optional[int] = None

    # Chunk counts broken down by type (populated after ingestion)
    child_chunk_count: Optional[int] = None
    parent_chunk_count: Optional[int] = None

    # Legacy field kept for backwards-compat
    chunk_count: Optional[int] = None

    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Settings:
        name = "pdf_documents"
        indexes = [
            [("status", ASCENDING)],
            [("semester_id", ASCENDING)],
            [("module_id", ASCENDING)],
            [("lesson_id", ASCENDING)],
            [("uploaded_by", ASCENDING)],
        ]


# ── Text Embedding (child + parent chunks) ────────────────────────────────────


class TextEmbedding(Document):
    """
    Stores a single text chunk from a PDF — either a *child* or a *parent*.

    Parent-child structure
    ──────────────────────
    Each PDF section is split into:

      • One *parent* chunk (~600 tokens, broad context for the LLM)
      • N *child*  chunks (~150 tokens each, precise semantic matching)

    Only child chunks carry an embedding vector and are indexed by Atlas
    Vector Search.  When a child chunk wins retrieval, its parent is fetched
    to give the LLM fuller context without wasting the full chapter's tokens.

    Parent chunks are stored text-only (``embedding = []``) and are searched
    only through BM25 (Atlas Search full-text index on ``text``).

    Fields
    ──────
    chunk_type          "child" | "parent"
    parent_chunk_index  (child only) chunk_index of the sibling parent doc
    section_title       optional section header extracted from the document
    embedding           768-dim float list (children only; [] for parents)
    """

    pdf_id: PydanticObjectId

    # Hierarchical scoping — copied from PDFDocument for efficient Atlas filter
    semester_id: Optional[PydanticObjectId] = None
    module_id: Optional[PydanticObjectId] = None
    lesson_id: Optional[PydanticObjectId] = None

    # Position within the document
    chunk_index: int

    # Parent-child relationship
    chunk_type: Literal["child", "parent"] = "child"
    parent_chunk_index: Optional[int] = None  # set on child chunks only

    # Optional structural metadata
    section_title: Optional[str] = None

    # Content
    text: str
    embedding: List[float] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Settings:
        name = "text_embeddings"
        indexes = [
            # Primary lookup: all chunks for a PDF, unique per position
            IndexModel(
                [("pdf_id", ASCENDING), ("chunk_index", ASCENDING)],
            ),
            # Fetch the parent of a retrieved child chunk
            IndexModel(
                [
                    ("pdf_id", ASCENDING),
                    ("chunk_type", ASCENDING),
                    ("chunk_index", ASCENDING),
                ]
            ),
            # Scoped retrieval filters used by both BM25 and vector pipelines
            [("semester_id", ASCENDING)],
            [("module_id", ASCENDING)],
            [("lesson_id", ASCENDING)],
            # Fast filter: child-only for vector search aggregation
            [("chunk_type", ASCENDING)],
        ]

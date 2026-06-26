# FastAPI Integration & Usage Guide

This guide explains how to integrate and use the PRISM RAG (`rag_sys`) architecture within a FastAPI application. The system was engineered specifically for asynchronous execution in FastAPI, leveraging MongoDB (via Beanie ODM) and Pydantic for data validation.

---

## 1. Architecture Overview

The modules in this repository are designed to drop directly into a standard FastAPI project structure:

- **`rag.py`**: Beanie ODM models (`PDFDocument`, `TextEmbedding`, `RAGModule`).
- **`rag_service.py`**: Core retrieval service (Hybrid Search, RRF, Parent-Child expansion).
- **`rag_grader.py`**: Corrective RAG evaluation service.
- **`pdf_pipeline.py`**: Document ingestion and chunking pipeline.
- **`rag_studio.py`**: FastAPI `APIRouter` providing full REST endpoints for document management and query testing.

---

## 2. Prerequisites & Database Setup

PRISM RAG uses **Beanie ODM** with MongoDB Atlas. You must initialize Beanie with the RAG models during your FastAPI application startup (typically in the `lifespan` context manager).

### Example `main.py` Setup

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie

# Import the RAG models and the studio router
from app.models.rag import PDFDocument, TextEmbedding, RAGModule
from app.routers.rag_studio import router as rag_studio_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Initialize MongoDB Client
    client = AsyncIOMotorClient("mongodb+srv://<username>:<password>@cluster.mongodb.net/medcore?retryWrites=true&w=majority")
    db = client.get_database("medcore")
    
    # 2. Initialize Beanie ODM with PRISM RAG models
    await init_beanie(
        database=db,
        document_models=[PDFDocument, TextEmbedding, RAGModule]
    )
    yield
    # Cleanup logic here if needed

app = FastAPI(lifespan=lifespan, title="MedCore API")

# 3. Include the RAG Studio Router
app.include_router(rag_studio_router, prefix="/api/v1/rag", tags=["RAG Studio"])
```

---

## 3. Using the FastAPI Endpoints (`rag_studio.py`)

Once the router is included in your FastAPI application, the following key REST endpoints are available:

### Module Management
- `GET /api/v1/rag/modules`: List all available knowledge modules/domains.
- `POST /api/v1/rag/modules`: Create a new knowledge module (e.g., Cardiology, Pharmacology).
- `PATCH /api/v1/rag/modules/{module_slug}`: Update module configuration or system prompts.

### Document Ingestion
- `POST /api/v1/rag/modules/{module_slug}/documents`: Upload a PDF document.
  - **Background Processing**: FastAPI `BackgroundTasks` are automatically used to handle PDF text extraction, parent-child chunking, embedding generation, and vector indexing without blocking the HTTP response.
- `GET /api/v1/rag/modules/{module_slug}/documents`: List all documents within a module.
- `DELETE /api/v1/rag/modules/{module_slug}/documents/{doc_id}`: Remove a document and delete its vector embeddings.

### RAG Query Testing
- `POST /api/v1/rag/modules/{module_slug}/search`: Test the RAG retrieval and generation pipeline.
  - Accepts a query and returns the generated answer, retrieved parent chunks, confidence scores, and RRF fusion metrics.

---

## 4. Programmatic Usage in Other Services

You can directly import and utilize `rag_service.py` in your other FastAPI dependencies, background workers, or custom route handlers.

### Executing a RAG Query Programmatically

```python
from fastapi import APIRouter, HTTPException
from app.services.rag_service import query_rag

custom_router = APIRouter()

@custom_router.post("/ask-medical-question")
async def ask_question(question: str, module_slug: str):
    try:
        # Executes Hybrid Search (BM25 + Vector) -> RRF -> Corrective Grader -> Parent Fetch -> LLM Answer
        result = await query_rag(
            query=question,
            module_slug=module_slug,
            top_k=5
        )
        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "confidence": result["confidence_score"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

## 5. Environment Configuration

Ensure the following environment variables are set in your `.env` file for the services to function correctly:

```env
MONGODB_URI=mongodb+srv://<username>:<password>@cluster.mongodb.net/medcore?retryWrites=true&w=majority
OPENAI_API_KEY=your_openai_api_key_for_embeddings
DEEPSEEK_API_KEY=your_deepseek_api_key_for_llm
# Optional Redis Cache for RAG queries
REDIS_URL=redis://localhost:6379/0
```

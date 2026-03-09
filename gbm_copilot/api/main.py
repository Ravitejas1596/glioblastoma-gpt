"""
FastAPI REST API
Streaming /chat endpoint + health check + admin /ingest trigger.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from gbm_copilot.config import CORS_ORIGINS, LANGCHAIN_TRACING_V2, LANGCHAIN_PROJECT
from gbm_copilot.api.schemas import (
    ChatRequest, ChatResponse, IngestRequest, HealthResponse
)
from gbm_copilot.agents.graph import run_query

# Configure LangSmith if enabled
if LANGCHAIN_TRACING_V2:
    import os
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

app = FastAPI(
    title="GlioblastomaGPT API",
    description="GBM Research & Care Intelligence System — RAG + Multi-Agent",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    import gbm_copilot.config as cfg
    stats = {}
    try:
        if cfg.RETRIEVAL_MODE == "numpy":
            from gbm_copilot.embeddings import numpy_store
            stats = numpy_store.get_stats()
        else:
            from gbm_copilot.embeddings import pinecone_store
            stats = pinecone_store.get_stats()
    except Exception:
        pass
    return HealthResponse(
        status="ok",
        retrieval_mode=cfg.RETRIEVAL_MODE,
        index_stats=stats,
    )


@app.get("/config")
async def get_config():
    """Get current runtime configuration."""
    import gbm_copilot.config as cfg
    return {
        "retrieval_mode": cfg.RETRIEVAL_MODE,
        "openai_model": cfg.OPENAI_MODEL,
        "embedding_model": cfg.EMBEDDING_MODEL,
        "chunk_size_tokens": cfg.CHUNK_SIZE_TOKENS,
        "min_confidence_threshold": cfg.MIN_CONFIDENCE_THRESHOLD,
        "ingest_quick_mode": cfg.INGEST_QUICK_MODE,
    }


@app.post("/config")
async def set_config(body: dict):
    """
    Update runtime configuration (no restart required).
    Supported keys: retrieval_mode (pinecone|numpy)
    """
    import gbm_copilot.config as cfg
    updated = {}

    if "retrieval_mode" in body:
        mode = body["retrieval_mode"]
        if mode not in ("pinecone", "numpy"):
            raise HTTPException(status_code=400, detail="retrieval_mode must be 'pinecone' or 'numpy'")
        cfg.RETRIEVAL_MODE = mode  # type: ignore[assignment]
        # Reset cached index so next query uses new mode
        if mode == "numpy":
            from gbm_copilot.embeddings import numpy_store
            numpy_store._embeddings_matrix = None
            numpy_store._metadata_list = None
        updated["retrieval_mode"] = mode

    return {"status": "updated", "changes": updated}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint (non-streaming).
    Returns full response with citations and metadata.
    """
    try:
        session_id = request.session_id or str(uuid.uuid4())
        result = await run_query(
            query=request.query,
            literacy_mode=request.literacy_mode,
            session_id=session_id,
            conversation_history=request.conversation_history,
        )
        return ChatResponse(
            answer=result["final_answer"],
            query_type=result.get("query_type", ""),
            citations=result.get("citations", []),
            confidence_score=result.get("confidence_score", 0.0),
            safety_flags=result.get("safety_flags", []),
            is_blocked=result.get("is_blocked", False),
            trial_results=result.get("trial_results", []),
            session_id=session_id,
            expanded_query=result.get("expanded_query", ""),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint using Server-Sent Events.
    Streams the answer token by token.
    """
    session_id = request.session_id or str(uuid.uuid4())

    async def generate() -> AsyncGenerator[dict, None]:
        try:
            # Yield thinking indicator
            yield {
                "event": "thinking",
                "data": json.dumps({"status": "Processing your question..."}),
            }

            result = await run_query(
                query=request.query,
                literacy_mode=request.literacy_mode,
                session_id=session_id,
                conversation_history=request.conversation_history,
            )

            # Stream answer in chunks
            answer = result["final_answer"]
            words = answer.split(" ")
            chunk_size = 5
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i:i+chunk_size])
                if i + chunk_size < len(words):
                    chunk += " "
                yield {
                    "event": "token",
                    "data": json.dumps({"token": chunk}),
                }
                await asyncio.sleep(0.02)  # ~50 tokens/sec pacing

            # Final metadata event
            yield {
                "event": "done",
                "data": json.dumps({
                    "query_type": result.get("query_type"),
                    "citations": result.get("citations", []),
                    "confidence_score": result.get("confidence_score", 0.0),
                    "safety_flags": result.get("safety_flags", []),
                    "is_blocked": result.get("is_blocked", False),
                    "trial_results": result.get("trial_results", []),
                    "session_id": session_id,
                }),
            }
        except Exception as e:
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)}),
            }

    return EventSourceResponse(generate())


@app.get("/trials")
async def get_trials(
    location: str | None = None,
    phase: str | None = None,
    max_results: int = 20,
):
    """Direct clinical trial search endpoint."""
    from gbm_copilot.ingest.clinical_trials_fetcher import search_trials
    from gbm_copilot.ingest.clinical_trials_fetcher import RELEVANT_PHASES

    phases_filter = None
    if phase:
        phase_map = {"2": ["PHASE2"], "3": ["PHASE3"], "1": ["PHASE1"], "1/2": ["PHASE1_PHASE2"]}
        phases_filter = phase_map.get(phase, RELEVANT_PHASES)

    trials = await search_trials(
        location=location,
        phases=phases_filter,
        max_results=max_results,
    )
    return {"count": len(trials), "trials": trials}


@app.post("/ingest")
async def trigger_ingest(request: IngestRequest, background_tasks: BackgroundTasks):
    """
    Admin endpoint: trigger knowledge base ingestion.
    Runs asynchronously in background.
    """
    from gbm_copilot.ingest.ingest import run_ingest

    background_tasks.add_task(run_ingest, quick_mode=request.quick_mode)
    return {
        "status": "ingestion started",
        "quick_mode": request.quick_mode,
        "message": "Ingestion running in background. Check /health for index stats.",
    }

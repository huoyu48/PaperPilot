"""API routes for PaperPilot backend."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from fastapi import APIRouter, File, UploadFile

from src.agent.graph import create_agent
from src.backend.schemas import (
    DocumentUploadResponse,
    MemoryResponse,
    ResearchRequest,
    ResearchResponse,
    ResearchStatusResponse,
)
from src.memory.long_term import ResearchProfile
from src.memory.sessions import SessionStore
from src.rag.chunker import chunk_documents
from src.rag.parser import parse_document
from src.rag.vectorstore import ResearchVectorStore
from src.report.generator import generate_report_file
from src.utils.logging import logger

router = APIRouter()

_store = SessionStore()


@router.post("/research", response_model=ResearchResponse)
async def start_research(req: ResearchRequest):
    """Start a new research session or continue an existing one."""
    is_followup = bool(req.session_id)
    session_id = req.session_id or uuid.uuid4().hex[:12]
    logger.info(f"API: {'followup' if is_followup else 'new'} research session {session_id}")

    try:
        # If followup, load prior context
        prior_context = ""
        if is_followup:
            existing = _store.get_session(session_id)
            if existing:
                prior_context = (
                    f"Previous research question: {existing['query']}\n"
                    f"Previous report summary: {existing['report'][:800]}\n"
                    f"Follow-up question: {req.query}\n"
                )

        agent = create_agent()
        initial_state = {
            "query": prior_context + req.query if prior_context else req.query,
            "session_id": session_id,
            "sub_queries": [],
            "research_results": [],
            "synthesis": "",
            "report": "",
            "review_feedback": "",
            "needs_revision": False,
            "revision_count": 0,
            "messages": [],
            "errors": [],
        }

        final_state = await asyncio.to_thread(agent.invoke, initial_state)

        report_text = final_state.get("report", "")
        sub_queries = final_state.get("sub_queries", [])

        # Persist to session store
        if is_followup:
            _store.save_followup(session_id, req.query, report_text)
        else:
            _store.save_session(session_id, req.query, sub_queries, report_text)

        report_path = generate_report_file(
            report_text,
            final_state.get("research_results", []),
            session_id,
        )

        # Save to long-term memory profile
        profile = ResearchProfile()
        profile.add_topic(req.query, session_id)
        if final_state.get("synthesis"):
            profile.save_summary(session_id, req.query, final_state["synthesis"][:500], report_path)

        return ResearchResponse(
            session_id=session_id,
            status="completed",
            message=f"Research complete. Report saved to {report_path}",
        )

    except Exception as exc:
        logger.error(f"Research failed: {exc}")
        return ResearchResponse(session_id=session_id, status="failed", message=str(exc))


@router.get("/sessions")
async def list_sessions():
    """List all past research sessions."""
    return _store.list_sessions()


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get full details of a past research session including followups."""
    session = _store.get_session(session_id)
    if not session:
        return {"error": "Session not found"}
    return session


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session from history."""
    _store.delete_session(session_id)
    return {"status": "deleted"}


@router.post("/documents", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    session_id: str = "",
):
    """Upload a document to the RAG knowledge base."""
    try:
        upload_dir = Path("data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)

        filename = file.filename or "uploaded_file"
        filepath = upload_dir / filename
        content = await file.read()
        filepath.write_bytes(content)

        doc = parse_document(filepath)
        chunks = chunk_documents([doc])

        store = ResearchVectorStore()
        store.add_documents(chunks, session_id=session_id or "global")

        logger.info(f"API: uploaded {filename} → {len(chunks)} chunks")
        return DocumentUploadResponse(
            filename=filename,
            chunk_count=len(chunks),
            message=f"Document processed: {len(chunks)} chunks indexed",
        )
    except Exception as exc:
        logger.error(f"Upload failed: {exc}")
        return DocumentUploadResponse(
            filename=file.filename or "unknown",
            chunk_count=0,
            message=f"Upload failed: {str(exc)}",
        )


@router.get("/memory", response_model=MemoryResponse)
async def get_memory():
    """Get research profile: past topics and summaries."""
    profile = ResearchProfile()
    return MemoryResponse(
        topics=profile.get_topics(),
        summaries=profile.get_summaries(),
    )

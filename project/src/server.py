from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from contextlib import asynccontextmanager
except Exception as exc:
    import sys

    print("Missing required packages for server: fastapi/uvicorn/pydantic not installed.")
    print("Install them with: python -m pip install fastapi uvicorn pydantic")
    raise SystemExit(1) from exc

# Globals populated at startup
TRANSLATOR = None
EMBEDDING = None
RETRIEVER = None
VECTOR_DIR = None


class AnswerRequest(BaseModel):
    question: str
    top_k: int = 3
    max_tokens: int = 256
    # Optional per-document restriction: when set, retrieval only considers
    # chunks from that individual document (e.g. one PDF of a multi-PDF source).
    document_name: Optional[str] = None


class EvidenceItem(BaseModel):
    """One retrieved chunk that served as the RAG basis for the answer."""
    chunk_id: str
    similarity: float
    paragraph_index: int
    heading: str
    source_name: str
    source_url: str
    document_name: str = ""
    quotation: str
    language: str
    original_text: str


class AnswerResponse(BaseModel):
    answer: str
    cited_indices: List[int]
    language: str
    request_language: str
    timings: Dict[str, float]
    evidence: List[EvidenceItem]
    metadata: Optional[Dict[str, Any]] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to load models at startup and keep them in memory."""
    global TRANSLATOR, EMBEDDING, RETRIEVER, VECTOR_DIR

    start_total = time.perf_counter()
    project_root = Path(__file__).resolve().parent.parent

    # Ensure HF env is configured
    from dotenv import load_dotenv
    load_dotenv(project_root / ".env", override=True)
    if os.getenv("HF_TOKEN") and not os.getenv("HUGGINGFACE_HUB_TOKEN"):
        os.environ["HUGGINGFACE_HUB_TOKEN"] = os.getenv("HF_TOKEN")

    # Reduce tokenizer parallelism overhead
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    timings: Dict[str, float] = {}

    # Ensure project root is on sys.path
    import sys
    sys.path.insert(0, str(project_root))

    # Import heavy modules lazily
    t0 = time.perf_counter()
    from src.translator import TranslationModel
    timings["import_translator"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    from src.embedding import EmbeddingModel
    timings["import_embedding"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    from src.retriever import Retriever
    timings["import_retriever"] = time.perf_counter() - t0

    # Initialize models
    t0 = time.perf_counter()
    TRANSLATOR = TranslationModel()
    timings["translator_init"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    EMBEDDING = EmbeddingModel()
    timings["embedding_init"] = time.perf_counter() - t0

    # Sync the data/cache against its original sources before exposing the server.
    # Any source whose cached content is missing or differs from the live page is
    # rebuilt so that the vector DB, chunks and downstream caches stay consistent.
    t0 = time.perf_counter()
    VECTOR_DIR = project_root / "vector_db"
    from src.vectorstore import VectorStore
    from src.cache_sync import sync_all_sources

    vector_store = VectorStore(persist_directory=VECTOR_DIR)
    updated_sources = sync_all_sources(
        project_root=project_root,
        vector_store=vector_store,
        translator=TRANSLATOR,
        embedding_model=EMBEDDING,
    )
    timings["cache_sync"] = time.perf_counter() - t0
    if updated_sources:
        print(f"Updated {len(updated_sources)} source(s) on startup: {updated_sources}")
    else:
        print("All cached sources are up to date.")

    t0 = time.perf_counter()
    RETRIEVER = Retriever(vector_dir=VECTOR_DIR, translator=TRANSLATOR, embedding_model=EMBEDDING)
    timings["retriever_init"] = time.perf_counter() - t0

    timings["total_startup"] = time.perf_counter() - start_total

    # Persist startup timing file
    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "server_startup_timings.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")

    try:
        yield
    finally:
        # Cleanup if needed
        pass


# Create the FastAPI app with lifespan
app = FastAPI(title="RAG Model Server", lifespan=lifespan)


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok", "models_loaded": bool(TRANSLATOR and EMBEDDING and RETRIEVER)}


def _extract_cited_indices(answer: str) -> List[int]:
    import re
    cited = re.findall(r"\[(\d+)\]", answer)
    return [int(index) for index in cited]


async def _run_blocking(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


class RebuildRequest(BaseModel):
    chunk_max_tokens: int = 800
    rebuild: bool = True


class RebuildResponse(BaseModel):
    status: str
    timings: Dict[str, float]
    processed_sources: List[str]


@app.post("/rebuild", response_model=RebuildResponse)
async def rebuild(req: RebuildRequest) -> RebuildResponse:
    if not TRANSLATOR or not EMBEDDING:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    if getattr(app.state, "rebuild_in_progress", False):
        raise HTTPException(status_code=409, detail="Rebuild already in progress")

    app.state.rebuild_in_progress = True
    try:
        from src.main import _load_sources, process_sources

        project_root = Path(__file__).resolve().parent.parent
        source_list = _load_sources(project_root)
        vector_dir = project_root / "vector_db"

        t0 = time.perf_counter()
        processed_state = await _run_blocking(
            process_sources,
            project_root,
            source_list,
            vector_dir,
            TRANSLATOR,
            EMBEDDING,
            req.chunk_max_tokens,
            req.rebuild,
        )
        total_time = time.perf_counter() - t0

        # Reinitialize retriever
        from src.retriever import Retriever

        global RETRIEVER
        RETRIEVER = Retriever(vector_dir=vector_dir, translator=TRANSLATOR, embedding_model=EMBEDDING)

        processed_sources = list(processed_state.keys())
        return RebuildResponse(status="rebuild_complete", timings={"rebuild_total": total_time}, processed_sources=processed_sources)
    finally:
        app.state.rebuild_in_progress = False


@app.get("/reload-status")
async def reload_status() -> Dict[str, Any]:
    """Check if per-source processed chunk files exist."""
    project_root = Path(__file__).resolve().parent.parent
    processed_dir = project_root / "data" / "processed"
    chunks_files = sorted(processed_dir.glob("*_chunks.json")) if processed_dir.exists() else []
    chunks_file = str(chunks_files[0]) if chunks_files else str(processed_dir / "chunks.json")

    return {
        "chunks_file": chunks_file,
        "file_exists": bool(chunks_files),
        "mode": "direct_file_access",
    }


@app.post("/answer", response_model=AnswerResponse)
async def answer(req: AnswerRequest) -> AnswerResponse:
    if not TRANSLATOR or not EMBEDDING or not RETRIEVER:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    timings: Dict[str, float] = {}
    t0 = time.perf_counter()

    # Retrieval happens exactly once. The same ``results`` are used to build the
    # LLM prompt AND to return the RAG basis to the caller, so the answer and
    # its evidence always stay consistent (no re-retrieval or drift).
    retrieval_start = time.perf_counter()
    results = await _run_blocking(
        RETRIEVER.retrieve, req.question, req.top_k, req.document_name
    )
    timings["retrieval"] = time.perf_counter() - retrieval_start

    # The LLM answer is generated from those exact retrieved chunks. ``answer()``
    # reuses them (it does not re-run retrieval) so the evidence below matches
    # precisely the RAG chunks used for the judgment.
    llm_start = time.perf_counter()
    answer_text, used_results, answer_language = await _run_blocking(
        RETRIEVER.answer, req.question, req.top_k, req.max_tokens, results,
    )
    timings["llm_generation"] = time.perf_counter() - llm_start

    timings["total"] = time.perf_counter() - t0

    cited_indices = _extract_cited_indices(answer_text)

    # Build the RAG evidence list: exactly the chunks used to answer, each with
    # its chunk_id, similarity and the quotation (with its language) actually
    # fed to the model. This is what the answerer used to make its judgment.
    evidence: List[EvidenceItem] = []
    for r in (used_results or []):
        evidence.append(
            EvidenceItem(
                chunk_id=r.chunk_id,
                similarity=round(float(r.similarity), 6),
                paragraph_index=r.paragraph_index,
                heading=r.heading,
                source_name=r.source_name,
                source_url=r.source_url,
                document_name=r.document_name,
                quotation=r.quotation,
                language=r.quotation_language,
                original_text=r.original_paragraph,
            )
        )

    # Serialize used_results so clients can correlate 1-based citations with
    # the RAG basis + chunk ids (chunk_id is already read from vector metadata).
    results_serialized = None
    try:
        if used_results:
            from src.main import _safe_name_for_source

            results_serialized = []
            for idx, r in enumerate(used_results, start=1):
                res_dict = asdict(r)
                # The processed filename always carries the source's safe prefix.
                res_dict.setdefault(
                    "processed_file",
                    "data/processed/%s_chunks.json"
                    % _safe_name_for_source(r.source_name or "", r.source_url or ""),
                )
                res_dict["rank"] = idx
                results_serialized.append(res_dict)
    except Exception as exc:
        print(f"Error serializing results: {exc}")
        results_serialized = None

    metadata = {
        "results": results_serialized,
        "query_language": answer_language,
        "request_language": answer_language,
    } if results_serialized is not None else None

    return AnswerResponse(
        answer=answer_text,
        cited_indices=cited_indices,
        language=answer_language,
        request_language=answer_language,
        timings=timings,
        evidence=evidence,
        metadata=metadata,
    )


def _run_server():
    """Entry point for running the server."""
    import uvicorn

    port = int(os.environ.get("RAG_SERVER_PORT", "8000"))
    uvicorn.run("src.server:app", host="127.0.0.1", port=port, log_level="info")


if __name__ == "__main__":
    _run_server()
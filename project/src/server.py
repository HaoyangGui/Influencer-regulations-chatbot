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
    preview_tokens: bool = False


class AnswerResponse(BaseModel):
    answer: str
    cited_indices: List[int]
    timings: Dict[str, float]
    metadata: Optional[Dict[str, Any]] = None


def _load_chunks_from_file(chunks_path: Path, source_url: str, paragraph_index: int) -> Optional[dict]:
    """Load chunk info directly from chunks.json file.
    
    Args:
        chunks_path: Path to chunks.json
        source_url: Source URL to look up
        paragraph_index: Paragraph index to look up
        
    Returns:
        Dict with chunk_id, source_name, processed_file or None if not found
    """
    if not chunks_path.exists():
        return None
    
    try:
        with chunks_path.open("r", encoding="utf-8") as f:
            chunks_data = json.load(f)
        
        # Find the chunk matching source_url and paragraph_index
        for chunk in chunks_data:
            if (chunk.get("source_url") == source_url and 
                chunk.get("paragraph_index") == paragraph_index):
                return {
                    "chunk_id": chunk.get("chunk_id", ""),
                    "source_name": chunk.get("source_name", ""),
                    "processed_file": str(chunks_path.relative_to(Path(__file__).resolve().parent.parent))
                }
    except Exception as exc:
        print(f"Error loading chunks from file: {exc}")
    
    return None


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

    t0 = time.perf_counter()
    VECTOR_DIR = project_root / "vector_db"
    RETRIEVER = Retriever(vector_dir=VECTOR_DIR, translator=TRANSLATOR, embedding_model=EMBEDDING)
    timings["retriever_init"] = time.perf_counter() - t0

    timings["total_startup"] = time.perf_counter() - start_total

    # Persist startup timing file
    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "server_startup_timings.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")

    app.state.timings = timings

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
    """Check if chunks.json exists."""
    project_root = Path(__file__).resolve().parent.parent
    chunks_path = project_root / "data" / "processed" / "chunks.json"
    
    return {
        "chunks_file": str(chunks_path),
        "file_exists": chunks_path.exists(),
        "mode": "direct_file_access",
    }


@app.post("/answer", response_model=AnswerResponse)
async def answer(req: AnswerRequest) -> AnswerResponse:
    if not TRANSLATOR or not EMBEDDING or not RETRIEVER:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    timings: Dict[str, float] = {}
    t0 = time.perf_counter()

    # Retrieval
    retrieval_start = time.perf_counter()
    results = await _run_blocking(RETRIEVER.retrieve, req.question, req.top_k)
    timings["retrieval"] = time.perf_counter() - retrieval_start

    # Build prompt
    prompt_build_start = time.perf_counter()
    prompt = await _run_blocking(RETRIEVER.build_prompt, req.question, results)
    timings["prompt_build"] = time.perf_counter() - prompt_build_start

    # Call LLM
    llm_start = time.perf_counter()
    answer_text, used_results = await _run_blocking(RETRIEVER.answer, req.question, req.top_k, req.max_tokens)
    timings["llm_generation"] = time.perf_counter() - llm_start

    timings["total"] = time.perf_counter() - t0

    cited_indices = _extract_cited_indices(answer_text)

    # Serialize used_results and load chunk info directly from file
    project_root = Path(__file__).resolve().parent.parent
    chunks_path = project_root / "data" / "processed" / "chunks.json"
    
    results_serialized = None
    try:
        if used_results:
            results_serialized = []
            for r in used_results:
                res_dict = asdict(r)
                
                # Load chunk info directly from file for each result
                source_url = res_dict.get('source_url', '')
                paragraph_index = res_dict.get('paragraph_index')
                
                if source_url and paragraph_index is not None:
                    chunk_info = _load_chunks_from_file(chunks_path, source_url, paragraph_index)
                    if chunk_info:
                        res_dict['chunk_id'] = chunk_info['chunk_id']
                        res_dict['source_name'] = chunk_info['source_name']
                        res_dict['processed_file'] = chunk_info['processed_file']
                
                # Ensure defaults if not found
                if not res_dict.get('processed_file'):
                    res_dict['processed_file'] = 'data/processed/chunks.json'
                if not res_dict.get('source_name'):
                    res_dict['source_name'] = 'De belangrijkste regels voor video-uploaders'
                
                results_serialized.append(res_dict)
    except Exception as exc:
        print(f"Error serializing results: {exc}")
        results_serialized = None

    metadata = {"results": results_serialized} if results_serialized is not None else None

    return AnswerResponse(answer=answer_text, cited_indices=cited_indices, timings=timings, metadata=metadata)


def _run_server():
    """Entry point for running the server."""
    import uvicorn

    port = int(os.environ.get("RAG_SERVER_PORT", "8000"))
    uvicorn.run("src.server:app", host="127.0.0.1", port=port, log_level="info")


if __name__ == "__main__":
    _run_server()
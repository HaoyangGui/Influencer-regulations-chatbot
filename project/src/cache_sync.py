"""Cache synchronization for the RAG server.

On server startup, this module verifies that the cached content stored in
``data/cache`` (e.g. the raw markdown fetched from each source page) is still
consistent with the live, original file it was derived from. When a cached
file is missing or differs from its original, the affected source is rebuilt
so that the vector database, chunks and downstream caches stay up to date.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import List, Optional

from src.vectorstore import VectorStore


def _fetch_live_content(url: str) -> Optional[str]:
    """Download and clean the live webpage content.

    Returns None if the page cannot be fetched (e.g. no internet access).
    """
    try:
        from src.crawler import clean_html_to_markdown, download_page

        html = download_page(url)
        return clean_html_to_markdown(html)
    except Exception as exc:
        print(f"Warning: could not fetch {url}: {exc}")
        return None


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _resolve_cached_paths(
    project_root: Path, source_url: str, source_name: str
) -> tuple[Path, Path]:
    """Resolve the raw/chunks cache paths, preferring recorded source state.

    The paths written into ``source_state.json`` are authoritative (they match
    the files actually on disk), falling back to the freshly computed naming
    convention from ``main._source_cache_paths`` when no state is recorded.
    """
    from src.main import _load_source_state, _source_cache_paths

    entry = _load_source_state(project_root).get(source_url) or {}
    raw_path = Path(entry.get("raw_path") or "")
    chunks_path = Path(entry.get("chunks_path") or "")
    if raw_path.name and chunks_path.name:
        return raw_path, chunks_path
    return _source_cache_paths(project_root, source_url, source_name)


def _cached_content_hash(raw_path: Path) -> Optional[str]:
    """Return the sha256 of the cached raw markdown file, or None if missing."""
    if not raw_path.exists():
        return None
    try:
        return _content_hash(raw_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def is_source_fresh(project_root: Path, source_url: str, source_name: str) -> bool:
    """Check whether the cached raw markdown matches the live webpage.

    A source is considered fresh when:
      - a cached raw markdown file exists, AND
      - its content hash equals the current live page content hash.

    If the live page cannot be fetched, the source is assumed fresh to avoid
    taking the server down just because the network is unreachable.
    """
    raw_path, _ = _resolve_cached_paths(project_root, source_url, source_name)
    cached_hash = _cached_content_hash(raw_path)
    if cached_hash is None:
        return False

    live_content = _fetch_live_content(source_url)
    if live_content is None:
        print(f"  (offline check) assuming cached source is fresh: {source_url}")
        return True

    live_hash = _content_hash(live_content)
    return cached_hash == live_hash


def rebuild_source(
    project_root: Path,
    source_url: str,
    source_name: str,
    vector_store: VectorStore,
    translator=None,
    embedding_model=None,
    chunk_max_tokens: int = 800,
) -> None:
    """Re-download and rebuild a single source, replacing its cached files."""
    from src.main import _load_source_state, _write_source_state, build_vector_database

    raw_path, chunks_path = _resolve_cached_paths(project_root, source_url, source_name)

    build_vector_database(
        source_url=source_url,
        source_name=source_name,
        raw_path=raw_path,
        chunks_path=chunks_path,
        vector_dir=vector_store.persist_directory,
        translator=translator,
        embedding_model=embedding_model,
        chunk_max_tokens=chunk_max_tokens,
        vector_store=vector_store,
        project_root=project_root,
    )

    # Update the persisted source state with the new paths and content hash.
    state = _load_source_state(project_root)
    state[source_url] = {
        "source_name": source_name,
        "raw_path": str(raw_path),
        "chunks_path": str(chunks_path),
        "content_hash": _cached_content_hash(raw_path) or "",
        "updated_at": time.time(),
    }
    _write_source_state(project_root, state)


def _update_processed_chunks(project_root: Path) -> None:
    """Aggregate all per-source chunk files into ``data/processed/chunks.json``.

    The server's result serialization reads from this single file, so it must
    always reflect the current cached chunks for every configured source.
    """
    from src.main import _load_sources

    sources = _load_sources(project_root)
    all_chunks: List[dict] = []

    for source in sources:
        source_url = str(source.get("url", "")).strip()
        if not source_url:
            continue
        source_name = str(source.get("source_name") or source.get("title") or "Unknown source")
        _, chunks_path = _resolve_cached_paths(project_root, source_url, source_name)
        if chunks_path.exists():
            try:
                with chunks_path.open("r", encoding="utf-8") as handle:
                    chunks = json.load(handle)
                if isinstance(chunks, list):
                    all_chunks.extend(chunks)
            except Exception as exc:
                print(f"Warning: could not read {chunks_path}: {exc}")

    processed_path = project_root / "data" / "processed" / "chunks.json"
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    with processed_path.open("w", encoding="utf-8") as handle:
        json.dump(all_chunks, handle, indent=2, ensure_ascii=False)


def sync_all_sources(
    project_root: Path,
    vector_store: VectorStore,
    translator=None,
    embedding_model=None,
    chunk_max_tokens: int = 800,
) -> List[str]:
    """Verify every configured source against its original and rebuild stale ones.

    Returns the list of source URLs that were rebuilt.
    """
    from src.main import _load_sources

    sources = _load_sources(project_root)
    updated: List[str] = []

    # Detect an empty collection or an embedding-model / embedding-field change:
    # if the stored vectors were produced by a different model (or on a different
    # text field), they are incompatible with the current multilingual embeddings,
    # so the whole collection is cleared and every source re-embedded.
    force_rebuild = False
    if embedding_model is not None:
        if not vector_store.has_documents():
            print("Vector collection is empty; building all sources.")
            force_rebuild = True
        else:
            current_embed = vector_store.get_collection_metadata().get("embedding_model")
            expected_embed = getattr(embedding_model, "model_name", None)
            if current_embed != expected_embed:
                print(f"Embedding model changed ({current_embed!r} -> {expected_embed!r}); "
                      "clearing vector store for a full re-embed.")
                vector_store.clear()
                force_rebuild = True

    for source in sources:
        source_url = str(source.get("url", "")).strip()
        if not source_url:
            continue
        source_name = str(source.get("source_name") or source.get("title") or "Unknown source")

        if not force_rebuild and is_source_fresh(project_root, source_url, source_name):
            print(f"Cache is up to date for source: {source_url}")
            continue

        print(f"Source cache is out of date, rebuilding: {source_url}")
        rebuild_source(
            project_root=project_root,
            source_url=source_url,
            source_name=source_name,
            vector_store=vector_store,
            translator=translator,
            embedding_model=embedding_model,
            chunk_max_tokens=chunk_max_tokens,
        )
        updated.append(source_url)

    # Ensure the aggregated chunks file the server reads from is up to date.
    _update_processed_chunks(project_root)

    return updated
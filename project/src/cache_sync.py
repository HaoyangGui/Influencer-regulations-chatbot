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

    The paths written into ``source_state.json`` are authoritative only when they
    already identify their source by name (matching ``main._safe_name_for_source``).
    Legacy entries recorded under generic names (``raw.md`` / ``chunks.json``)
    are ignored so rebuilds use the source-prefixed canonical naming instead,
    guaranteeing every cached file visibly maps back to its source.
    """
    from src.main import _load_source_state, _source_cache_paths, _safe_name_for_source

    entry = _load_source_state(project_root).get(source_url) or {}
    raw_path = Path(entry.get("raw_path") or "")
    chunks_path = Path(entry.get("chunks_path") or "")
    safe = _safe_name_for_source(source_name, source_url)
    raw_name_ok = raw_path.name.startswith(f"{safe}.") or raw_path.name.startswith(f"{safe}_")
    chunks_name_ok = chunks_path.name.startswith(f"{safe}_")
    if raw_path.name and chunks_path.name and raw_name_ok and chunks_name_ok:
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


def is_source_fresh(
    project_root: Path, source_url: str, source_name: str, source_type: str = "website"
) -> bool:
    """Check whether the cached raw markdown matches the live source.

    For ``website`` sources, a source is considered fresh when:
      - a cached raw markdown file exists, AND
      - its content hash equals the current live page content hash.

    If the live page cannot be fetched, the source is assumed fresh to avoid
    taking the server down just because the network is unreachable.

    For ``pdf`` sources the OCR output is not regenerated on every startup
    (OCR via the Mistral API is expensive). Instead the source is considered
    fresh whenever a non-empty cleaned raw markdown file already exists on
    disk, so a previously processed local PDF set is not re-OCRed.
    """
    raw_path, _ = _resolve_cached_paths(project_root, source_url, source_name)
    cached_hash = _cached_content_hash(raw_path)
    if source_type == "pdf":
        return cached_hash is not None

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
    source_type: str = "website",
) -> None:
    """Re-download and rebuild a single source, replacing its cached files."""
    import shutil

    from src.main import (
        _load_source_state,
        _write_source_state,
        _safe_name_for_source,
        _with_source_prefix,
        build_vector_database,
    )

    raw_path, chunks_path = _resolve_cached_paths(project_root, source_url, source_name)

    build_vector_database(
        source_url=source_url,
        source_name=source_name,
        source_type=source_type,
        raw_path=raw_path,
        chunks_path=chunks_path,
        vector_dir=vector_store.persist_directory,
        translator=translator,
        embedding_model=embedding_model,
        chunk_max_tokens=chunk_max_tokens,
        vector_store=vector_store,
        project_root=project_root,
    )

    # Mirror the freshly built raw content into data/raw/ for human inspection.
    # This runs only when the source was actually rebuilt (i.e. the cached raw
    # was missing or differed), so data/raw stays in sync with the authoritative
    # cache without adding per-startup work.
    try:
        safe_name = _safe_name_for_source(source_name, source_url)
        raw_mirror_dir = project_root / "data" / "raw"
        raw_mirror_dir.mkdir(parents=True, exist_ok=True)
        raw_copy_path = raw_mirror_dir / f"{safe_name}.md"
        # PDF sources keep one markdown file per PDF next to the combined
        # source-level raw file; mirror each individually so data/raw holds one
        # .md per PDF. Website sources mirror the single source-level markdown.
        per_pdf_files = sorted(
            p for p in raw_path.parent.glob("*.md") if p.resolve() != raw_path.resolve()
        ) if source_type == "pdf" else []
        if per_pdf_files:
            for pdf_md in per_pdf_files:
                shutil.copy2(pdf_md, raw_mirror_dir / _with_source_prefix(safe_name, pdf_md.name))
        else:
            shutil.copy2(raw_path, raw_copy_path)
    except Exception as exc:
        print(f"Warning: could not mirror raw file to data/raw for {source_url}: {exc}")

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

    # Remove legacy generic-named cache files (raw.md / chunks.json) that do not
    # identify their source, so the source's cache dir holds only source-prefixed
    # names after the rebuild.
    for legacy in (raw_path.parent / "raw.md", raw_path.parent / "chunks.json"):
        try:
            if (
                legacy.exists()
                and legacy != raw_path
                and legacy != chunks_path
            ):
                legacy.unlink()
        except Exception:
            pass


def _update_processed_chunks(project_root: Path) -> None:
    """Mirror each source's chunk files into ``data/processed``.

    Every source gets a source-prefixed aggregate chunk file
    (``<safe>_chunks.json``) plus, for PDF sources, one per-document
    ``<safe>_<document>_chunks.json`` — each named after its source just like
    the authoritative files in ``data/cache/sources``.
    """
    from src.main import _load_sources, _safe_name_for_source, _mirror_source_chunks

    sources = _load_sources(project_root)
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    for source in sources:
        source_url = str(source.get("url", "")).strip()
        if not source_url:
            continue
        source_name = str(source.get("source_name") or source.get("title") or "Unknown source")
        _, chunks_path = _resolve_cached_paths(project_root, source_url, source_name)
        safe = _safe_name_for_source(source_name, source_url)
        _mirror_source_chunks(processed_dir, safe, chunks_path)


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
        source_type = str(source.get("type") or "website").strip().lower()

        if not force_rebuild and is_source_fresh(project_root, source_url, source_name, source_type):
            print(f"Cache is up to date for source: {source_url}")
            continue

        print(f"Source cache is out of date, rebuilding: {source_url}")
        rebuild_source(
            project_root=project_root,
            source_url=source_url,
            source_name=source_name,
            source_type=source_type,
            vector_store=vector_store,
            translator=translator,
            embedding_model=embedding_model,
            chunk_max_tokens=chunk_max_tokens,
        )
        updated.append(source_url)

    # Ensure the aggregated chunks file the server reads from is up to date.
    _update_processed_chunks(project_root)

    return updated
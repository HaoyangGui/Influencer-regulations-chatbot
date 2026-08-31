from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

# Ensure the project root is on the Python path when the script is executed from src/
# or imported as a module, so that ``src.*`` imports resolve correctly below.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunker import chunk_document, save_chunks_to_json
from src.crawler import (
    download_and_save_markdown,
    iter_local_pdfs,
    ocr_pdfs_to_sections,
)
from src.vectorstore import VectorStore

# Heavy modules (transformers, sentence_transformers, torch) are imported later to avoid long startup delays
# They are imported lazily after environment variables are printed so the user sees quick feedback.
# Actual imports occur below, after _configure_hf_environment is applied.


URL = "https://www.cvdm.nl/sector/video-uploaders/de-belangrijkste-regels-voor-video-uploaders/"
SOURCE_NAME = "De belangrijkste regels voor video-uploaders"


def _source_manifest_path(project_root: Path) -> Path:
    return project_root / "data" / "sources.json"


def _source_state_path(project_root: Path) -> Path:
    return project_root / "data" / "cache" / "source_state.json"


def _default_sources() -> List[Dict[str, str]]:
    return [{"url": URL, "source_name": SOURCE_NAME, "type": "website"}]


def _load_sources(project_root: Path) -> List[Dict[str, str]]:
    manifest_path = _source_manifest_path(project_root)
    if manifest_path.exists():
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, list):
                normalized: List[Dict[str, str]] = []
                for item in payload:
                    if isinstance(item, dict):
                        url = str(item.get("url", "")).strip()
                        if url:
                            source_type = str(item.get("type") or "website").strip().lower()
                            normalized.append(
                                {
                                    "url": url,
                                    "source_name": str(item.get("source_name") or item.get("title") or "Unknown source"),
                                    "type": source_type if source_type in {"website", "pdf"} else "website",
                                }
                            )
                if normalized:
                    return normalized
        except Exception:
            pass

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(_default_sources(), indent=2, ensure_ascii=False), encoding="utf-8")
    return _default_sources()


def _load_source_state(project_root: Path) -> Dict[str, Dict[str, str]]:
    state_path = _source_state_path(project_root)
    if not state_path.exists():
        return {}
    try:
        with state_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            return {str(k): {str(sk): str(sv) for sk, sv in v.items()} for k, v in payload.items() if isinstance(v, dict)}
    except Exception:
        return {}
    return {}


def _write_source_state(project_root: Path, state: Dict[str, Dict[str, str]]) -> None:
    state_path = _source_state_path(project_root)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, ensure_ascii=False)


def _source_cache_dir(project_root: Path, source_url: str) -> Path:
    source_hash = hashlib.sha256(source_url.encode("utf-8")).hexdigest()[:16]
    cache_dir = project_root / "data" / "cache" / "sources" / source_hash
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _safe_name_for_source(source_name: str, source_url: str) -> str:
    # Create a filesystem-safe name for the source using the provided source_name or fallback to hostname
    import re
    from urllib.parse import urlparse

    name = (source_name or '').strip()
    if not name:
        try:
            parsed = urlparse(source_url)
            name = parsed.netloc or parsed.path or 'source'
        except Exception:
            name = 'source'
    # Keep alphanum, dash and underscore
    safe = re.sub(r"[^A-Za-z0-9\-_]", "_", name)
    safe = re.sub(r"_+", "_", safe).strip("_")
    if not safe:
        safe = "source"
    return safe[:80]


def _source_cache_paths(project_root: Path, source_url: str, source_name: str | None = None) -> tuple[Path, Path]:
    cache_dir = _source_cache_dir(project_root, source_url)
    safe = _safe_name_for_source(source_name or '', source_url)
    raw_name = f"{safe}.md"
    chunks_name = f"{safe}_chunks.json"
    return cache_dir / raw_name, cache_dir / chunks_name


def _safe_file_stem(stem: str) -> str:
    """Sanitize a PDF stem so it can be used as a markdown filename."""
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", stem).strip()
    return cleaned or "document"


def _per_document_basename(source_name: str, source_url: str, document_stem: str) -> str:
    """Filesystem-safe, source-prefixed basename for a per-document (per-PDF) file.

    Prefixes a document stem with the source's safe name so every cached and
    mirrored raw/chunk file immediately shows which source it belongs to, e.g.
    ``Influencer_Legal_Hub_Legal brief 1``. Callers append a ``.md`` or
    ``_chunks.json`` extension.
    """
    return f"{_safe_name_for_source(source_name, source_url)}_{_safe_file_stem(document_stem)}"


def _with_source_prefix(safe_src: str, name: str) -> str:
    """Prefix ``name`` with the source safe name unless already prefixed.

    Helps migrate legacy cache/mirror files (named only by document, e.g.
    ``Legal brief 1.md``) to source-prefixed names without doubling the prefix
    for newly built files that already carry it.
    """
    if name.startswith(f"{safe_src}_"):
        return name
    return f"{safe_src}_{name}"


def _resolve_pdf_source_dir(project_root: Path, source: Dict[str, str]) -> Path:
    """Locate the local folder holding the PDFs for a ``"type": "pdf"`` source.

    The PDFs are NOT downloaded from the web anymore: they live under the
    The PDFs are NOT downloaded from the web anymore: they live under the
    project's ``data/pdf`` directory. Resolution order:
      1. A relative sub-folder name in ``source["url"]`` matched under ``data/pdf``.
      2. ``data/pdf/<source_name>`` (case-insensitive folder match).
      3. ``data/pdf`` itself (all local PDFs recursively).
    """
    pdf_root = project_root / "data" / "pdf"
    url = str(source.get("url", "") or "").strip()
    source_name = str(source.get("source_name") or source.get("title") or "").strip()

    # 1) The url may already point at a local sub-folder under data/pdf.
    if url and not url.startswith(("http://", "https://")):
        folder = Path(url.replace("\\", "/"))
        if not folder.is_absolute():
            folder = pdf_root / folder
        if folder.is_dir():
            return folder

    # 2) A folder whose name matches the source name (case-insensitive).
    if source_name:
        for child in (pdf_root.iterdir() if pdf_root.is_dir() else []):
            if child.is_dir() and child.name.lower() == source_name.lower():
                return child

    # 3) Fallback: the whole data/pdf tree.
    return pdf_root


def _make_chunk_id(source_url: str, paragraph_index: int) -> str:
    source_hash = hashlib.sha256(source_url.encode("utf-8")).hexdigest()[:12]
    return f"{source_hash}-chunk-{paragraph_index}"


def build_vector_database(
    source_url: str,
    source_name: str,
    raw_path: Path,
    chunks_path: Path,
    vector_dir: Path,
    translator: TranslationModel | None = None,
    embedding_model: EmbeddingModel | None = None,
    chunk_max_tokens: int = 800,
    vector_store: VectorStore | None = None,
    project_root: Path | None = None,
    source_type: str = "website",
) -> None:
    print(f"Processing source: {source_url} (type={source_type})")
    start = time.perf_counter()
    if source_type == "pdf":
        # PDF sources are read from the local data/pdf folder (no downloading).
        project_root = project_root or Path(__file__).resolve().parent.parent
        pdf_dir = _resolve_pdf_source_dir(project_root, {"url": source_url, "source_name": source_name})
        pdf_paths = iter_local_pdfs(pdf_dir)
        if not pdf_paths:
            print(f"Warning: no local PDFs found under {pdf_dir}; skipping {source_url}.")
            return
        print(f"OCR {len(pdf_paths)} PDF(s) from {pdf_dir}...")
        per_pdf_sections = ocr_pdfs_to_sections(pdf_paths)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        # Each PDF is saved to its own markdown file next to the combined
        # source-level raw file, so data/raw can mirror one .md per PDF.
        combined_parts: List[str] = []
        for stem, cleaned in per_pdf_sections:
            per_pdf_path = raw_path.parent / f"{_per_document_basename(source_name, source_url, stem)}.md"
            per_pdf_path.write_text(cleaned, encoding="utf-8")
            combined_parts.append(f"# {stem}\n\n{cleaned}")
        raw_path.write_text("\n\n".join(combined_parts).strip(), encoding="utf-8")
    else:
        per_pdf_sections = None
        download_and_save_markdown(source_url, raw_path)
    print(f"Source ready in {time.perf_counter() - start:.2f}s")

    markdown_text = raw_path.read_text(encoding="utf-8")
    print("Chunking document...")
    start = time.perf_counter()
    if per_pdf_sections is not None and source_type == "pdf":
        # Chunk each PDF separately so that every PDF becomes its own
        # processed chunk set: each chunk records which PDF it came from via
        # ``document_name`` and a per-PDF chunk JSON file is stored next to
        # its per-PDF raw markdown, letting the RAG system query any single
        # document individually.
        chunks: List = []
        for stem, cleaned in per_pdf_sections:
            safe_stem = _safe_file_stem(stem)
            doc_chunks = chunk_document(
                cleaned,
                source_name=source_name,
                source_url=source_url,
                max_tokens=chunk_max_tokens,
                document_name=stem,
            )
            # Make chunk ids unique across PDFs within this source (the
            # generic source-hash prefix below would otherwise collide).
            doc_tag = re.sub(r"[^A-Za-z0-9_-]+", "-", safe_stem).strip("-").lower() or "doc"
            for doc_chunk in doc_chunks:
                doc_chunk.chunk_id = f"{doc_tag}-{doc_chunk.chunk_id}"
            chunks.extend(doc_chunks)
            save_chunks_to_json(
                doc_chunks,
                raw_path.parent / f"{_per_document_basename(source_name, source_url, stem)}_chunks.json",
            )
    else:
        chunks = chunk_document(
            markdown_text,
            source_name=source_name,
            source_url=source_url,
            max_tokens=chunk_max_tokens,
        )
    # Ensure chunk ids are unique across sources by prefixing with a short source hash
    source_hash = hashlib.sha256(source_url.encode("utf-8")).hexdigest()[:12]
    for chunk in chunks:
        chunk.chunk_id = f"{source_hash}-{chunk.chunk_id}"
    save_chunks_to_json(chunks, chunks_path)
    print(f"Chunking finished in {time.perf_counter() - start:.2f}s")

    print("Filling chunk language metadata (English translations are generated lazily at query time)...")
    start = time.perf_counter()
    from src.translator import detect_language

    # Embeddings are computed on the ORIGINAL text (multilingual model), so no
    # translation is required for retrieval. Startup therefore only records the
    # source language; any English quotation bridge is translated lazily on
    # demand by the retriever (cached on disk), removing the slowest startup
    # step entirely.
    source_lang = detect_language(markdown_text) if markdown_text else "nl"
    for chunk in chunks:
        chunk.source_language = source_lang
        chunk.translations[source_lang] = chunk.original_paragraph
        if source_lang == "en":
            # Source already English: nothing to translate.
            chunk.translated_paragraph = chunk.original_paragraph
            chunk.embedding_text_en = chunk.embedding_text
            chunk.translations["en"] = chunk.original_paragraph
    print(f"Language metadata finished in {time.perf_counter() - start:.2f}s")

    print("Generating embeddings...")
    start = time.perf_counter()
    from src.embedding import EmbeddingModel

    embedding_model = embedding_model or EmbeddingModel()
    # Embed the original (source-language) text with the multilingual model so
    # queries in any language map to the same vector space. The English
    # translation is still kept in the chunk metadata for cross-lingual quotes.
    embeddings = embedding_model.embed_texts([chunk.embedding_text for chunk in chunks])
    print(f"Embedding finished in {time.perf_counter() - start:.2f}s")

    save_chunks_to_json(chunks, chunks_path)

    print("Storing vectors...")
    start = time.perf_counter()
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent
    if vector_store is None:
        vector_store = VectorStore(persist_directory=vector_dir)
    # Clean old chunks for this source before inserting the rebuilt ones
    try:
        vector_store.delete_by_source_url(source_url)
    except Exception:
        pass
    # Store the relative path from project root for proper source file identification
    processed_file_relative = chunks_path.relative_to(project_root)
    vector_store.store_chunks(chunks, embeddings, processed_file=str(processed_file_relative))
    # Record which embedding model + text field produced these vectors, so the
    # server can detect an embedding change and trigger a full re-embed.
    try:
        vector_store.set_collection_metadata(
            {"embedding_model": embedding_model.model_name, "embedded_field": "original_text"}
        )
    except Exception:
        pass
    print(f"Vector store write finished in {time.perf_counter() - start:.2f}s")
    print(f"Vector database build complete for {source_name}.")


def _mirror_source_chunks(processed_dir: Path, safe: str, chunks_path: Path) -> None:
    """Mirror a source's chunk JSON files into ``data/processed`` with source-prefixed names.

    Writes the source-level aggregate as ``<safe>_chunks.json`` (named exactly
    like the cache file in ``data/cache/sources``) and mirrors any per-document
    sibling files as ``<safe>_<document>_chunks.json``, so every processed chunk
    file visibly maps back to its source.
    """
    import shutil

    if not chunks_path.exists():
        return
    try:
        shutil.copy2(chunks_path, processed_dir / f"{safe}_chunks.json")
        # PDF sources keep one chunk JSON per PDF next to the combined
        # source-level chunks file; mirror each individually so data/processed
        # holds one *_chunks.json per document. Website sources have no sibling
        # chunk files, so this is a no-op.
        for extra in sorted(chunks_path.parent.glob("*_chunks.json")):
            if extra.resolve() == chunks_path.resolve():
                continue
            shutil.copy2(extra, processed_dir / _with_source_prefix(safe, extra.name))
    except Exception as exc:
        print(f"Warning: could not sync {chunks_path}: {exc}")


def _sync_raw_and_processed(project_root: Path) -> None:
    """Mirror the built raw markdown and per-source chunks to data/raw + data/processed.

    This keeps the human-readable raw markdown and the source-prefixed chunk
    JSON in the conventional ``data/raw`` / ``data/processed`` locations (each
    named after its source, mirroring the authoritative ``data/cache/sources``
    files), even when sources were built directly (pdf OCR or website crawl)
    rather than through the server's cache-sync path.
    """
    import shutil

    state = _load_source_state(project_root)
    raw_dir = project_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    for source_url, entry in state.items():
        raw_path = Path(entry.get("raw_path") or "")
        chunks_path = Path(entry.get("chunks_path") or "")
        safe = _safe_name_for_source(entry.get("source_name") or source_url, source_url)
        if raw_path.exists():
            try:
                # PDF sources keep one markdown file per PDF next to the
                # combined source-level raw file; mirror each individually so
                # data/raw holds one .md per PDF. Website sources fall back to
                # the single source-level markdown.
                per_pdf_files = sorted(
                    p for p in raw_path.parent.glob("*.md") if p.resolve() != raw_path.resolve()
                )
                if per_pdf_files:
                    for pdf_md in per_pdf_files:
                        shutil.copy2(pdf_md, raw_dir / _with_source_prefix(safe, pdf_md.name))
                else:
                    shutil.copy2(raw_path, raw_dir / f"{safe}.md")
            except Exception as exc:
                print(f"Warning: could not mirror {raw_path}: {exc}")
        _mirror_source_chunks(processed_dir, safe, chunks_path)


def process_sources(
    project_root: Path,
    sources: List[Dict[str, str]],
    vector_dir: Path,
    translator: TranslationModel | None = None,
    embedding_model: EmbeddingModel | None = None,
    chunk_max_tokens: int = 800,
    rebuild: bool = False,
) -> Dict[str, Dict[str, str]]:
    state = {} if rebuild else _load_source_state(project_root)
    if rebuild:
        vector_store = VectorStore(persist_directory=vector_dir)
        if vector_store.has_documents():
            vector_store.clear()

    processed_state = dict(state)
    for source in sources:
        source_url = str(source.get("url", "")).strip()
        if not source_url:
            continue
        source_name = str(source.get("source_name") or source.get("title") or "Unknown source")
        source_type = str(source.get("type") or "website").strip().lower()
        if not rebuild and source_url in processed_state:
            print(f"Skipping already processed source: {source_url}")
            continue

        raw_path, chunks_path = _source_cache_paths(project_root, source_url, source_name=source_name)
        build_vector_database(
            source_url=source_url,
            source_name=source_name,
            source_type=source_type,
            raw_path=raw_path,
            chunks_path=chunks_path,
            vector_dir=vector_dir,
            translator=translator,
            embedding_model=embedding_model,
            chunk_max_tokens=chunk_max_tokens,
            project_root=project_root,
        )
        processed_state[source_url] = {
            "source_name": source_name,
            "raw_path": str(raw_path),
            "chunks_path": str(chunks_path),
        }

    _write_source_state(project_root, processed_state)
    _sync_raw_and_processed(project_root)
    return processed_state


def print_retrieval_results(results: List[RetrievalResult]) -> None:
    if not results:
        print("No relevant chunks found.")
        return

    for index, result in enumerate(results, start=1):
        print("-" * 80)
        print(f"Rank {index}")
        print(f"Similarity score: {result.similarity:.4f}")
        print(f"Chunk ID: {result.chunk_id}")
        print(f"Heading: {result.heading}")
        print(f"Paragraph index: {result.paragraph_index}")
        print(f"Source: {result.source_name}")
        print(f"URL: {result.source_url}")
        print("Original paragraph:")
        print(result.original_paragraph)
        if result.translated_paragraph:
            print("English translation:")
            print(result.translated_paragraph)
    print("-" * 80)


def _estimate_tokens(text: str, model_name: str) -> int:
    """Estimate token count for text. Prefer tiktoken when available, otherwise fallback to a char-based heuristic."""
    try:
        import tiktoken

        try:
            enc = tiktoken.encoding_for_model(model_name)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def _configure_hf_environment(project_root: Path) -> None:
    cache_root = project_root / "data" / "cache" / "hf"
    cache_root.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_root))
    os.environ.setdefault("HF_HOME", str(cache_root))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_root))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(project_root / "data" / "cache" / "sentence_transformers"))

    if os.getenv("HF_TOKEN") and not os.getenv("HUGGINGFACE_HUB_TOKEN"):
        os.environ["HUGGINGFACE_HUB_TOKEN"] = os.getenv("HF_TOKEN")


def _ensure_model_cache(translator: TranslationModel, embedding_model: EmbeddingModel) -> None:
    print("Ensuring local cache for translation and embedding models...")
    translator.ensure_local_cache()
    embedding_model.ensure_local_cache()
    print("Local model cache ready.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and query a simple RAG retrieval pipeline.")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the vector database from the source pages listed in data/sources.json.")
    parser.add_argument(
        "--preview-tokens",
        action="store_true",
        help=(
            "Estimate input/output token usage for the LLM request and ask for user confirmation before sending. "
            "If enabled, shows estimated input tokens for the prompt and the requested max output tokens."
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens to request from the LLM (used for token preview and request).")
    parser.add_argument("--chunk-max-tokens", type=int, default=800, help="Target maximum tokens per chunk when splitting source content.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env", override=True)
    _configure_hf_environment(project_root)
    print("OPENAI_API_KEY loaded:", bool(os.getenv("OPENAI_API_KEY")))
    print("HF_TOKEN loaded:", bool(os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")))
    vector_dir = project_root / "vector_db"
    sources = _load_sources(project_root)

    # Reduce thread/parallelism overhead which can slow startup
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    # Use the local RAG server for retrieval and LLM calls to avoid loading models locally
    server_host = os.environ.get("RAG_SERVER_HOST", "127.0.0.1")
    server_port = int(os.environ.get("RAG_SERVER_PORT", "8000"))
    server_url = f"http://{server_host}:{server_port}"

    try:
        import requests

        print(f"Checking RAG server at {server_url}...")
        health = requests.get(f"{server_url}/health", timeout=5).json()
        print("Server health:", health)
        if not health.get("models_loaded"):
            print("Warning: server reports models not loaded yet. Wait for server startup to complete before asking questions.")
    except Exception as exc:
        print(f"Unable to contact local RAG server at {server_url}: {exc}")
        print("Please start the server: python project\\src\\server.py or python -m uvicorn src.server:app --port 8000")
        return

    if args.rebuild:
        print("Rebuild requested — this must be performed on the server. Start the server and call its rebuild endpoint if available.")
        print("(Server-side rebuild endpoint not implemented in this version.)")

    try:
        question = input("Ask a question about Dutch influencer marketing regulations: ").strip()
    except EOFError:
        print("No question provided. Exiting.")
        return

    if not question:
        print("No question entered. Exiting.")
        return

    payload = {"question": question, "top_k": 3, "max_tokens": args.max_tokens, "preview_tokens": args.preview_tokens}

    import requests

    print("Sending request to local RAG server...")
    start = time.perf_counter()
    resp = requests.post(f"{server_url}/answer", json=payload, timeout=120)
    elapsed = time.perf_counter() - start
    print(f"Server request finished in {elapsed:.2f}s")

    if resp.status_code != 200:
        print("Server returned error:", resp.status_code, resp.text)
        return

    body = resp.json()
    answer_text = body.get("answer", "")
    timings = body.get("timings", {})
    metadata = body.get("metadata") or {}
    results_meta = metadata.get("results") if isinstance(metadata, dict) else None

    print("\nAnswer:\n")
    print(answer_text)
    print('\nTimings (server-side):')
    for k, v in timings.items():
        print(f"  {k}: {v:.3f}s")

    # Print only cited evidence quotes
    cited_indices = body.get("cited_indices", []) or []
    if results_meta and cited_indices:
        print("\nEvidence quotes cited in answer:")
        seen = set()
        # Renumber cited quotes with normal sequential counting (1, 2, 3, ...),
        # regardless of the raw citation numbers the answer happens to use
        # (e.g. an answer that only cites [1] and [3] lists just two quotes, [1] and [2]).
        citation_number = 0
        for index in cited_indices:
            if index in seen or index < 1 or index > len(results_meta):
                continue
            seen.add(index)
            citation_number += 1
            res = results_meta[index - 1]
            processed_file = res.get('processed_file') or res.get('processed_file', '')
            chunk_id = res.get('chunk_id', '')
            # Strip source hash prefix to match chunks.json format (e.g., "fbfbc3e65862a-chunk-6" -> "chunk-6")
            if chunk_id and '-' in chunk_id:
                parts = chunk_id.split('-')
                if len(parts) >= 3 and parts[-2] == 'chunk':
                    chunk_id = f"chunk-{parts[-1]}"
            print(f"[{citation_number}] File: {processed_file}  Chunk: {chunk_id}")
            snippet = (res.get('quotation') or res.get('translated_paragraph') or res.get('original_paragraph') or '').strip()
            if snippet:
                q_lang = res.get('quotation_language') or ''
                lang_tag = f" [{q_lang}]" if q_lang else ""
                print(f"   Quote{lang_tag}: {snippet}")
    else:
        print("\nNo cited evidence quotes were detected or available.")



if __name__ == "__main__":
    main()

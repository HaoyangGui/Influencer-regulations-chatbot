from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class Chunk:
    chunk_id: str
    heading: str
    source_name: str
    source_url: str
    paragraph_index: int
    original_paragraph: str
    embedding_text: str

    translated_paragraph: str = ""
    embedding_text_en: str = ""

    # Name of the individual source document this chunk came from. For
    # multi-document sources (e.g. a folder of PDFs) each chunk records which
    # document it belongs to, so the RAG system can query one document alone.
    document_name: str = ""

    # language code of the original (source) text
    source_language: str = "nl"
    # mapping of language code -> text for that language
    translations: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.*)$")
_HORIZONTAL_RULE_RE = re.compile(r"^\s*(?:[-*_]\s*){3,}$")


def _split_markdown_blocks(markdown_text: str) -> List[tuple[str, str]]:
    """Walk markdown line by line and emit ``(heading, paragraph)`` blocks.

    Mirrors ``03_chunk``'s parsing approach: every ATX heading switches the
    active section, every blank line closes the current paragraph and every
    list item is treated as its own paragraph, so PDF-derived OCR markdown is
    separated per paragraph exactly like crawled web pages. The project's
    existing heading rule is kept: the chunk heading is simply the most recent
    heading title (no hierarchy suffix).
    """
    blocks: List[tuple[str, str]] = []
    heading = ""
    buffer: List[str] = []
    in_code_fence = False

    def flush_buffer() -> None:
        nonlocal buffer
        if not buffer:
            return
        paragraph = " ".join(line.strip() for line in buffer if line.strip()).strip()
        buffer = []
        if paragraph:
            blocks.append((heading, paragraph))

    for raw_line in markdown_text.replace("\r\n", "\n").splitlines():
        line = raw_line.strip()

        if line.startswith("```"):
            flush_buffer()
            in_code_fence = not in_code_fence
            continue
        if in_code_fence:
            buffer.append(raw_line)
            continue
        if not line:
            flush_buffer()
            continue

        heading_match = _HEADING_RE.match(line)
        if heading_match:
            flush_buffer()
            heading = heading_match.group(2).strip()
            continue

        if _HORIZONTAL_RULE_RE.match(line):
            flush_buffer()
            continue

        list_match = _LIST_ITEM_RE.match(line)
        if list_match:
            flush_buffer()
            item_text = list_match.group(1).strip()
            if item_text:
                blocks.append((heading, item_text))
            continue

        buffer.append(line)

    flush_buffer()
    return blocks


def _split_into_sentences(text: str) -> List[str]:
    # Prefer nltk punkt tokenizer if available for robust sentence splitting
    try:
        import nltk

        try:
            nltk.data.find("tokenizers/punkt")
        except Exception:
            nltk.download("punkt")
        from nltk.tokenize import sent_tokenize

        return sent_tokenize(text)
    except Exception:
        # Fallback simple regex-based splitter
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]


def _estimate_tokens_for_text(text: str) -> int:
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def chunk_document(
    markdown_text: str,
    source_name: str,
    source_url: str,
    *,
    max_tokens: int = 800,
    min_sentences: int = 1,
    document_name: str = "",
) -> List[Chunk]:

    """
    Paragraph-aware chunking (mirrors ``03_chunk``'s parsing):

    - Walk the markdown line by line; headings switch the active section,
      blank lines close paragraphs and list items are their own paragraph,
      so PDF-derived OCR markdown is separated per paragraph like web pages.
    - Split each paragraph into sentences (prefer nltk).
    - Group sentences into chunks aiming for ~max_tokens (tiktoken when
      available); chunks never span paragraph boundaries.
    """
    blocks = _split_markdown_blocks(markdown_text)

    chunks: List[Chunk] = []
    chunk_idx = 0
    para_idx = 0

    for heading, paragraph in blocks:
        para_idx += 1

        # Split paragraph into sentences
        sentences = _split_into_sentences(paragraph)
        if not sentences:
            continue

        current_chunk_sentences: List[str] = []
        current_chunk_tokens = 0

        for sent in sentences:
            sent_tokens = _estimate_tokens_for_text(sent)
            if current_chunk_sentences and (current_chunk_tokens + sent_tokens > max_tokens):
                # flush current chunk
                chunk_idx += 1
                chunk_text = " ".join(current_chunk_sentences)
                embedding_text = f"{heading}\n\n{chunk_text}".strip()
                chunk = Chunk(
                    chunk_id=f"chunk-{chunk_idx}",
                    heading=heading,
                    source_name=source_name,
                    source_url=source_url,
                    paragraph_index=para_idx,
                    original_paragraph=chunk_text,
                    embedding_text=embedding_text,
                    document_name=document_name,
                )
                chunks.append(chunk)
                current_chunk_sentences = [sent]
                current_chunk_tokens = sent_tokens
            else:
                current_chunk_sentences.append(sent)
                current_chunk_tokens += sent_tokens

        # flush any remaining sentences in the paragraph
        if current_chunk_sentences:
            chunk_idx += 1
            chunk_text = " ".join(current_chunk_sentences)
            embedding_text = f"{heading}\n\n{chunk_text}".strip()
            chunk = Chunk(
                chunk_id=f"chunk-{chunk_idx}",
                heading=heading,
                source_name=source_name,
                source_url=source_url,
                paragraph_index=para_idx,
                original_paragraph=chunk_text,
                embedding_text=embedding_text,
                document_name=document_name,
            )
            chunks.append(chunk)

    return chunks


def save_chunks_to_json(chunks: List[Chunk], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump([chunk.to_dict() for chunk in chunks], handle, indent=2, ensure_ascii=False)
    return output_path

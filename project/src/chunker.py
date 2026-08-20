from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


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

    # language code of the original (source) text
    source_language: str = "nl"
    # mapping of language code -> text for that language
    translations: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "Chunk":
        return Chunk(**data)


def _normalize_text(text: str) -> str:
    # Collapse multiple whitespace, preserve newlines where paragraphs were
    text = re.sub(r"[ \t\xa0]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text.strip()


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


def chunk_document(markdown_text: str, source_name: str, source_url: str, *, max_tokens: int = 800, min_sentences: int = 1) -> List[Chunk]:

    """
    Improved chunking:
    - Split the document into sections by headings
    - Split sections into sentences (prefer nltk)
    - Group sentences into chunks aiming for ~max_tokens (using tiktoken when available)
    """
    text = _normalize_text(markdown_text)
    lines = text.splitlines()

    # Build sections: tuples of (heading, section_text)
    sections: List[tuple[str, str]] = []
    current_heading = ""
    buffer_lines: List[str] = []

    def flush_section():
        nonlocal buffer_lines, current_heading
        section_text = "\n".join(buffer_lines).strip()
        if section_text:
            sections.append((current_heading, section_text))
        buffer_lines = []

    for raw in lines:
        line = raw.strip()
        if not line:
            # keep paragraphs
            buffer_lines.append("")
            continue
        if line.startswith("#"):
            # heading
            flush_section()
            # heading text after hashes
            heading = line.lstrip("#").strip()
            current_heading = heading
            continue
        buffer_lines.append(line)

    flush_section()

    chunks: List[Chunk] = []
    chunk_idx = 0
    para_idx = 0

    for heading, section_text in sections:
        # Split section into paragraphs
        paragraphs = [p.strip() for p in section_text.split("\n\n") if p.strip()]
        for para in paragraphs:
            para_idx += 1
            # Split paragraph into sentences
            sentences = _split_into_sentences(para)
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
                )
                chunks.append(chunk)

    return chunks


def save_chunks_to_json(chunks: List[Chunk], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump([chunk.to_dict() for chunk in chunks], handle, indent=2, ensure_ascii=False)
    return output_path


def load_chunks_from_json(input_path: Path) -> List[Chunk]:
    with input_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return [Chunk.from_dict(item) for item in data]

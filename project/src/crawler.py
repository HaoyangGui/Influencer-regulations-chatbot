from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup, Tag


def download_page(url: str, timeout: int = 20) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; RetrieverBot/1.0; +https://example.com)"
    }
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.text


def _find_main_content(soup: BeautifulSoup) -> Tag:
    selectors = ["main#main-content", "article", "div.content-container", "div.rich-text", "body"]
    for selector in selectors:
        candidate = soup.select_one(selector)
        if candidate is not None:
            return candidate
    raise ValueError("Unable to locate the main article content in the page.")


def _clean_text(text: str) -> str:
    # Remove repeated whitespace, preserve paragraphs.
    normalized = re.sub(r"[ \t\xa0]+", " ", text)
    normalized = re.sub(r"\s*\n\s*", "\n", normalized)
    normalized = normalized.strip()
    return normalized


def _convert_node_to_markdown(node: Tag) -> list[str]:
    paragraphs: list[str] = []
    for child in node.children:
        if isinstance(child, Tag):
            if child.name in {"h1", "h2", "h3"}:
                level = int(child.name[1])
                heading = f"{'#' * level} {_clean_text(child.get_text(separator=' ', strip=True))}"
                paragraphs.append(heading)
            elif child.name == "p":
                text = _clean_text(child.get_text(separator=' ', strip=True))
                if text:
                    paragraphs.append(text)
            elif child.name in {"ul", "ol"}:
                for item in child.find_all("li", recursive=False):
                    item_text = _clean_text(item.get_text(separator=' ', strip=True))
                    if item_text:
                        paragraphs.append(f"- {item_text}")
            elif child.name in {"div", "section", "article", "main"}:
                paragraphs.extend(_convert_node_to_markdown(child))
            else:
                paragraphs.extend(_convert_node_to_markdown(child))
    return paragraphs


def clean_html_to_markdown(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag_name in ["script", "style", "noscript", "header", "footer", "nav", "aside", "form"]:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    article_node = _find_main_content(soup)
    lines = _convert_node_to_markdown(article_node)

    # Merge adjacent list items and paragraphs with blank lines between logical blocks.
    markdown = "\n\n".join(lines).strip()
    return markdown


def download_and_save_markdown(url: str, output_path: Path, source_name: Optional[str] = None) -> Path:
    html = download_page(url)
    markdown = clean_html_to_markdown(html)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    return output_path


# --------------------------------------------------------------------------- #
# PDF mode (local)
#
# When a source is registered with ``"type": "pdf"``, the crawler does NOT
# download anything from the web: the target PDFs are already placed under the
# project's ``data/pdf`` directory. The crawler lists those local PDFs, OCRs
# each one to markdown (mirroring ``01_ocr.py`` via the Mistral OCR API), then
# cleans the PDF-conversion artefacts (mirroring ``02_clean.py``) so the
# existing RAG chunker/embedder can index them like any other source.
# --------------------------------------------------------------------------- #


def iter_local_pdfs(pdf_dir: Path) -> list[Path]:
    """Return the sorted list of local ``.pdf`` files under ``pdf_dir``."""
    if not pdf_dir.exists() or not pdf_dir.is_dir():
        return []
    return sorted(p for p in pdf_dir.rglob("*.pdf") if p.is_file())


def _is_noise_line(text: str) -> bool:
    """Apply ``03_chunk``-style heuristics to flag likely-noise lines.

    A line is treated as noise when it has few words or is dominated by digits
    or contains little alphabetic content (typical of OCR headers, footers and
    repeated decorative text).
    """
    if not text:
        return True
    words = re.findall(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?", text)
    if len(words) < 2:
        return True
    non_space = re.findall(r"\S", text)
    if not non_space:
        return True
    alpha = sum(1 for ch in non_space if ch.isalpha())
    digits = sum(1 for ch in non_space if ch.isdigit())
    alpha_ratio = alpha / len(non_space)
    digit_ratio = digits / len(non_space)
    if alpha_ratio < 0.35:
        return True
    if alpha_ratio < 0.55 and digit_ratio > 0.35:
        return True
    return False


def clean_ocr_markdown(markdown: str) -> str:
    """Remove PDF-conversion artefacts from OCR markdown (mirrors ``02_clean``).

    Drops Markdown image placeholders, base64 payloads, isolated page numbers,
    footnote definitions and the ``<!-- PAGE ... -->`` markers emitted by the
    OCR step, and applies the ``03_chunk`` noise heuristics for short / low
    alpha / high digit lines (headers, footers, repeated page furniture).
    """
    image_patterns = [
        re.compile(r"!\[.*?\]\(.*?\)"),
        re.compile(r"<img\b", re.IGNORECASE),
        re.compile(r"data:image/", re.IGNORECASE),
        re.compile(r"\bbase64\b", re.IGNORECASE),
    ]
    page_number_pattern = re.compile(r"^\s*\d{1,4}\s*$")
    footnote_pattern = re.compile(r"^\s*\[\^\w+\]\s*:")
    html_comment_pattern = re.compile(r"<!--.*?-->")

    markdown = html_comment_pattern.sub("", markdown)

    kept: list[str] = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if page_number_pattern.match(stripped):
            continue
        if footnote_pattern.match(stripped):
            continue
        if any(p.search(line) for p in image_patterns):
            continue
        if _is_noise_line(stripped):
            continue
        kept.append(stripped)
    return "\n\n".join(kept).strip()


def _build_mistral_client(api_key: str | None = None):
    """Build a Mistral client for OCR extraction (mirrors ``01_ocr.build_client``).

    Self-contained in ``crawler.py`` so the pipeline never imports from
    ``01_ocr.py``. The ``mistralai`` package is imported lazily.
    """
    try:
        from mistralai.client import Mistral
    except Exception as exc:
        raise RuntimeError(
            "Cannot initialize Mistral OCR; install the 'mistralai' package "
            "(python -m pip install mistralai)."
        ) from exc

    api_key = api_key or os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("Missing MISTRAL_API_KEY (env var) or api_key argument.")
    return Mistral(api_key=api_key)


def _upload_pdf_for_ocr(client, pdf_path: Path) -> str:
    """Upload a PDF to Mistral and return a signed URL (mirrors ``01_ocr.upload_pdf``)."""
    pdf_path = pdf_path.expanduser().resolve()
    uploaded = client.files.upload(
        file={"file_name": pdf_path.name, "content": pdf_path.open("rb")},
        purpose="ocr",
    )
    signed_url = client.files.get_signed_url(file_id=uploaded.id)
    return signed_url.url


def ocr_pdf_to_markdown(
    pdf_path: Path,
    *,
    client=None,
    api_key: str | None = None,
    model: str = "mistral-ocr-latest",
    include_image_base64: bool = True,
) -> str:
    """OCR a single local PDF to Markdown via the Mistral OCR API.

    Self-contained equivalent of ``01_ocr.ocr_pdf``/``process_pdf`` that returns
    the per-page markdown as a string instead of writing files. ``client`` may
    be passed to reuse a client across calls (avoids re-uploading when iterating
    many PDFs) and is built from ``MISTRAL_API_KEY`` when omitted.
    """
    client = client or _build_mistral_client(api_key=api_key)
    response = client.ocr.process(
        model=model,
        document={"type": "document_url", "document_url": _upload_pdf_for_ocr(client, pdf_path)},
        include_image_base64=include_image_base64,
    )
    pages = list(getattr(response, "pages", []) or [])
    page_md: list[str] = []
    for page in pages:
        content = (getattr(page, "markdown", "") or "").strip()
        if content:
            page_md.append(content)
    return "\n\n".join(page_md).strip()


def ocr_pdfs_to_sections(
    pdf_paths: list[Path],
    *,
    client=None,
    api_key: str | None = None,
    model: str = "mistral-ocr-latest",
    include_image_base64: bool = True,
) -> list[tuple[str, str]]:
    """OCR local PDFs into cleaned markdown sections, one per PDF.

    Returns an ordered list of ``(pdf_stem, cleaned_markdown)`` tuples so each
    PDF can be saved to its own ``.md`` file. PDFs that fail OCR are skipped
    with a warning and do not abort the batch.
    """
    client = client or _build_mistral_client(api_key=api_key)
    sections: list[tuple[str, str]] = []
    for pdf_path in pdf_paths:
        try:
            text = ocr_pdf_to_markdown(
                pdf_path,
                client=client,
                model=model,
                include_image_base64=include_image_base64,
            )
        except Exception as exc:
            print(f"Warning: could not OCR {pdf_path.name}: {exc}")
            continue
        cleaned = clean_ocr_markdown(text)
        if cleaned:
            sections.append((pdf_path.stem, cleaned))
    return sections


def ocr_pdfs_to_clean_markdown(
    pdf_paths: list[Path],
    *,
    client=None,
    api_key: str | None = None,
    model: str = "mistral-ocr-latest",
    include_image_base64: bool = True,
) -> str:
    """OCR a list of local PDFs into one combined cleaned Markdown document.

    Convenience wrapper around :func:`ocr_pdfs_to_sections` that joins the
    per-PDF sections (each introduced by a ``# <filename>`` heading) into the
    single source-level markdown consumed by the chunk/embed pipeline.
    """
    sections = ocr_pdfs_to_sections(
        pdf_paths,
        client=client,
        api_key=api_key,
        model=model,
        include_image_base64=include_image_base64,
    )
    return "\n\n".join(f"# {stem}\n\n{md}" for stem, md in sections).strip()

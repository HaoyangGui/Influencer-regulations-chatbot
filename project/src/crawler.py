from __future__ import annotations

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

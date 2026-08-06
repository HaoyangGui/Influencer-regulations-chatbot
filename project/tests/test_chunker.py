from src.chunker import chunk_document


def test_chunk_document_creates_heading_aware_paragraphs() -> None:
    markdown = """
    # Top heading

    First paragraph.

    ## Section one

    - Bullet one
    - Bullet two

    Second paragraph under section.
    """
    chunks = chunk_document(markdown, source_name="source", source_url="https://example.com")

    assert len(chunks) == 4
    assert chunks[0].heading == "Top heading"
    assert chunks[0].original_paragraph == "First paragraph."
    assert chunks[0].embedding_text.startswith("Top heading")
    assert chunks[1].heading == "Section one"
    assert chunks[1].original_paragraph == "Bullet one"
    assert chunks[2].original_paragraph == "Bullet two"
    assert chunks[3].original_paragraph == "Second paragraph under section."

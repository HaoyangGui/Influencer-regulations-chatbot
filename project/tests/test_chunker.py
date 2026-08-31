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


def test_chunk_document_separates_pdf_markdown_per_paragraph() -> None:
    # Simulates cleaned OCR markdown: a document heading, a section heading,
    # and paragraphs separated by blank lines (one per OCR paragraph).
    markdown = (
        "# Legal brief 1\n"
        "\n"
        "Intro paragraph about consumer law.\n"
        "\n"
        "## Enforcement rules\n"
        "\n"
        "Influencers must disclose commercial content clearly.\n"
        "\n"
        "Sanctions can apply when disclosures are missing.\n"
    )
    chunks = chunk_document(markdown, source_name="hub", source_url="https://example.com")

    # One chunk per paragraph; no chunk spans a paragraph boundary.
    assert len(chunks) == 3
    assert [c.original_paragraph for c in chunks] == [
        "Intro paragraph about consumer law.",
        "Influencers must disclose commercial content clearly.",
        "Sanctions can apply when disclosures are missing.",
    ]
    # Heading rule unchanged: the most recent heading title is used.
    assert chunks[0].heading == "Legal brief 1"
    assert chunks[1].heading == "Enforcement rules"
    assert chunks[2].heading == "Enforcement rules"
    # Paragraph indices increase across the whole document.
    assert [c.paragraph_index for c in chunks] == [1, 2, 3]


def test_chunk_document_wraps_long_paragraph_by_tokens() -> None:
    long_sentence = "Dit is een redelijk lange zin voor de test. " * 40
    chunks = chunk_document(
        f"# Head\n\n{long_sentence}",
        source_name="s",
        source_url="https://example.com",
        max_tokens=100,
    )
    assert len(chunks) > 1
    # Every chunk stays under the token budget (plus the heading prefix).
    from src.chunker import _estimate_tokens_for_text

    for chunk in chunks:
        assert _estimate_tokens_for_text(chunk.embedding_text) <= 100 + _estimate_tokens_for_text("Head") + 2


def test_chunk_document_sets_document_name_per_pdf() -> None:
    """Every chunk from one PDF's markdown carries that PDF's document_name."""
    markdown = (
        "Legal brief intro.\n"
        "\n"
        "## Rules\n"
        "\n"
        "Influencers must disclose ads.\n"
    )
    chunks = chunk_document(
        markdown,
        source_name="hub",
        source_url="https://example.com",
        document_name="Legal brief 1",
    )
    assert len(chunks) == 2
    assert all(c.document_name == "Legal brief 1" for c in chunks)


def test_chunk_document_document_name_defaults_to_empty() -> None:
    chunks = chunk_document("# Head\n\nBody text.", source_name="s", source_url="https://e.com")
    assert chunks[0].document_name == ""

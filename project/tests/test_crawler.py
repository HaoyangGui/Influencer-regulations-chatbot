from src.crawler import (
    clean_html_to_markdown,
    clean_ocr_markdown,
    iter_local_pdfs,
    ocr_pdf_to_markdown,
    ocr_pdfs_to_clean_markdown,
)


def test_clean_html_to_markdown_preserves_headings_and_paragraphs() -> None:
    html = """
    <html>
      <body>
        <header>Navigation</header>
        <main id='main-content'>
          <h1>Title</h1>
          <p>First paragraph.</p>
          <ul><li>First item</li><li>Second item</li></ul>
        </main>
        <footer>Footer content</footer>
      </body>
    </html>
    """

    markdown = clean_html_to_markdown(html)

    assert "# Title" in markdown
    assert "First paragraph." in markdown
    assert "- First item" in markdown
    assert "- Second item" in markdown
    assert "Navigation" not in markdown
    assert "Footer content" not in markdown


def test_iter_local_pdfs(tmp_path) -> None:
    import src.crawler as crawler_mod

    (tmp_path / "sub").mkdir()
    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "sub" / "b.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "notes.txt").write_text("not a pdf")

    files = crawler_mod.iter_local_pdfs(tmp_path)
    assert [p.name for p in files] == ["a.pdf", "b.pdf"]


def test_clean_ocr_markdown_removes_artefacts() -> None:
    messy = (
        "<!-- PAGE 1 of 2 -->\n\n"
        "![image](https://example.com/x.png)\n\n"
        "12\n\n"
        "[^1]: A footnote definition\n\n"
        "The real rule is that influencers must label ads.\n\n"
        "###\n"
    )
    cleaned = clean_ocr_markdown(messy)
    assert "PAGE 1 of 2" not in cleaned
    assert "![image]" not in cleaned
    assert cleaned.split() == [
        "The",
        "real",
        "rule",
        "is",
        "that",
        "influencers",
        "must",
        "label",
        "ads.",
    ]


def test_ocr_pdf_to_markdown_uses_mistral_ocr(tmp_path) -> None:
    import src.crawler as crawler_mod

    class FakePage:
        markdown = "# Fake heading\nOCR content line"

    class FakeResponse:
        pages = [FakePage(), FakePage()]

    uploaded = {}

    class FakeFiles:
        def upload(self, file, purpose):
            uploaded["name"] = file["file_name"]
            file["content"].close()
            return type("U", (), {"id": "file-123"})()

        def get_signed_url(self, file_id):
            return type("S", (), {"url": "https://files.mistral/signed"})()

    class FakeOcr:
        def process(self, **kwargs):
            assert kwargs["model"] == "mistral-ocr-latest"
            assert kwargs["document"] == {
                "type": "document_url",
                "document_url": "https://files.mistral/signed",
            }
            return FakeResponse()

    class FakeClient:
        files = FakeFiles()
        ocr = FakeOcr()

    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    markdown = crawler_mod.ocr_pdf_to_markdown(pdf_path, client=FakeClient())

    assert uploaded["name"] == "doc.pdf"
    assert "OCR content line" in markdown
    # The two OCR pages are concatenated with a blank line between them.
    assert markdown.count("# Fake heading") == 2


def test_ocr_pdfs_to_sections_returns_one_section_per_pdf(tmp_path) -> None:
    import src.crawler as crawler_mod

    class FakePage:
        markdown = "Real content minus artefacts.\n\n![image](/img.png)\n\n7\n\n"

    class FakeResponse:
        pages = [FakePage()]

    class FakeFiles:
        def upload(self, file, purpose):
            file["content"].close()
            return type("U", (), {"id": "file-123"})()

        def get_signed_url(self, file_id):
            return type("S", (), {"url": "https://files.mistral/signed"})()

    class FakeOcr:
        def process(self, **kwargs):
            return FakeResponse()

    class FakeClient:
        files = FakeFiles()
        ocr = FakeOcr()

    pdf_a = tmp_path / "legal_brief.pdf"
    pdf_b = tmp_path / "guide.pdf"
    pdf_a.write_bytes(b"%PDF-1.4")
    pdf_b.write_bytes(b"%PDF-1.4")

    sections = crawler_mod.ocr_pdfs_to_sections([pdf_a, pdf_b], client=FakeClient())

    # One (stem, cleaned_markdown) tuple per PDF, in order.
    assert [stem for stem, _ in sections] == ["legal_brief", "guide"]
    for _, md in sections:
        assert "![image]" not in md
        assert "Real content minus artefacts." in md


def test_ocr_pdfs_to_clean_markdown_cleans_each_pdf(tmp_path) -> None:
    import src.crawler as crawler_mod

    class FakePage:
        markdown = "Real content minus artefacts.\n\n![image](/img.png)\n\n7\n\n"

    class FakeResponse:
        pages = [FakePage()]

    class FakeFiles:
        def upload(self, file, purpose):
            file["content"].close()
            return type("U", (), {"id": "file-123"})()

        def get_signed_url(self, file_id):
            return type("S", (), {"url": "https://files.mistral/signed"})()

    class FakeOcr:
        def process(self, **kwargs):
            return FakeResponse()

    class FakeClient:
        files = FakeFiles()
        ocr = FakeOcr()

    pdf_a = tmp_path / "legal_brief.pdf"
    pdf_b = tmp_path / "guide.pdf"
    pdf_a.write_bytes(b"%PDF-1.4")
    pdf_b.write_bytes(b"%PDF-1.4")

    md = crawler_mod.ocr_pdfs_to_clean_markdown([pdf_a, pdf_b], client=FakeClient())

    assert md.startswith("# legal_brief")
    assert "# guide" in md
    # Image placeholder and isolated page number are removed by the cleaner.
    assert "![image]" not in md
    assert "Real content minus artefacts." in md


def test_build_mistral_client_requires_api_key(monkeypatch) -> None:
    import src.crawler as crawler_mod

    import os

    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    try:
        crawler_mod._build_mistral_client(api_key=None)
        raise AssertionError("expected RuntimeError when MISTRAL_API_KEY is unset")
    except RuntimeError as exc:
        assert "MISTRAL_API_KEY" in str(exc)

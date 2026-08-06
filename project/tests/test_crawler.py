from src.crawler import clean_html_to_markdown


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

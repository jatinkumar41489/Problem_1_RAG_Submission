import pytest
from pypdf import PdfWriter

from app.config import BASE_DIR
from app.exceptions import DocumentLoadError, UnsupportedFileType
from app.ingestion.loaders import detect_document_type, load_document

S = BASE_DIR / "data" / "sample_documents"


@pytest.mark.parametrize("name,t", [("a.PDF", "pdf"), ("a.html", "html"), ("a.htm", "html"), ("a.md", "markdown"), ("a.markdown", "markdown")])
def test_type_detection(tmp_path, name, t):
    assert detect_document_type(tmp_path / name) == t


def test_unsupported_type(tmp_path):
    with pytest.raises(UnsupportedFileType):
        detect_document_type(tmp_path / "a.docx")


def test_markdown_headings_and_no_markup():
    d = load_document(S / "hr/leave_policy.md", "hr/leave_policy.md")
    seg = d.segments[0]
    assert d.title == "Meridian Cloud Leave Policy" and d.category == "hr"
    assert "##" not in seg.text and "**" not in seg.text
    assert any("Annual leave" in h for _, h in seg.headings)


def test_html_strips_script_style_and_nav():
    d = load_document(S / "hr/remote_work_policy.html", "hr/remote_work_policy.html")
    t = d.segments[0].text
    assert "SHOULD-NOT-APPEAR" not in t and "font-family" not in t and "<" not in t
    assert "Home" not in load_document(S / "product/atlas_database_overview.html", "x").segments[0].text
    assert "Benefit | Amount | Frequency" in t   # table rows kept as text


def test_pdf_pages_preserved():
    d = load_document(S / "finance/expense_policy.pdf", "finance/expense_policy.pdf")
    assert [s.page for s in d.segments] == [1, 2, 3] and d.document_type == "pdf"
    assert "$220 per night" in " ".join(d.segments[1].text.split())


def test_empty_markdown_and_html(tmp_path):
    (tmp_path / "e.md").write_text("   \n")
    (tmp_path / "e.html").write_text("<html><body><script>x=1</script></body></html>")
    for n in ("e.md", "e.html"):
        with pytest.raises(DocumentLoadError):
            load_document(tmp_path / n, n)


def test_corrupted_pdf(tmp_path):
    (tmp_path / "bad.pdf").write_bytes(b"%PDF-1.4 this is not really a pdf")
    with pytest.raises(DocumentLoadError):
        load_document(tmp_path / "bad.pdf", "bad.pdf")


def test_scanned_pdf_without_text_reports_ocr_unsupported(tmp_path):
    w = PdfWriter(); w.add_blank_page(300, 300)
    with open(tmp_path / "scan.pdf", "wb") as f:
        w.write(f)
    with pytest.raises(DocumentLoadError, match="OCR is not implemented"):
        load_document(tmp_path / "scan.pdf", "scan.pdf")


def test_front_matter_removed(tmp_path):
    (tmp_path / "f.md").write_text("---\ntitle: x\nsecret: abc123\n---\n# Hello\nBody text here.\n")
    assert "abc123" not in load_document(tmp_path / "f.md", "f.md").segments[0].text

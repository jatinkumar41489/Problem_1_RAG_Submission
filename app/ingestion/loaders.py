"""Document loaders for PDF, HTML and Markdown.

Every loader returns a :class:`LoadedDocument` made of *segments*. A segment is the unit that
chunking never crosses: one PDF page, or one whole HTML/Markdown body. Heading positions inside a
segment are recorded so each chunk can be labelled with its section.

OCR is NOT implemented: scanned PDFs without a text layer are reported as errors, not guessed at.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup, Tag

from app.exceptions import DocumentLoadError, UnsupportedFileType
from app.ingestion.hashing import content_hash

SUPPORTED_EXTENSIONS = {".pdf", ".html", ".htm", ".md", ".markdown"}
_DOC_TYPE = {".pdf": "pdf", ".html": "html", ".htm": "html", ".md": "markdown", ".markdown": "markdown"}
_BLOCK_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "tr", "blockquote", "dt", "dd"]
_HEADING_TAGS = {f"h{i}" for i in range(1, 7)}
_DROP_TAGS = ["script", "style", "noscript", "template", "iframe", "svg", "nav", "footer"]


@dataclass
class Segment:
    text: str
    page: int | None = None
    # (char_offset_in_text, "H1 > H2 > H3") sorted by offset
    headings: list[tuple[int, str]] = field(default_factory=list)


@dataclass
class LoadedDocument:
    source_id: str
    path: Path
    document_type: str
    title: str
    segments: list[Segment]
    content_hash: str
    category: str


def detect_document_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileType(f"Unsupported file type {ext!r} for {path.name}. "
                                  f"Supported: {sorted(SUPPORTED_EXTENSIONS)}")
    return _DOC_TYPE[ext]


def derive_category(source_id: str) -> str:
    parts = source_id.split("/")
    return parts[0] if len(parts) > 1 else "general"


# --------------------------------------------------------------------------- PDF
def _load_pdf(path: Path) -> tuple[str, list[Segment]]:
    from pypdf import PdfReader
    from pypdf.errors import PyPdfError

    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            raise DocumentLoadError(f"{path.name}: PDF is encrypted; decrypt it before ingesting.")
        title = None
        try:
            title = (reader.metadata.title or "").strip() if reader.metadata else None
        except Exception:  # metadata is optional
            title = None
        segments: list[Segment] = []
        for i, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                segments.append(Segment(text=text, page=i))
    except DocumentLoadError:
        raise
    except (PyPdfError, OSError, ValueError, KeyError, TypeError) as exc:
        raise DocumentLoadError(f"{path.name}: unreadable or corrupted PDF ({type(exc).__name__}: {exc})") from exc
    if not segments:
        raise DocumentLoadError(
            f"{path.name}: no extractable text. It may be a scanned PDF; OCR is not implemented.")
    return title or path.stem, segments


# --------------------------------------------------------------------------- HTML / Markdown
def _extract_from_soup(soup: BeautifulSoup, fallback_title: str) -> tuple[str, Segment]:
    title_tag = soup.find("title")
    title = title_tag.get_text(" ", strip=True) if title_tag else ""
    if title_tag:
        title_tag.decompose()
    for t in soup.find_all(_DROP_TAGS):
        t.decompose()
    body = soup.body or soup
    if not title:
        h1 = body.find("h1")
        title = h1.get_text(" ", strip=True) if h1 else fallback_title

    blocks: list[Tag] = []
    for el in body.find_all(_BLOCK_TAGS):
        # keep only leaf blocks to avoid duplicated text from nested blocks (li > p, etc.)
        if not el.find(_BLOCK_TAGS):
            blocks.append(el)

    parts: list[str] = []
    headings: list[tuple[int, str]] = []
    stack: list[tuple[int, str]] = []
    offset = 0
    for el in blocks:
        if el.name == "tr":
            text = " | ".join(c.get_text(" ", strip=True) for c in el.find_all(["th", "td"]))
        else:
            text = el.get_text(" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        if el.name in _HEADING_TAGS:
            level = int(el.name[1])
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, text))
            headings.append((offset, " > ".join(t for _, t in stack)))
        parts.append(text)
        offset += len(text) + 1  # joined with "\n"
    return title, Segment(text="\n".join(parts), headings=headings)


def _load_html(path: Path) -> tuple[str, list[Segment]]:
    raw = path.read_bytes().decode("utf-8", errors="replace")
    title, seg = _extract_from_soup(BeautifulSoup(raw, "html.parser"), path.stem)
    if not seg.text.strip():
        raise DocumentLoadError(f"{path.name}: HTML contains no readable text.")
    return title, [seg]


def _load_markdown(path: Path) -> tuple[str, list[Segment]]:
    import markdown as md

    raw = path.read_bytes().decode("utf-8", errors="replace")
    raw = re.sub(r"\A---\s*\n.*?\n---\s*\n", "", raw, flags=re.S)  # YAML front matter
    html = md.markdown(raw, extensions=["tables", "fenced_code", "sane_lists"])
    title, seg = _extract_from_soup(BeautifulSoup(f"<body>{html}</body>", "html.parser"), path.stem)
    if not seg.text.strip():
        raise DocumentLoadError(f"{path.name}: Markdown contains no readable text.")
    return title, [seg]


# --------------------------------------------------------------------------- public API
def load_document(path: Path, source_id: str) -> LoadedDocument:
    """Load one file. Raises UnsupportedFileType / DocumentLoadError with a clear message."""
    doc_type = detect_document_type(path)
    if not path.is_file():
        raise DocumentLoadError(f"{path} is not a file.")
    loader = {"pdf": _load_pdf, "html": _load_html, "markdown": _load_markdown}[doc_type]
    title, segments = loader(path)
    return LoadedDocument(
        source_id=source_id, path=path, document_type=doc_type, title=title, segments=segments,
        content_hash=content_hash([(s.page, s.text) for s in segments]),
        category=derive_category(source_id),
    )

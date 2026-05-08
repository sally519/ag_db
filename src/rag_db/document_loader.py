from __future__ import annotations

from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import Request, url2pathname, urlopen
import mimetypes

from rag_db.models import SourceDocument


class DocumentLoader:
    """Loads source content from local files or HTTP(S) URLs."""

    def load(self, source: str) -> SourceDocument:
        parsed = urlparse(source)
        if parsed.scheme in {"http", "https"}:
            return self._load_remote(source, parsed)
        if parsed.scheme == "file":
            file_path = url2pathname(unquote(parsed.path))
            if parsed.netloc:
                file_path = f"//{parsed.netloc}{file_path}"
            return self._load_local(Path(file_path))
        return self._load_local(Path(source).expanduser())

    def _load_remote(self, source: str, parsed) -> SourceDocument:
        request = Request(source, headers={"User-Agent": "rag-db/0.1"})
        with urlopen(request, timeout=60) as response:
            payload = response.read()
            media_type = response.headers.get_content_type()
        file_name = Path(unquote(parsed.path)).name or "remote_document"
        text = self._extract_text(payload, file_name=file_name, media_type=media_type)
        return SourceDocument(
            source=source,
            source_type="url",
            file_name=file_name,
            media_type=media_type,
            text=text,
        )

    def _load_local(self, path: Path) -> SourceDocument:
        if not path.exists():
            raise FileNotFoundError(f"source file does not exist: {path}")
        payload = path.read_bytes()
        media_type, _ = mimetypes.guess_type(path.name)
        text = self._extract_text(payload, file_name=path.name, media_type=media_type)
        return SourceDocument(
            source=str(path),
            source_type="path",
            file_name=path.name,
            media_type=media_type,
            text=text,
        )

    def _extract_text(self, payload: bytes, *, file_name: str, media_type: str | None) -> str:
        suffix = Path(file_name).suffix.lower()
        if suffix == ".pdf" or media_type == "application/pdf":
            return _extract_pdf_text(payload)
        if suffix in {".html", ".htm"} or media_type == "text/html":
            return _extract_html_text(payload)
        if suffix in {".txt", ".md", ".py", ".json", ".yaml", ".yml", ".csv", ".log"}:
            return _decode_text(payload)
        if media_type is not None and media_type.startswith("text/"):
            return _decode_text(payload)
        raise ValueError(f"unsupported document type for source: {file_name}")


def _extract_pdf_text(payload: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(payload))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(page.strip() for page in pages if page.strip()).strip()
    if not text:
        raise ValueError("no extractable text found in pdf")
    return text


def _extract_html_text(payload: bytes) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(_decode_text(payload))
    text = parser.get_text().strip()
    if not text:
        raise ValueError("no extractable text found in html")
    return text


def _decode_text(payload: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    return payload.decode("utf-8", errors="ignore")


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        stripped = data.strip()
        if stripped:
            self._parts.append(stripped)

    def get_text(self) -> str:
        return "\n".join(self._parts)

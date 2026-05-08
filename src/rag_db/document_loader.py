from __future__ import annotations

from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import Request, url2pathname, urlopen
import mimetypes

from rag_db.models import SourceDocument


class DocumentLoader:
    """负责把不同来源的文档统一加载为纯文本。

    当前支持三类输入：
    - 本地文件路径
    - `file://` 形式的文件 URL
    - HTTP/HTTPS 远程文件地址

    输出统一为 `SourceDocument`，供后续切块、去重和向量化流程复用。
    """

    def load(self, source: str) -> SourceDocument:
        """根据来源类型选择本地或远程加载逻辑。"""
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
        """下载远程文件并抽取文本内容。"""
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
        """读取本地文件并抽取文本内容。"""
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
        """按文件后缀或媒体类型选择文本抽取策略。"""
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
    """从 PDF 二进制内容中抽取可读文本。"""
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(payload))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(page.strip() for page in pages if page.strip()).strip()
    if not text:
        raise ValueError("no extractable text found in pdf")
    return text


def _extract_html_text(payload: bytes) -> str:
    """从 HTML 中剥离标签，仅保留可见文本。"""
    parser = _HTMLTextExtractor()
    parser.feed(_decode_text(payload))
    text = parser.get_text().strip()
    if not text:
        raise ValueError("no extractable text found in html")
    return text


def _decode_text(payload: bytes) -> str:
    """按常见中文和 UTF 编码顺序尝试解码文本。"""
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    return payload.decode("utf-8", errors="ignore")


class _HTMLTextExtractor(HTMLParser):
    """最小 HTML 文本提取器。

    这里只关心提取正文文本，不做复杂的 DOM 语义分析。
    """

    def __init__(self) -> None:
        """初始化内部文本缓冲区。"""
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        """收集非空文本节点。"""
        stripped = data.strip()
        if stripped:
            self._parts.append(stripped)

    def get_text(self) -> str:
        """将收集到的文本按换行拼接为单个字符串。"""
        return "\n".join(self._parts)

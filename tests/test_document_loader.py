from pathlib import Path

from rag_db.document_loader import DocumentLoader


def test_document_loader_reads_local_text(tmp_path: Path) -> None:
    target = tmp_path / "demo.txt"
    target.write_text("hello world", encoding="utf-8")

    document = DocumentLoader().load(str(target))

    assert document.source_type == "path"
    assert document.file_name == "demo.txt"
    assert document.text == "hello world"


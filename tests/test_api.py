from fastapi.testclient import TestClient

from quantum_atomic_rag.api import app


def test_text_upload_creates_memory_atom(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MEMORY_DATABASE_PATH", str(tmp_path / "memory.sqlite3"))

    with TestClient(app) as client:
        response = client.post(
            "/api/knowledge/upload",
            files={"file": ("notes.md", b"A durable retrieval note", "text/markdown")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "notes.md"
    assert body["node"]["content"] == "A durable retrieval note"

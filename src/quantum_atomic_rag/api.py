"""Browser-facing API for the Quantum Atomic RAG workflow."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, cast

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from .client import Gemma4VLLMClient
from .config import RuntimeSettings
from .memory import KnowledgeStore
from .orchestrator import SwarmOrchestrator
from .schemas import UniversalAgentPayload

_STATIC_INDEX = Path(__file__).with_name("static") / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = RuntimeSettings()
    client = Gemma4VLLMClient(settings)
    await client.start()
    app.state.settings = settings
    app.state.client = client
    app.state.orchestrator = SwarmOrchestrator(client)
    app.state.memory = KnowledgeStore(settings.memory_database_path)
    try:
        yield
    finally:
        await client.close()


app = FastAPI(
    title="Quantum Atomic RAG",
    description="Browser API for the staged multi-agent orchestration workflow.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/api/health")
async def health(request: Request) -> dict[str, str]:
    settings: RuntimeSettings = request.app.state.settings
    return {
        "status": "ok",
        "model": settings.vllm_model_name,
        "endpoint": str(settings.vllm_endpoint_url),
    }


@app.post("/api/analyze")
async def analyze(payload: UniversalAgentPayload, request: Request) -> dict[str, Any]:
    orchestrator: SwarmOrchestrator = request.app.state.orchestrator
    try:
        result = await orchestrator.route_task(payload)
    except Exception as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    return result.model_dump(mode="json")


@app.post("/api/knowledge")
async def add_knowledge(request: Request) -> dict[str, Any]:
    memory = cast(KnowledgeStore, request.app.state.memory)
    body = await request.json()
    content = body.get("content", "")
    if not isinstance(content, str) or not content.strip():
        raise HTTPException(status_code=422, detail="content is required")
    try:
        node = memory.ingest(
            content,
            source=str(body.get("source", "manual")),
            metadata=body.get("metadata") if isinstance(body.get("metadata"), dict) else {},
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {"node": node.model_dump(mode="json"), "total_nodes": memory.count()}


@app.post("/api/knowledge/upload")
async def upload_knowledge(
    request: Request, file: Annotated[UploadFile, File()]
) -> dict[str, Any]:
    filename = file.filename or "uploaded-source"
    extension = Path(filename).suffix.lower()
    raw = await file.read(5_000_001)
    if len(raw) > 5_000_000:
        raise HTTPException(status_code=413, detail="file exceeds the 5 MiB upload limit")
    try:
        if extension == ".pdf":
            from io import BytesIO

            from pypdf import PdfReader

            pages = PdfReader(BytesIO(raw)).pages
            content = "\n\n".join(page.extract_text() or "" for page in pages).strip()
        elif extension in {".txt", ".md", ".markdown", ".csv", ".json", ".yaml", ".yml", ".xml", ".html"}:
            content = raw.decode("utf-8-sig").strip()
        elif extension in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp3", ".wav", ".m4a", ".mp4", ".mov"}:
            content = f"Uploaded multimodal source: {filename}. Content requires a multimodal extraction worker."
        else:
            raise HTTPException(status_code=415, detail="unsupported format; use text, PDF, JSON, CSV, image, audio, or video")
    except HTTPException:
        raise
    except UnicodeDecodeError as error:
        raise HTTPException(status_code=415, detail="text source must be UTF-8 encoded") from error
    except Exception as error:
        raise HTTPException(status_code=422, detail=f"could not extract {filename}: {error}") from error
    if not content:
        raise HTTPException(status_code=422, detail="uploaded source contains no extractable text")
    memory = cast(KnowledgeStore, request.app.state.memory)
    node = memory.ingest(content, source=filename, metadata={"filename": filename, "content_type": file.content_type})
    return {"node": node.model_dump(mode="json"), "filename": filename, "total_nodes": memory.count()}


@app.post("/api/query")
async def query_memory(request: Request) -> dict[str, Any]:
    memory = cast(KnowledgeStore, request.app.state.memory)
    body = await request.json()
    query = body.get("query", "")
    if not isinstance(query, str) or not query.strip():
        raise HTTPException(status_code=422, detail="query is required")
    return memory.search(query).model_dump(mode="json")


@app.get("/api/memory/graph")
async def memory_graph(request: Request) -> dict[str, Any]:
    memory = cast(KnowledgeStore, request.app.state.memory)
    return cast(dict[str, Any], memory.graph())


@app.get("/", include_in_schema=False)
async def dashboard() -> FileResponse:
    return FileResponse(_STATIC_INDEX)

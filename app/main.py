from __future__ import annotations

import os
import uuid
from datetime import datetime
from typing import Any

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.models import FileRecord, RetrievalLog
from app.pipeline import chunk_document, embed, parse_content
from app.store import InMemoryStore, bm25_like_score, cosine_similarity

app = FastAPI(title="Deep Research API", version="0.1.0")
store = InMemoryStore()


class ProjectCreate(BaseModel):
    name: str


class QueryFilters(BaseModel):
    fileIds: list[str] = Field(default_factory=list)
    fileTypes: list[str] = Field(default_factory=list)
    dateRange: list[str] | None = None


class QueryRequest(BaseModel):
    projectId: str
    userId: str
    query: str
    topK: int = 10
    useWeb: bool = False
    filters: QueryFilters = Field(default_factory=QueryFilters)


class PinRequest(BaseModel):
    chunkId: str


class ProviderRequest(BaseModel):
    llmProvider: str
    embeddingProvider: str
    allowWeb: bool = False


@app.post("/api/projects")
def create_project(body: ProjectCreate, x_user_id: str = Header(...)) -> dict[str, str]:
    project = store.create_project(owner_id=x_user_id, name=body.name)
    return {"projectId": project.id, "name": project.name}


@app.post("/api/upload")
async def upload_file(projectId: str, file: UploadFile = File(...), x_user_id: str = Header(...)) -> dict[str, Any]:
    try:
        store.ensure_project_access(x_user_id, projectId)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e

    fid = str(uuid.uuid4())
    os.makedirs(store.blob_root, exist_ok=True)
    local_path = os.path.join(store.blob_root, f"{fid}-{file.filename}")
    content = await file.read()
    with open(local_path, "wb") as f:
        f.write(content)

    record = FileRecord(
        id=fid,
        owner_id=x_user_id,
        project_id=projectId,
        filename=file.filename,
        size=len(content),
        mime_type=file.content_type or "application/octet-stream",
        upload_timestamp=datetime.utcnow(),
        parsed=False,
        local_path=local_path,
    )
    store.upsert_file(record)
    return {"fileId": fid, "parseStatus": "pending"}


@app.get("/api/files")
def list_files(projectId: str, x_user_id: str = Header(...)) -> list[dict[str, Any]]:
    try:
        store.ensure_project_access(x_user_id, projectId)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e
    return [
        {
            "id": f.id,
            "filename": f.filename,
            "mimeType": f.mime_type,
            "parsed": f.parsed,
            "parseDetails": f.parse_details,
        }
        for f in store.files.values()
        if f.project_id == projectId
    ]


@app.post("/api/ingest/{file_id}")
def ingest_file(file_id: str, x_user_id: str = Header(...)) -> dict[str, Any]:
    file_record = store.files.get(file_id)
    if not file_record:
        raise HTTPException(status_code=404, detail="File not found")
    try:
        store.ensure_project_access(x_user_id, file_record.project_id)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e

    text, details = parse_content(file_record.local_path, file_record.mime_type)
    chunks = chunk_document(file_record.id, file_record.filename, file_record.upload_timestamp, text)
    store.add_chunks(file_record.project_id, chunks)
    file_record.parsed = True
    file_record.parse_details = details | {"chunkCount": len(chunks)}
    return {"fileId": file_id, "chunks": len(chunks), "parseDetails": file_record.parse_details}


@app.post("/api/query")
def query(body: QueryRequest, x_user_id: str = Header(...)) -> dict[str, Any]:
    if x_user_id != body.userId:
        raise HTTPException(status_code=403, detail="User mismatch")
    try:
        store.ensure_project_access(body.userId, body.projectId)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e

    query_emb = embed(body.query)
    candidate_ids = list(store.project_chunks[body.projectId])
    pinned = store.pinned_chunks[body.projectId]

    scored = []
    for cid in candidate_ids:
        c = store.chunks[cid]
        fr = store.files[c.file_id]
        if body.filters.fileIds and c.file_id not in body.filters.fileIds:
            continue
        if body.filters.fileTypes and fr.mime_type not in body.filters.fileTypes:
            continue
        dense = cosine_similarity(query_emb, c.embedding)
        sparse = bm25_like_score(body.query, c.text)
        pin_boost = 0.1 if cid in pinned else 0.0
        score = 0.7 * dense + 0.3 * sparse + pin_boost
        if score > 0.05:
            scored.append((score, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[: body.topK]

    sources = []
    evidence_lines = []
    for i, (score, c) in enumerate(top, start=1):
        f = store.files[c.file_id]
        snippet = c.text[:220]
        sources.append(
            {
                "id": c.id,
                "fileId": c.file_id,
                "filename": f.filename,
                "page": c.page,
                "paragraphIndex": c.paragraph_index,
                "startOffset": c.start_offset,
                "endOffset": c.end_offset,
                "snippet": snippet,
                "score": round(score, 4),
            }
        )
        evidence_lines.append(f"[{i}] {f.filename} p.{c.page} ¶{c.paragraph_index}: {snippet}")

    if not sources:
        answer = "I don't know based on the currently indexed project corpus."
        caveat = "No retrieved chunks passed citation threshold."
    else:
        answer = "\n".join([f"- {s['snippet']}" for s in sources[:3]])
        caveat = "Derived from local corpus retrieval."

    if body.useWeb and store.settings.get(body.userId, {}).get("allowWeb", False):
        sources.append(
            {
                "id": "web-1",
                "fileId": None,
                "filename": "web",
                "page": None,
                "snippet": "External web augmentation is enabled; connect search provider in production.",
                "score": 0.5,
                "url": "https://example.com/search-placeholder",
            }
        )

    store.log_retrieval(
        RetrievalLog(
            user_id=body.userId,
            project_id=body.projectId,
            query=body.query,
            retrieved_chunk_ids=[c.id for _, c in top],
            timestamp=datetime.utcnow(),
        )
    )

    return {
        "answerText": answer,
        "sources": sources,
        "evidence": evidence_lines,
        "confidence": round(sum(s[0] for s in top) / max(len(top), 1), 3),
        "caveats": caveat,
        "followups": ["Refine by file type", "Pin best evidence chunks", "Enable web augmentation if needed"],
        "rawModelOutput": "rule-based placeholder",
    }


@app.get("/api/files/{file_id}/preview")
def preview_file(file_id: str, pageno: int = 1, offset: int = 0, x_user_id: str = Header(...)) -> dict[str, Any]:
    record = store.files.get(file_id)
    if not record:
        raise HTTPException(status_code=404, detail="File not found")
    try:
        store.ensure_project_access(x_user_id, record.project_id)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e

    text, _ = parse_content(record.local_path, record.mime_type)
    start = max(offset, 0)
    return {"fileId": file_id, "page": pageno, "offset": start, "snippet": text[start : start + 500]}


@app.post("/api/project/{project_id}/pin-chunk")
def pin_chunk(project_id: str, body: PinRequest, x_user_id: str = Header(...)) -> dict[str, str]:
    try:
        store.ensure_project_access(x_user_id, project_id)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e
    if body.chunkId not in store.chunks:
        raise HTTPException(status_code=404, detail="Chunk not found")
    store.pinned_chunks[project_id].add(body.chunkId)
    return {"status": "pinned", "chunkId": body.chunkId}


@app.post("/api/settings/provider")
def set_provider(body: ProviderRequest, x_user_id: str = Header(...)) -> dict[str, Any]:
    store.settings[x_user_id] = body.model_dump()
    return {"status": "ok", "settings": store.settings[x_user_id]}


@app.delete("/api/files/{file_id}")
def delete_file(file_id: str, x_user_id: str = Header(...)) -> dict[str, str]:
    record = store.files.get(file_id)
    if not record:
        raise HTTPException(status_code=404, detail="File not found")
    try:
        store.ensure_project_access(x_user_id, record.project_id)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e
    store.remove_file(file_id)
    return {"status": "deleted", "fileId": file_id}

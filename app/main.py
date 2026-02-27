from __future__ import annotations

import os
import uuid
from datetime import datetime
from typing import Any, Literal

from fastapi import FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.models import FileRecord, RetrievalLog
from app.pipeline import chunk_document, embed, parse_content
from app.store import InMemoryStore, bm25_like_score, cosine_similarity

app = FastAPI(title="Deep Research API", version="0.2.0")
store = InMemoryStore()
MAX_FILE_SIZE = 25 * 1024 * 1024
ALLOWED_EXTENSIONS = {".txt", ".md", ".csv", ".pdf", ".docx", ".xlsx", ".xls", ".png", ".jpg", ".jpeg", ".heic"}

SOCIALS = {
    "instagram": "https://www.instagram.com/girish_lade_/",
    "linkedin": "https://www.linkedin.com/in/girish-lade-075bba201/",
    "github": "https://github.com/girishlade111",
    "codepen": "https://codepen.io/Girish-Lade-the-looper",
    "email": "admin@ladestack.in",
    "website": "https://ladestack.in",
}


class ProjectCreate(BaseModel):
    name: str = Field(min_length=2, max_length=120)


class ShareProjectRequest(BaseModel):
    targetUserId: str
    role: Literal["read", "query", "write"]


class QueryFilters(BaseModel):
    fileIds: list[str] = Field(default_factory=list)
    fileTypes: list[str] = Field(default_factory=list)
    dateRange: list[str] | None = None


class QueryRequest(BaseModel):
    projectId: str
    userId: str
    query: str = Field(min_length=2, max_length=2000)
    topK: int = Field(default=10, ge=1, le=50)
    useWeb: bool = False
    filters: QueryFilters = Field(default_factory=QueryFilters)


class PinRequest(BaseModel):
    chunkId: str


class ProviderRequest(BaseModel):
    llmProvider: str
    embeddingProvider: str
    allowWeb: bool = False


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return """<!doctype html><html><head><meta charset='utf-8'><title>Deep Research</title>
<style>
body{margin:0;font-family:Inter,Arial;background:#0b1220;color:#e5e7eb} .wrap{max-width:1180px;margin:0 auto;padding:24px}
.top{display:flex;justify-content:space-between;align-items:center}.logo{font-weight:700;font-size:22px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:18px}.card{background:#111827;border:1px solid #1f2937;border-radius:12px;padding:16px}
input,button,textarea{border-radius:10px;border:1px solid #334155;background:#0f172a;color:#e5e7eb;padding:10px;width:100%;box-sizing:border-box}
button{background:#2563eb;border:none;cursor:pointer;font-weight:600}button:hover{background:#1d4ed8}
.sources{margin-top:10px;font-size:13px;line-height:1.5}.social{display:flex;gap:10px;flex-wrap:wrap;margin-top:20px}
.icon{display:flex;align-items:center;gap:8px;padding:8px 10px;background:#0f172a;border:1px solid #334155;border-radius:999px;color:#cbd5e1;text-decoration:none}
small{color:#94a3b8}
</style></head><body><div class='wrap'>
<div class='top'><div class='logo'>🔎 Deep Research Enterprise</div><small>Secure RAG over private project files</small></div>
<div class='grid'>
<div class='card'>
<h3>Create Project</h3><input id='uid' placeholder='User ID' value='u1'><br><br>
<input id='pname' placeholder='Project name' value='Enterprise Research'><br><br><button onclick='createProject()'>Create Project</button>
<p id='projectOut'></p><h3>Upload File</h3><input id='file' type='file'><br><br><button onclick='upload()'>Upload</button><p id='uploadOut'></p>
</div>
<div class='card'>
<h3>Ask Research Question</h3><textarea id='query' rows='4' placeholder='Ask question based on indexed corpus...'></textarea><br><br>
<button onclick='ask()'>Ask</button><p id='answer'></p><div class='sources' id='sources'></div>
</div></div>
<div class='social'>
<a class='icon' href='https://www.instagram.com/girish_lade_/' target='_blank'>📸 Instagram</a>
<a class='icon' href='https://www.linkedin.com/in/girish-lade-075bba201/' target='_blank'>💼 LinkedIn</a>
<a class='icon' href='https://github.com/girishlade111' target='_blank'>🐙 GitHub</a>
<a class='icon' href='https://codepen.io/Girish-Lade-the-looper' target='_blank'>🧪 CodePen</a>
<a class='icon' href='mailto:admin@ladestack.in'>✉️ admin@ladestack.in</a>
<a class='icon' href='https://ladestack.in' target='_blank'>🌐 ladestack.in</a>
</div>
<script>
let projectId=''; let fileId='';
async function createProject(){const uid=document.getElementById('uid').value;const res=await fetch('/api/projects',{method:'POST',headers:{'content-type':'application/json','x-user-id':uid},body:JSON.stringify({name:document.getElementById('pname').value})});const j=await res.json();projectId=j.projectId;document.getElementById('projectOut').textContent='Project: '+projectId;}
async function upload(){const uid=document.getElementById('uid').value;const f=document.getElementById('file').files[0];const fd=new FormData();fd.append('file',f);const up=await fetch('/api/upload?projectId='+projectId,{method:'POST',headers:{'x-user-id':uid},body:fd});const uj=await up.json();fileId=uj.fileId;await fetch('/api/ingest/'+fileId,{method:'POST',headers:{'x-user-id':uid}});document.getElementById('uploadOut').textContent='Uploaded + indexed: '+fileId;}
async function ask(){const uid=document.getElementById('uid').value;const res=await fetch('/api/query',{method:'POST',headers:{'content-type':'application/json','x-user-id':uid},body:JSON.stringify({projectId,userId:uid,query:document.getElementById('query').value,topK:10,useWeb:false,filters:{}})});const j=await res.json();document.getElementById('answer').textContent=j.answerText;document.getElementById('sources').innerHTML=(j.sources||[]).map((s,i)=>`<div><b>[${i+1}]</b> ${s.filename} p.${s.page??'-'} ¶${s.paragraphIndex??'-'} — ${s.snippet}</div>`).join('');}
</script></div></body></html>"""


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "deep-research-api"}


@app.get("/api/me/profiles")
def profiles() -> dict[str, str]:
    return SOCIALS


@app.post("/api/projects")
def create_project(body: ProjectCreate, x_user_id: str = Header(...)) -> dict[str, str]:
    project = store.create_project(owner_id=x_user_id, name=body.name)
    return {"projectId": project.id, "name": project.name}


@app.post("/api/project/{project_id}/share")
def share_project(project_id: str, body: ShareProjectRequest, x_user_id: str = Header(...)) -> dict[str, str]:
    p = store.projects.get(project_id)
    if not p:
        raise HTTPException(status_code=404, detail="Project not found")
    if p.owner_id != x_user_id:
        raise HTTPException(status_code=403, detail="Only owner can share project")
    store.share_project(project_id, body.targetUserId, body.role)
    return {"status": "shared", "projectId": project_id, "targetUserId": body.targetUserId, "role": body.role}


@app.post("/api/upload")
async def upload_file(projectId: str, file: UploadFile = File(...), x_user_id: str = Header(...)) -> dict[str, Any]:
    try:
        store.ensure_project_access(x_user_id, projectId, required="write")
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e

    extension = os.path.splitext(file.filename or "")[1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    fid = str(uuid.uuid4())
    os.makedirs(store.blob_root, exist_ok=True)
    local_path = os.path.join(store.blob_root, f"{fid}-{file.filename}")
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large")
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
    job = store.create_job(file_id=fid, project_id=projectId)
    return {"fileId": fid, "parseStatus": "pending", "ingestionJobId": job.id}


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


@app.get("/api/ingest/jobs/{job_id}")
def ingest_status(job_id: str, x_user_id: str = Header(...)) -> dict[str, Any]:
    job = store.jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    store.ensure_project_access(x_user_id, job.project_id)
    return job.__dict__


@app.post("/api/ingest/{file_id}")
def ingest_file(file_id: str, x_user_id: str = Header(...)) -> dict[str, Any]:
    file_record = store.files.get(file_id)
    if not file_record:
        raise HTTPException(status_code=404, detail="File not found")
    try:
        store.ensure_project_access(x_user_id, file_record.project_id, required="write")
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e

    linked_jobs = [j for j in store.jobs.values() if j.file_id == file_id]
    current_job = linked_jobs[-1] if linked_jobs else store.create_job(file_id=file_id, project_id=file_record.project_id)
    current_job.status = "running"

    try:
        text, details = parse_content(file_record.local_path, file_record.mime_type)
        chunks = chunk_document(file_record.id, file_record.filename, file_record.upload_timestamp, text)
        store.add_chunks(file_record.project_id, chunks)
        file_record.parsed = True
        file_record.parse_details = details | {"chunkCount": len(chunks)}
        current_job.status = "completed"
        return {"fileId": file_id, "chunks": len(chunks), "parseDetails": file_record.parse_details}
    except Exception as err:
        current_job.status = "failed"
        current_job.error = str(err)
        current_job.retries += 1
        raise HTTPException(status_code=500, detail="Ingestion failed") from err


@app.post("/api/query")
def query(body: QueryRequest, x_user_id: str = Header(...)) -> dict[str, Any]:
    if x_user_id != body.userId:
        raise HTTPException(status_code=403, detail="User mismatch")
    try:
        store.ensure_project_access(body.userId, body.projectId, required="query")
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e

    query_emb = embed(body.query)
    candidate_ids = list(store.project_chunks[body.projectId])
    pinned = store.pinned_chunks[body.projectId]

    scored: list[tuple[float, Any]] = []
    for cid in candidate_ids:
        c = store.chunks[cid]
        fr = store.files[c.file_id]
        if body.filters.fileIds and c.file_id not in body.filters.fileIds:
            continue
        if body.filters.fileTypes and fr.mime_type not in body.filters.fileTypes:
            continue
        if body.filters.dateRange and len(body.filters.dateRange) == 2:
            start, end = body.filters.dateRange
            ts = fr.upload_timestamp.isoformat()
            if not (start <= ts <= end):
                continue
        dense = cosine_similarity(query_emb, c.embedding)
        sparse = bm25_like_score(body.query, c.text)
        pin_boost = 0.1 if cid in pinned else 0.0
        score = 0.7 * dense + 0.3 * sparse + pin_boost
        if score >= 0.08:
            scored.append((score, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[: body.topK]

    sources = []
    evidence_lines = []
    for i, (score, c) in enumerate(top, start=1):
        f = store.files[c.file_id]
        snippet = c.text[:220]
        src = {
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
        sources.append(src)
        evidence_lines.append(f"[{i}] {f.filename} p.{c.page} ¶{c.paragraph_index}: {snippet}")

    if not sources:
        answer = "I don't know based on the currently indexed project corpus."
        caveat = "No retrieved chunks passed citation threshold."
    else:
        answer = "\n".join([f"- {s['snippet']}" for s in sources[:3]])
        caveat = "Derived from local corpus retrieval with citation threshold checks."

    if body.useWeb and store.settings.get(body.userId, {}).get("allowWeb", False):
        sources.append(
            {
                "id": "web-1",
                "fileId": None,
                "filename": "web",
                "page": None,
                "snippet": "External web augmentation is enabled; connect production search provider.",
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
            use_web=body.useWeb,
        )
    )

    return {
        "answerText": answer,
        "sources": sources,
        "evidence": evidence_lines,
        "confidence": round(sum(s[0] for s in top) / max(len(top), 1), 3),
        "caveats": caveat,
        "followups": ["Refine by file/date filters", "Pin best evidence chunks", "Enable web augmentation if policy allows"],
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
        store.ensure_project_access(x_user_id, project_id, required="query")
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


@app.get("/api/audit/retrievals")
def audit_retrievals(projectId: str = Query(...), x_user_id: str = Header(...)) -> list[dict[str, Any]]:
    store.ensure_project_access(x_user_id, projectId, required="owner")
    return [
        {
            "userId": l.user_id,
            "query": l.query,
            "retrievedChunkIds": l.retrieved_chunk_ids,
            "useWeb": l.use_web,
            "timestamp": l.timestamp.isoformat() + "Z",
        }
        for l in store.logs
        if l.project_id == projectId
    ]


@app.delete("/api/files/{file_id}")
def delete_file(file_id: str, x_user_id: str = Header(...)) -> dict[str, str]:
    record = store.files.get(file_id)
    if not record:
        raise HTTPException(status_code=404, detail="File not found")
    try:
        store.ensure_project_access(x_user_id, record.project_id, required="write")
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e
    store.remove_file(file_id)
    return {"status": "deleted", "fileId": file_id}

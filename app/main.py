from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from fastapi import FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.answer_engine import compute_confidence, generate_answer, generate_followups
from app.embeddings import embed, refit
from app.models import FileRecord, RetrievalLog
from app.pipeline import chunk_document, parse_content
from app.skills.comparator import compare_documents, find_contradictions
from app.skills.conversation import ConversationManager, rewrite_query
from app.skills.extractor import extract_entities, extract_key_facts, extract_topics
from app.skills.summarizer import extract_key_points, summarize_chunks, summarize_document
from app.skills.web_search import WebSearchProvider, format_web_results
from app.store import InMemoryStore, bm25_like_score, cosine_similarity

app = FastAPI(title="Deep Research API", version="0.2.0")
store = InMemoryStore()
conversation_manager = ConversationManager()
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
    conversationId: str | None = None


class PinRequest(BaseModel):
    chunkId: str


class ProviderRequest(BaseModel):
    llmProvider: str
    embeddingProvider: str
    allowWeb: bool = False


class SummarizeRequest(BaseModel):
    projectId: str
    fileIds: list[str] = Field(default_factory=list)
    maxSentences: int = Field(default=10, ge=1, le=50)


class CompareRequest(BaseModel):
    projectId: str
    fileIds: list[str] = Field(min_length=2)


class ExtractRequest(BaseModel):
    projectId: str
    fileIds: list[str] = Field(default_factory=list)
    extractionType: Literal["entities", "facts", "topics"]


class ConversationStartRequest(BaseModel):
    projectId: str


class ConversationMessageRequest(BaseModel):
    conversationId: str
    message: str = Field(min_length=1, max_length=2000)


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
        upload_timestamp=datetime.now(UTC),
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
    try:
        store.ensure_project_access(x_user_id, job.project_id)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e
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

        # Refit the embedding system and reindex all chunks
        all_texts = store.get_all_chunk_texts(file_record.project_id)
        refit(all_texts)
        store.reindex_embeddings(file_record.project_id)

        file_record.parsed = True
        file_record.parse_details = details | {"chunkCount": len(chunks)}
        current_job.status = "completed"
        return {"fileId": file_id, "chunks": len(chunks), "parseDetails": file_record.parse_details}
    except Exception as err:
        current_job.status = "failed"
        current_job.error = str(err)
        current_job.retries += 1
        raise HTTPException(status_code=500, detail="Ingestion failed") from err


def _dynamic_threshold(scores: list[float]) -> float:
    """Compute a dynamic score threshold based on score distribution.

    Uses mean - 1 standard deviation of top scores as the threshold,
    with a minimum of 0.01 to avoid filtering nothing.
    """
    if not scores:
        return 0.01
    if len(scores) == 1:
        return scores[0] * 0.5

    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    std_dev = variance**0.5
    threshold = mean - std_dev

    # Floor at 0.01, ceiling at mean
    return max(0.01, min(threshold, mean))


@app.post("/api/query")
def query(body: QueryRequest, x_user_id: str = Header(...)) -> dict[str, Any]:
    if x_user_id != body.userId:
        raise HTTPException(status_code=403, detail="User mismatch")
    try:
        store.ensure_project_access(body.userId, body.projectId, required="query")
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e

    # Optionally rewrite query using conversation context
    effective_query = body.query
    if body.conversationId:
        conv = conversation_manager.get_conversation(body.conversationId)
        if conv and conv.project_id == body.projectId:
            history = conversation_manager.get_context(body.conversationId, max_turns=6)
            effective_query = rewrite_query(body.query, history)

    query_emb = embed(effective_query)
    candidate_ids = list(store.project_chunks[body.projectId])
    pinned = store.pinned_chunks[body.projectId]

    scored: list[tuple[float, Any]] = []
    all_scores: list[float] = []

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
        sparse = bm25_like_score(effective_query, c.text)
        pin_boost = 0.1 if cid in pinned else 0.0
        score = 0.7 * dense + 0.3 * sparse + pin_boost
        all_scores.append(score)
        scored.append((score, c))

    # Apply dynamic threshold instead of fixed 0.08
    threshold = _dynamic_threshold(all_scores)
    scored = [(s, c) for s, c in scored if s >= threshold]
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
        evidence_lines.append(f"[{i}] {f.filename} p.{c.page} para.{c.paragraph_index}: {snippet}")

    # Use answer engine for synthesis
    if not sources:
        answer = "I don't have enough information in the indexed corpus to answer this question."
        caveat = "No retrieved chunks passed the citation threshold."
    else:
        answer = generate_answer(effective_query, top)
        caveat = "Derived from local corpus retrieval with extractive summarization."

    # Generate follow-up suggestions
    followups = generate_followups(effective_query, top)

    # Compute confidence from score distribution
    top_scores = [s for s, _ in top]
    confidence = compute_confidence(top_scores)

    # Web search integration - use real DuckDuckGo search when enabled
    if body.useWeb and store.settings.get(body.userId, {}).get("allowWeb", False):
        try:
            web_provider = WebSearchProvider(timeout=8)
            web_results = web_provider.search(effective_query, num_results=5)
            for j, wr in enumerate(web_results):
                sources.append(
                    {
                        "id": f"web-{j + 1}",
                        "fileId": None,
                        "filename": "web",
                        "page": None,
                        "snippet": f"{wr.title}: {wr.snippet}",
                        "score": 0.5,
                        "url": wr.url,
                    }
                )
            if web_results:
                formatted = format_web_results(web_results)
                evidence_lines.append(f"[Web] {formatted[:300]}")
        except Exception:
            # Gracefully handle web search failures
            pass

    store.log_retrieval(
        RetrievalLog(
            user_id=body.userId,
            project_id=body.projectId,
            query=body.query,
            retrieved_chunk_ids=[c.id for _, c in top],
            timestamp=datetime.now(UTC),
            use_web=body.useWeb,
        )
    )

    return {
        "answerText": answer,
        "sources": sources,
        "evidence": evidence_lines,
        "confidence": confidence,
        "caveats": caveat,
        "followups": followups,
        "rawModelOutput": "extractive-summarization-v1",
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
    try:
        store.ensure_project_access(x_user_id, projectId, required="owner")
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e
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


# --------------- Skills endpoints ---------------


def _get_project_file_texts(project_id: str, file_ids: list[str] | None = None) -> list[tuple[str, str]]:
    """Helper to get (filename, text) for files in a project.

    Args:
        project_id: Project to load files from.
        file_ids: Optional filter; if empty/None, all project files are used.

    Returns:
        List of (filename, full_text) tuples.
    """
    results: list[tuple[str, str]] = []
    for f in store.files.values():
        if f.project_id != project_id:
            continue
        if file_ids and f.id not in file_ids:
            continue
        if not f.parsed:
            continue
        text, _ = parse_content(f.local_path, f.mime_type)
        results.append((f.filename, text))
    return results


def _get_project_chunk_texts(project_id: str, file_ids: list[str] | None = None) -> list[str]:
    """Helper to get chunk texts for a project."""
    texts: list[str] = []
    for cid in store.project_chunks.get(project_id, set()):
        chunk = store.chunks.get(cid)
        if chunk is None:
            continue
        if file_ids and chunk.file_id not in file_ids:
            continue
        texts.append(chunk.text)
    return texts


@app.post("/api/summarize")
def summarize(body: SummarizeRequest, x_user_id: str = Header(...)) -> dict[str, Any]:
    """Summarize documents in a project using extractive summarization."""
    try:
        store.ensure_project_access(x_user_id, body.projectId, required="query")
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e

    file_texts = _get_project_file_texts(body.projectId, body.fileIds or None)
    if not file_texts:
        # Fall back to chunks if no parseable files found
        chunk_texts = _get_project_chunk_texts(body.projectId, body.fileIds or None)
        if not chunk_texts:
            raise HTTPException(status_code=404, detail="No indexed content found for this project")
        summary = summarize_chunks(chunk_texts, max_sentences=body.maxSentences)
        key_points = extract_key_points(" ".join(chunk_texts), max_points=5)
    else:
        combined_text = "\n\n".join(text for _, text in file_texts)
        summary = summarize_document(combined_text, max_sentences=body.maxSentences)
        key_points = extract_key_points(combined_text, max_points=5)

    return {
        "summary": summary,
        "keyPoints": key_points,
        "documentCount": len(file_texts) if file_texts else 0,
        "maxSentences": body.maxSentences,
    }


@app.post("/api/compare")
def compare(body: CompareRequest, x_user_id: str = Header(...)) -> dict[str, Any]:
    """Compare 2+ documents to find shared and unique themes."""
    try:
        store.ensure_project_access(x_user_id, body.projectId, required="query")
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e

    if len(body.fileIds) < 2:
        raise HTTPException(status_code=400, detail="At least 2 file IDs are required for comparison")

    file_texts = _get_project_file_texts(body.projectId, body.fileIds)
    if len(file_texts) < 2:
        # Try using chunks grouped by file
        doc_texts: list[tuple[str, str]] = []
        for fid in body.fileIds:
            f = store.files.get(fid)
            if not f or f.project_id != body.projectId:
                continue
            chunks = [
                store.chunks[cid].text
                for cid in store.project_chunks.get(body.projectId, set())
                if store.chunks.get(cid) and store.chunks[cid].file_id == fid
            ]
            if chunks:
                doc_texts.append((f.filename, " ".join(chunks)))
        if len(doc_texts) < 2:
            raise HTTPException(status_code=404, detail="Not enough parsed documents found for comparison")
        file_texts = doc_texts

    comparison = compare_documents(file_texts)

    # Also find contradictions between first two documents
    chunks_a = _get_project_chunk_texts(body.projectId, [body.fileIds[0]])
    chunks_b = _get_project_chunk_texts(body.projectId, [body.fileIds[1]])
    contradictions = find_contradictions(chunks_a, chunks_b)
    comparison["contradictions"] = contradictions

    return comparison


@app.post("/api/extract")
def extract(body: ExtractRequest, x_user_id: str = Header(...)) -> dict[str, Any]:
    """Extract entities, facts, or topics from project documents."""
    try:
        store.ensure_project_access(x_user_id, body.projectId, required="query")
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e

    file_ids = body.fileIds or None
    chunk_texts = _get_project_chunk_texts(body.projectId, file_ids)

    if not chunk_texts:
        raise HTTPException(status_code=404, detail="No indexed content found for this project")

    combined_text = " ".join(chunk_texts)

    if body.extractionType == "entities":
        result = extract_entities(combined_text)
        return {"extractionType": "entities", "entities": result}
    elif body.extractionType == "facts":
        facts = extract_key_facts(combined_text, top_n=10)
        return {"extractionType": "facts", "facts": facts}
    elif body.extractionType == "topics":
        topics = extract_topics(chunk_texts, top_n=10)
        return {"extractionType": "topics", "topics": topics}
    else:
        raise HTTPException(status_code=400, detail="Invalid extractionType")


@app.post("/api/conversation/start")
def conversation_start(body: ConversationStartRequest, x_user_id: str = Header(...)) -> dict[str, str]:
    """Start a new multi-turn conversation."""
    try:
        store.ensure_project_access(x_user_id, body.projectId, required="query")
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e

    conv_id = conversation_manager.start_conversation(x_user_id, body.projectId)
    return {"conversationId": conv_id, "projectId": body.projectId}


@app.post("/api/conversation/message")
def conversation_message(body: ConversationMessageRequest, x_user_id: str = Header(...)) -> dict[str, Any]:
    """Send a message in a conversation and get an answer with context."""
    conv = conversation_manager.get_conversation(body.conversationId)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if conv.user_id != x_user_id:
        raise HTTPException(status_code=403, detail="Not your conversation")

    try:
        store.ensure_project_access(x_user_id, conv.project_id, required="query")
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e

    # Add user message to history
    conversation_manager.add_turn(body.conversationId, "user", body.message)

    # Rewrite query using conversation context
    history = conversation_manager.get_context(body.conversationId, max_turns=6)
    effective_query = rewrite_query(body.message, history[:-1])  # exclude current message

    # Perform retrieval
    query_emb = embed(effective_query)
    candidate_ids = list(store.project_chunks.get(conv.project_id, set()))

    scored: list[tuple[float, Any]] = []
    for cid in candidate_ids:
        c = store.chunks[cid]
        dense = cosine_similarity(query_emb, c.embedding)
        sparse = bm25_like_score(effective_query, c.text)
        score = 0.7 * dense + 0.3 * sparse
        scored.append((score, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:5]

    # Generate answer
    if top:
        answer = generate_answer(effective_query, top)
    else:
        answer = "I don't have enough information in the indexed corpus to answer this question."

    # Add assistant response to history
    conversation_manager.add_turn(body.conversationId, "assistant", answer)

    sources = []
    for score, c in top:
        f = store.files.get(c.file_id)
        sources.append({
            "id": c.id,
            "fileId": c.file_id,
            "filename": f.filename if f else "unknown",
            "snippet": c.text[:200],
            "score": round(score, 4),
        })

    return {
        "conversationId": body.conversationId,
        "answer": answer,
        "sources": sources,
        "rewrittenQuery": effective_query if effective_query != body.message else None,
    }


@app.get("/api/conversation/{conversation_id}/history")
def conversation_history(conversation_id: str, x_user_id: str = Header(...)) -> dict[str, Any]:
    """Get conversation history."""
    conv = conversation_manager.get_conversation(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if conv.user_id != x_user_id:
        raise HTTPException(status_code=403, detail="Not your conversation")

    try:
        store.ensure_project_access(x_user_id, conv.project_id, required="query")
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e

    history = conversation_manager.get_history(conversation_id)
    return {
        "conversationId": conversation_id,
        "projectId": conv.project_id,
        "turns": history,
    }

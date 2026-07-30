from __future__ import annotations

import math
import os
import shutil
import uuid
from collections import defaultdict
from datetime import UTC, datetime

from app.models import Chunk, FileRecord, IngestionJob, Project, RetrievalLog


class InMemoryStore:
    def __init__(self, blob_root: str = "data/blobs") -> None:
        self.projects: dict[str, Project] = {}
        self.files: dict[str, FileRecord] = {}
        self.chunks: dict[str, Chunk] = {}
        self.jobs: dict[str, IngestionJob] = {}
        self.project_chunks: dict[str, set[str]] = defaultdict(set)
        self.pinned_chunks: dict[str, set[str]] = defaultdict(set)
        self.logs: list[RetrievalLog] = []
        self.settings: dict[str, dict[str, str | bool]] = defaultdict(dict)
        self.blob_root = blob_root
        os.makedirs(blob_root, exist_ok=True)

    def create_project(self, owner_id: str, name: str) -> Project:
        pid = str(uuid.uuid4())
        project = Project(id=pid, owner_id=owner_id, name=name, members={owner_id: "owner"})
        self.projects[pid] = project
        return project

    def share_project(self, project_id: str, target_user_id: str, role: str) -> None:
        self.projects[project_id].members[target_user_id] = role

    def ensure_project_access(self, user_id: str, project_id: str, required: str = "read") -> None:
        """Check user has required access level to a project.

        Raises PermissionError with a descriptive message if access is denied.
        Handles missing projects and users gracefully.
        """
        project = self.projects.get(project_id)
        if project is None:
            raise PermissionError(f"Project '{project_id}' not found")
        if user_id not in project.members:
            raise PermissionError("User does not have access to this project")
        hierarchy = {"read": 1, "query": 2, "write": 3, "owner": 4}
        user_role = project.members[user_id]
        if hierarchy.get(user_role, 0) < hierarchy.get(required, 0):
            raise PermissionError(
                f"Insufficient role: user has '{user_role}' but '{required}' is required"
            )

    def upsert_file(self, record: FileRecord) -> None:
        self.files[record.id] = record

    def add_chunks(self, project_id: str, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            self.chunks[chunk.id] = chunk
            self.project_chunks[project_id].add(chunk.id)

    def remove_file(self, file_id: str) -> None:
        file_record = self.files.pop(file_id, None)
        if not file_record:
            return
        doomed = [
            cid
            for cid in self.project_chunks[file_record.project_id]
            if self.chunks[cid].file_id == file_id
        ]
        for cid in doomed:
            self.project_chunks[file_record.project_id].discard(cid)
            self.chunks.pop(cid, None)
            self.pinned_chunks[file_record.project_id].discard(cid)
        if file_record.local_path and os.path.exists(file_record.local_path):
            os.remove(file_record.local_path)

    def create_job(self, file_id: str, project_id: str) -> IngestionJob:
        job = IngestionJob(
            id=str(uuid.uuid4()),
            file_id=file_id,
            project_id=project_id,
            status="queued",
        )
        self.jobs[job.id] = job
        return job

    def log_retrieval(self, log: RetrievalLog) -> None:
        self.logs.append(log)

    def reindex_embeddings(self, project_id: str) -> None:
        """Re-embed all chunks for a project using the current vectorizer state.

        Called after the TF-IDF vectorizer is refitted so that all existing
        chunks get embeddings consistent with the updated vocabulary.
        """
        from app.embeddings import embed

        chunk_ids = list(self.project_chunks.get(project_id, set()))
        for cid in chunk_ids:
            chunk = self.chunks.get(cid)
            if chunk is not None:
                chunk.embedding = embed(chunk.text)

    def get_all_chunk_texts(self, project_id: str) -> list[str]:
        """Get all chunk texts for a project (used for refitting the vectorizer)."""
        texts: list[str] = []
        for cid in self.project_chunks.get(project_id, set()):
            chunk = self.chunks.get(cid)
            if chunk is not None:
                texts.append(chunk.text)
        return texts


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def bm25_like_score(query: str, text: str) -> float:
    q_tokens = set(query.lower().split())
    t_tokens = text.lower().split()
    if not q_tokens or not t_tokens:
        return 0.0
    overlap = sum(1 for t in t_tokens if t in q_tokens)
    return overlap / max(len(t_tokens), 1)


def reset_data_dir(path: str = "data") -> None:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")

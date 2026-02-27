from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Project:
    id: str
    owner_id: str
    name: str
    members: dict[str, str] = field(default_factory=dict)  # user_id -> role (owner/read/query/write)


@dataclass
class FileRecord:
    id: str
    owner_id: str
    project_id: str
    filename: str
    size: int
    mime_type: str
    upload_timestamp: datetime
    parsed: bool = False
    parse_details: dict[str, Any] = field(default_factory=dict)
    local_path: str = ""


@dataclass
class Chunk:
    id: str
    file_id: str
    text: str
    start_offset: int
    end_offset: int
    page: int | None
    paragraph_index: int
    embedding: list[float]
    metadata: dict[str, Any]


@dataclass
class RetrievalLog:
    user_id: str
    project_id: str
    query: str
    retrieved_chunk_ids: list[str]
    timestamp: datetime
    use_web: bool = False


@dataclass
class IngestionJob:
    id: str
    file_id: str
    project_id: str
    status: str
    retries: int = 0
    error: str | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)

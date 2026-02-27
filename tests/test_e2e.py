from datetime import datetime

from app.models import FileRecord
from app.pipeline import chunk_document, embed
from app.store import InMemoryStore, bm25_like_score, cosine_similarity, reset_data_dir


def setup_function() -> None:
    reset_data_dir()


def test_ingestion_retrieval_and_citation_mapping() -> None:
    store = InMemoryStore()
    project = store.create_project("u1", "alpha")

    file_record = FileRecord(
        id="f1",
        owner_id="u1",
        project_id=project.id,
        filename="report.txt",
        size=100,
        mime_type="text/plain",
        upload_timestamp=datetime.utcnow(),
    )
    store.upsert_file(file_record)

    text = "Climate report 2024\n\nRevenue increased by 22 percent in Q4.\n\nRisk remains supply chain volatility."
    chunks = chunk_document(file_record.id, file_record.filename, file_record.upload_timestamp, text)
    store.add_chunks(project.id, chunks)

    qemb = embed("What happened to revenue in Q4?")
    ranked = []
    for cid in store.project_chunks[project.id]:
        chunk = store.chunks[cid]
        score = 0.7 * cosine_similarity(qemb, chunk.embedding) + 0.3 * bm25_like_score("revenue Q4", chunk.text)
        ranked.append((score, chunk))
    ranked.sort(key=lambda x: x[0], reverse=True)

    assert ranked
    top = ranked[0][1]
    assert top.metadata["filename"] == "report.txt"
    assert top.metadata["charOffsets"][1] > top.metadata["charOffsets"][0]


def test_delete_file_removes_chunks() -> None:
    store = InMemoryStore()
    p = store.create_project("u1", "alpha")
    f = FileRecord(
        id="f2",
        owner_id="u1",
        project_id=p.id,
        filename="tiny.txt",
        size=10,
        mime_type="text/plain",
        upload_timestamp=datetime.utcnow(),
        local_path="",
    )
    store.upsert_file(f)
    chunks = chunk_document(f.id, f.filename, f.upload_timestamp, "alpha\n\nbeta")
    store.add_chunks(p.id, chunks)

    assert store.project_chunks[p.id]
    store.remove_file(f.id)
    assert not store.project_chunks[p.id]

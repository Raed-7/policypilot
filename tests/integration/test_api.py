import pytest
from fastapi.testclient import TestClient

from policypilot.api.main import create_app
from tests.conftest import SAMPLE_DOCS


@pytest.fixture
def client(indexed_pipeline):
    with TestClient(create_app(indexed_pipeline)) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["provider"] == "extractive"
    assert body["documents"] == 6
    assert body["chunks"] > 0


def test_list_documents(client):
    docs = client.get("/documents").json()
    assert {d["source"] for d in docs} >= {"employee-handbook.pdf", "remote-work-policy.md"}


def test_ask_returns_answer_with_citations(client):
    r = client.post("/ask", json={"question": "How many days of annual leave do full-time employees get?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answerable"]
    assert "25 days" in body["answer"]
    assert body["citations"] and all(c["cited"] for c in body["citations"])
    assert body["citations"][0]["source"] == "employee-handbook.pdf"
    assert body["citations"][0]["page"] == 3
    assert len(body["sources"]) >= len(body["citations"])


def test_ask_validates_input(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "x", "top_k": 100}).status_code == 422


def test_search(client):
    r = client.post("/search", json={"query": "passwords at least 14 characters", "top_k": 3})
    results = r.json()["results"]
    assert 0 < len(results) <= 3
    assert results[0]["source"] == "information-security-policy.md"


def test_upload_and_delete_document(client):
    files = [("files", ("gym-policy.md", b"# Gym Policy\n\nEmployees get a gym subsidy of \xc2\xa320 per month.", "text/markdown"))]
    r = client.post("/documents", files=files)
    assert r.status_code == 201
    doc = r.json()["documents"][0]
    assert doc["status"] == "indexed" and doc["chunks"] >= 1

    answer = client.post("/ask", json={"question": "How much is the gym subsidy per month?"}).json()
    assert "£20" in answer["answer"]
    assert answer["citations"][0]["source"] == "gym-policy.md"

    assert client.delete(f"/documents/{doc['doc_id']}").status_code == 200
    assert client.delete(f"/documents/{doc['doc_id']}").status_code == 404


def test_upload_rejects_unsupported_and_empty_files(client):
    r = client.post("/documents", files=[("files", ("data.xlsx", b"binary", "application/octet-stream"))])
    assert r.status_code == 422
    r = client.post(
        "/documents",
        files=[("files", ("empty.txt", b"   ", "text/plain")), ("files", ("ok.txt", b"Valid policy text here.", "text/plain"))],
    )
    assert r.status_code == 201
    assert r.json()["errors"][0]["source"] == "empty.txt"


def test_evaluation_endpoint_without_results(client):
    assert client.get("/evaluation").status_code == 404


def test_formats(client):
    assert ".pdf" in client.get("/formats").json()["extensions"]

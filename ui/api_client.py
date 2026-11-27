import os

import httpx

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")


class APIError(RuntimeError):
    pass


def _request(method: str, path: str, timeout: float = 120.0, **kwargs):
    try:
        response = httpx.request(method, f"{API_URL}{path}", timeout=timeout, **kwargs)
    except httpx.HTTPError as exc:
        raise APIError(f"Cannot reach the PolicyPilot API at {API_URL}") from exc
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise APIError(detail if isinstance(detail, str) else "; ".join(f"{d.get('source')}: {d.get('error')}" for d in detail))
    return response.json()


def health() -> dict:
    return _request("GET", "/health", timeout=5)


def documents() -> list[dict]:
    return _request("GET", "/documents")


def upload(files: list[tuple[str, bytes]]) -> dict:
    return _request("POST", "/documents", files=[("files", (name, data)) for name, data in files], timeout=600)


def delete(doc_id: str) -> dict:
    return _request("DELETE", f"/documents/{doc_id}")


def ask(question: str, **params) -> dict:
    return _request("POST", "/ask", json={"question": question, **{k: v for k, v in params.items() if v is not None}})


def search(query: str, **params) -> dict:
    return _request("POST", "/search", json={"query": query, **{k: v for k, v in params.items() if v is not None}})


def evaluation() -> dict | None:
    try:
        return _request("GET", "/evaluation")
    except APIError:
        return None

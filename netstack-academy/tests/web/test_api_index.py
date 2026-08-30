"""Contract for the index JSON API.

Three questions, three answers: what is indexed right now, make sure
something is indexed, and index it again from scratch. The distinction
between the last two is the whole reason this API exists -- one is what a
page needs before it can show a symbol, the other is what a learner presses
after a ``git pull``.

A failed run is reported as a *result*, not as an HTTP error. "ctags is not
installed" is information the dashboard should show, and turning it into a
500 would make the caller guess whether the request or the machine was at
fault.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from index_fixtures import index_kernel_repo
from web_fakes import RecordingOrchestrator, failed_result, reindexed_result

STALE_HEAD = "0" * 40


def test_status_is_json(client: TestClient) -> None:
    response = client.get("/api/index/status")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")


def test_status_reports_the_persisted_generation(
    client: TestClient, kernel_head: str, indexed_generation
) -> None:
    payload = client.get("/api/index/status").json()

    assert payload["indexed_head"] == kernel_head
    assert payload["symbol_count"] == indexed_generation.symbol_count
    assert payload["edge_count"] == indexed_generation.edge_count


def test_status_reports_the_repository_head(
    client: TestClient, kernel_head: str
) -> None:
    payload = client.get("/api/index/status").json()

    assert payload["repository_available"] is True
    assert payload["repository_head"] == kernel_head
    assert payload["stale"] is False


def test_status_flags_a_generation_from_another_commit(
    client: TestClient, index_storage, indexed_generation
) -> None:
    index_kernel_repo(index_storage, head=STALE_HEAD)

    payload = client.get("/api/index/status").json()

    assert payload["indexed_head"] == STALE_HEAD
    assert payload["stale"] is True


def test_status_starts_out_unensured_with_no_run_recorded(
    client: TestClient,
) -> None:
    payload = client.get("/api/index/status").json()

    assert payload["ensured"] is False
    assert payload["last_run"] is None
    assert payload["providers"] == []


def test_status_never_triggers_indexing(
    client: TestClient, orchestrator: RecordingOrchestrator
) -> None:
    client.get("/api/index/status")

    assert orchestrator.call_count == 0


def test_ensure_runs_the_pipeline_once(
    client: TestClient, orchestrator: RecordingOrchestrator
) -> None:
    first = client.post("/api/index/ensure")
    second = client.post("/api/index/ensure")

    assert first.status_code == 200
    assert second.status_code == 200
    assert orchestrator.forces == [False]


def test_ensure_reports_the_run(client: TestClient, kernel_head: str) -> None:
    payload = client.post("/api/index/ensure").json()

    assert payload["status"] == "reused"
    assert payload["head"] == kernel_head
    assert payload["ensured"] is True


def test_force_always_reruns_the_pipeline(
    client: TestClient, orchestrator: RecordingOrchestrator
) -> None:
    client.post("/api/index/ensure")
    client.post("/api/index/ensure", params={"force": "true"})

    assert orchestrator.forces == [False, True]


def test_status_reports_provider_availability_after_a_run(
    make_client, kernel_head: str
) -> None:
    """Which providers were available is only knowable from the run that
    consulted them, and it is what explains a graph with no semantic edges.
    """
    orchestrator = RecordingOrchestrator(reindexed_result(kernel_head))
    client = make_client(orchestrator_override=orchestrator)

    client.post("/api/index/ensure")
    payload = client.get("/api/index/status").json()

    providers = {entry["provider_name"]: entry for entry in payload["providers"]}
    assert providers["ctags"]["available"] is True
    assert providers["clangd"]["available"] is False
    assert "clangd" in providers["clangd"]["reason"]


def test_status_reports_the_last_run_status(make_client, kernel_head: str) -> None:
    client = make_client(
        orchestrator_override=RecordingOrchestrator(reindexed_result(kernel_head))
    )

    client.post("/api/index/ensure")

    assert client.get("/api/index/status").json()["last_run"]["status"] == "reindexed"


def test_a_failed_run_is_reported_without_an_http_error(make_client) -> None:
    client = make_client(
        orchestrator_override=RecordingOrchestrator(failed_result("ctags not found"))
    )

    response = client.post("/api/index/ensure")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "failed"
    assert "ctags not found" in payload["reason"]
    assert payload["ensured"] is False


def test_a_failed_run_is_retried_by_the_next_request(make_client) -> None:
    orchestrator = RecordingOrchestrator(failed_result("ctags not found"))
    client = make_client(orchestrator_override=orchestrator)

    client.post("/api/index/ensure")
    client.post("/api/index/ensure")

    assert orchestrator.call_count == 2


def test_ensure_is_not_a_get(client: TestClient) -> None:
    """It changes the machine's state and can take minutes; a link, a
    prefetch or a crawler must not be able to start it.
    """
    assert client.get("/api/index/ensure").status_code == 405

"""Contract for the production entry point that starts a real ``clangd``.

Everything below the factory is injectable and fake-able; the factory itself
is the one piece that decides *which* binary to launch and what to do when it
is not installed. Because the semantic provider is optional by design (see
``IndexOrchestrator``), a missing ``clangd`` must produce a provider that
reports itself unavailable -- never an exception the orchestrator would have
to guard against.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from netstack_academy.indexing.semantic import transport as transport_module
from netstack_academy.indexing.semantic.factory import create_clangd_provider
from netstack_academy.indexing.semantic.models import CallHierarchyItem
from netstack_academy.indexing.semantic.provider import SemanticProvider

from lsp_fakes import RecordingPopen


def _install_popen(
    monkeypatch: pytest.MonkeyPatch, recorder: RecordingPopen
) -> RecordingPopen:
    monkeypatch.setattr(transport_module.subprocess, "Popen", recorder)
    return recorder


def test_factory_launches_a_fixed_clangd_argv_without_a_shell(
    monkeypatch: pytest.MonkeyPatch, git_repository: Path
) -> None:
    recorder = _install_popen(
        monkeypatch, RecordingPopen(error=FileNotFoundError("no clangd"))
    )

    create_clangd_provider(git_repository)

    call = recorder.calls[0]
    assert list(call.argv)[0] == "clangd"
    assert all(isinstance(argument, str) for argument in call.argv)
    assert not call.kwargs.get("shell", False)
    assert Path(call.kwargs["cwd"]) == git_repository


def test_factory_returns_an_unavailable_provider_when_clangd_is_missing(
    monkeypatch: pytest.MonkeyPatch, git_repository: Path
) -> None:
    _install_popen(monkeypatch, RecordingPopen(error=FileNotFoundError("no clangd")))

    provider = create_clangd_provider(git_repository)

    capabilities = provider.capabilities()
    assert isinstance(provider, SemanticProvider)
    assert capabilities.available is False
    assert capabilities.provider_name == "clangd"
    assert "clangd" in (capabilities.reason or "")


def test_unavailable_provider_degrades_every_request_instead_of_raising(
    monkeypatch: pytest.MonkeyPatch, git_repository: Path
) -> None:
    _install_popen(monkeypatch, RecordingPopen(error=FileNotFoundError("no clangd")))
    provider = create_clangd_provider(git_repository)

    prepared = provider.prepare_call_hierarchy(
        "net/ipv4/tcp_input.c", line=1, column=1
    )
    referenced = provider.references("net/ipv4/tcp_input.c", line=1, column=1)
    calls = provider.outgoing_calls(
        CallHierarchyItem(
            name="tcp_input", relative_path="net/ipv4/tcp_input.c", line=1, column=1
        )
    )

    assert prepared.status == "unavailable"
    assert referenced.status == "unavailable"
    assert calls.status == "unavailable"
    assert prepared.items == ()
    assert referenced.locations == ()
    assert calls.calls == ()


def test_unavailable_provider_can_still_be_closed(
    monkeypatch: pytest.MonkeyPatch, git_repository: Path
) -> None:
    _install_popen(monkeypatch, RecordingPopen(error=FileNotFoundError("no clangd")))
    provider = create_clangd_provider(git_repository)

    provider.close()
    provider.close()


def test_factory_completes_the_handshake_against_a_started_clangd(
    monkeypatch: pytest.MonkeyPatch, git_repository: Path
) -> None:
    recorder = _install_popen(monkeypatch, RecordingPopen(serve=True))

    provider = create_clangd_provider(git_repository)

    try:
        capabilities = provider.capabilities()
        assert capabilities.available is True
        assert capabilities.provider_name == "clangd"
        server = recorder.servers[0]
        assert server.wait_for_method("initialized")
        assert server.received_methods[:2] == ["initialize", "initialized"]
    finally:
        provider.close()
        recorder.shutdown()

    assert recorder.processes[0].terminate_calls == 1

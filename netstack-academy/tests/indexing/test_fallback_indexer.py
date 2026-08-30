from __future__ import annotations

from pathlib import Path

from netstack_academy.indexing.fallback_indexer import index_fallback


def test_fallback_finds_function_definitions_in_all_files(
    duplicate_static_c_repo: Path,
) -> None:
    result = index_fallback(duplicate_static_c_repo, roots=["net"])

    names = {symbol.name for symbol in result.symbols}
    assert names == {"helper", "process", "process6", "shared_util"}


def test_fallback_marks_static_functions_as_static(
    duplicate_static_c_repo: Path,
) -> None:
    result = index_fallback(duplicate_static_c_repo, roots=["net"])

    helpers = [s for s in result.symbols if s.name == "helper"]
    assert len(helpers) == 2
    assert all(s.is_static for s in helpers)

    non_static = [s for s in result.symbols if s.name in {"process", "shared_util"}]
    assert all(not s.is_static for s in non_static)


def test_fallback_records_correct_relative_paths_and_lines(
    duplicate_static_c_repo: Path,
) -> None:
    result = index_fallback(duplicate_static_c_repo, roots=["net"])

    by_path = {(s.name, s.relative_path) for s in result.symbols}
    assert ("helper", "net/ipv4/a.c") in by_path
    assert ("helper", "net/ipv6/b.c") in by_path
    assert ("shared_util", "net/util.c") in by_path

    process = next(s for s in result.symbols if s.name == "process")
    assert process.line == 6


def test_fallback_resolves_call_to_static_helper_within_same_file(
    duplicate_static_c_repo: Path,
) -> None:
    result = index_fallback(duplicate_static_c_repo, roots=["net"])

    edge = next(
        e
        for e in result.edges
        if e.source_name == "process" and e.target_name == "helper"
    )

    assert edge.source_relative_path == "net/ipv4/a.c"
    assert edge.target_relative_path == "net/ipv4/a.c"


def test_fallback_does_not_cross_link_duplicate_static_helper(
    duplicate_static_c_repo: Path,
) -> None:
    result = index_fallback(duplicate_static_c_repo, roots=["net"])

    edge = next(
        e
        for e in result.edges
        if e.source_name == "process6" and e.target_name == "helper"
    )

    assert edge.source_relative_path == "net/ipv6/b.c"
    assert edge.target_relative_path == "net/ipv6/b.c"


def test_fallback_resolves_call_to_unique_non_static_function_across_files(
    duplicate_static_c_repo: Path,
) -> None:
    result = index_fallback(duplicate_static_c_repo, roots=["net"])

    edge = next(
        e
        for e in result.edges
        if e.source_name == "process" and e.target_name == "shared_util"
    )

    assert edge.target_relative_path == "net/util.c"


def test_fallback_edges_are_marked_heuristic(
    duplicate_static_c_repo: Path,
) -> None:
    result = index_fallback(duplicate_static_c_repo, roots=["net"])

    assert result.edges
    assert all(edge.provenance == "heuristic" for edge in result.edges)


def test_fallback_restricts_scanning_to_configured_roots(
    duplicate_static_c_repo: Path,
) -> None:
    result = index_fallback(duplicate_static_c_repo, roots=["net"])

    paths = {s.relative_path for s in result.symbols}
    assert not any(path.startswith("unrelated/") for path in paths)


def test_fallback_returns_empty_result_for_directory_without_c_files(
    tmp_path: Path,
) -> None:
    empty_repo = tmp_path / "empty"
    (empty_repo / "net").mkdir(parents=True)

    result = index_fallback(empty_repo, roots=["net"])

    assert result.symbols == []
    assert result.edges == []


def test_fallback_scans_header_files_for_definitions_under_configured_roots(
    tmp_path: Path,
) -> None:
    """Network headers commonly define ``static inline`` helpers directly
    in the ``.h`` file; the fallback scanner must not restrict itself to
    ``.c`` sources, or every such definition is silently invisible.
    """
    repo = tmp_path / "kernel"
    net_dir = repo / "net"
    net_dir.mkdir(parents=True)
    (net_dir / "proto.h").write_text(
        "static inline int helper_inline(int x)\n"
        "{\n"
        "    return x + 1;\n"
        "}\n",
        encoding="utf-8",
    )

    result = index_fallback(repo, roots=["net"])

    names = {symbol.name for symbol in result.symbols}
    assert "helper_inline" in names

    helper = next(s for s in result.symbols if s.name == "helper_inline")
    assert helper.relative_path == "net/proto.h"
    assert helper.is_static is True


def test_fallback_default_roots_exclude_unrelated_directories(
    tmp_path: Path,
) -> None:
    """With no ``roots`` argument, the fallback scanner must use the same
    curated, bounded root list as the rest of the pipeline -- never a
    whole-repo ``"."`` scan -- so an unrelated tree such as ``fs/`` is
    never touched even when it contains C sources.
    """
    repo = tmp_path / "kernel"
    net_dir = repo / "net"
    net_dir.mkdir(parents=True)
    (net_dir / "a.c").write_text(
        "int helper(int x)\n{\n    return x;\n}\n",
        encoding="utf-8",
    )

    fs_dir = repo / "fs"
    fs_dir.mkdir(parents=True)
    (fs_dir / "unrelated.c").write_text(
        "int fs_helper(int x)\n{\n    return x;\n}\n",
        encoding="utf-8",
    )

    result = index_fallback(repo)

    paths = {symbol.relative_path for symbol in result.symbols}
    assert not any(path.startswith("fs/") for path in paths)
    assert any(path.startswith("net/") for path in paths)


def test_fallback_default_roots_include_curated_single_file_header_definitions(
    tmp_path: Path,
) -> None:
    """The curated default root list names specific header *files* (e.g.
    ``include/linux/skbuff.h``), not just directories. With no ``roots``
    argument, the fallback scanner must still resolve those file roots and
    extract their ``.h`` definitions.
    """
    repo = tmp_path / "kernel"
    header_dir = repo / "include" / "linux"
    header_dir.mkdir(parents=True)
    (header_dir / "skbuff.h").write_text(
        "static inline int skb_helper(int x)\n"
        "{\n"
        "    return x + 1;\n"
        "}\n",
        encoding="utf-8",
    )

    result = index_fallback(repo)

    names = {symbol.name for symbol in result.symbols}
    assert "skb_helper" in names

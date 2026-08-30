from __future__ import annotations

import json

from netstack_academy.indexing.ctags_parser import parse_ctags_jsonlines


def _tag_line(**fields: object) -> str:
    payload = {"_type": "tag", "kind": "function", **fields}
    return json.dumps(payload)


def test_parse_extracts_definition_fields() -> None:
    line = _tag_line(
        name="tcp_input",
        path="net/ipv4/tcp_input.c",
        line=42,
        signature="(struct sock *sk, struct sk_buff *skb)",
        scope="tcp_v4_rcv",
        scopeKind="function",
    )

    result = parse_ctags_jsonlines([line])

    assert len(result.definitions) == 1
    definition = result.definitions[0]
    assert definition.name == "tcp_input"
    assert definition.kind == "function"
    assert definition.path == "net/ipv4/tcp_input.c"
    assert definition.line == 42
    assert definition.signature == "(struct sock *sk, struct sk_buff *skb)"
    assert definition.scope == "tcp_v4_rcv"
    assert result.diagnostics == []


def test_parse_marks_file_scoped_symbol_as_static() -> None:
    line = _tag_line(
        name="helper",
        path="net/ipv4/a.c",
        line=1,
        file=True,
    )

    result = parse_ctags_jsonlines([line])

    assert result.definitions[0].is_static is True


def test_parse_marks_non_file_scoped_symbol_as_not_static() -> None:
    line = _tag_line(
        name="tcp_input",
        path="net/ipv4/tcp_input.c",
        line=42,
    )

    result = parse_ctags_jsonlines([line])

    assert result.definitions[0].is_static is False


def test_parse_preserves_order_of_multiple_definitions() -> None:
    lines = [
        _tag_line(name="first", path="net/a.c", line=1),
        _tag_line(name="second", path="net/b.c", line=2),
        _tag_line(name="third", path="net/c.c", line=3),
    ]

    result = parse_ctags_jsonlines(lines)

    assert [d.name for d in result.definitions] == ["first", "second", "third"]


def test_parse_skips_non_tag_type_records_and_records_diagnostic() -> None:
    lines = [
        json.dumps({"_type": "ptag", "name": "TAG_PROGRAM_NAME", "path": "Universal Ctags"}),
        _tag_line(name="tcp_input", path="net/ipv4/tcp_input.c", line=42),
    ]

    result = parse_ctags_jsonlines(lines)

    assert len(result.definitions) == 1
    assert result.definitions[0].name == "tcp_input"
    assert len(result.diagnostics) == 1


def test_parse_skips_malformed_json_line_without_raising() -> None:
    lines = [
        "{not valid json",
        _tag_line(name="tcp_input", path="net/ipv4/tcp_input.c", line=42),
    ]

    result = parse_ctags_jsonlines(lines)

    assert len(result.definitions) == 1
    assert len(result.diagnostics) == 1
    assert "1" in result.diagnostics[0] or "json" in result.diagnostics[0].lower()


def test_parse_skips_record_missing_required_field() -> None:
    lines = [
        json.dumps({"_type": "tag", "kind": "function", "name": "incomplete"}),
        _tag_line(name="tcp_input", path="net/ipv4/tcp_input.c", line=42),
    ]

    result = parse_ctags_jsonlines(lines)

    assert len(result.definitions) == 1
    assert result.definitions[0].name == "tcp_input"
    assert len(result.diagnostics) == 1


def test_parse_empty_input_returns_empty_result() -> None:
    result = parse_ctags_jsonlines([])

    assert result.definitions == []
    assert result.diagnostics == []


def test_parse_never_raises_for_arbitrary_garbage_lines() -> None:
    lines = ["", "   ", "null", "[]", "42", '"just a string"']

    result = parse_ctags_jsonlines(lines)

    assert result.definitions == []

# Task 2 — Important hardening findings: RED test coverage report

Scope: write failing (RED) tests only for the five remaining Task 2
"Important" hardening findings below. No production code was changed. No
tests were run against a modified implementation — all RED tests below
fail against the current `HEAD` of this branch and are expected to turn
green once the corresponding production fix lands.

Baseline before this change: `224 passed` (full `tests/` suite).
After this change: `229 passed, 28 failed` (full `tests/` suite, 30s
per-test timeout). The 28 failures are all newly added and intentional
(RED). No previously passing test was modified in a way that changes its
assertions; no production code under `src/` was touched.

## 1. Orchestrator must pass curated roots to both collectors

File: `tests/indexing/test_orchestrator.py`

- `test_ensure_index_passes_curated_default_roots_to_fallback_indexer` — RED.
  Asserts the fallback indexer is called with an explicit `roots=` kwarg
  equal to `ctags_runner.default_index_roots()`. Currently
  `IndexOrchestrator._reindex` calls
  `self._fallback_indexer(self._kernel_repo)` with no `roots` at all, so the
  fallback indexer's own default (`"."`, i.e. the whole repo) is used
  instead of the curated list.
- `test_ensure_index_passes_curated_default_roots_to_ctags_runner` — RED.
  Same assertion for `ctags_runner`: currently
  `self._ctags_runner(self._kernel_repo)` also omits `roots`, relying on
  `run_ctags`'s own internal default instead of an explicit, orchestrator-
  owned value.
- `test_ensure_index_gives_ctags_and_fallback_the_same_roots` — RED.
  Asserts both collectors receive *identical* `roots` tuples, proving
  coherence between the two rather than each guessing its own default.

## 2. Fallback default must be curated/bounded and cover `.h` too

File: `tests/indexing/test_fallback_indexer.py`

- `test_fallback_scans_header_files_for_definitions_under_configured_roots`
  — RED. `index_fallback(repo, roots=["net"])` with a `net/proto.h`
  containing a `static inline` function definition. Currently
  `_iter_source_files` only globs `*.c`, so `.h` definitions are invisible
  to the fallback scanner no matter what roots are configured.
- `test_fallback_default_roots_exclude_unrelated_directories` — RED.
  Calling `index_fallback(repo)` with **no** `roots` argument must not
  touch an unrelated `fs/` tree. Currently `index_fallback`'s default is
  `["."]` (whole-repo scan), so unrelated trees are always included when
  callers omit `roots`.
- `test_fallback_default_roots_include_curated_single_file_header_definitions`
  — RED. The curated root list (`DEFAULT_INDEX_ROOTS` in `ctags_runner.py`)
  names specific header *files* such as `include/linux/skbuff.h`, not just
  directories. With no `roots` argument, the fallback scanner must resolve
  those file roots and extract their `.h` definitions — currently fails for
  both reasons above (wrong default *and* no `.h` support).

## 3. `run_ctags` must degrade safely and use coherent, complete ctags flags

File: `tests/indexing/test_ctags_runner.py`

- `test_run_ctags_skips_indexing_subprocess_when_no_configured_roots_exist`
  — RED. When none of the caller's `roots` exist on disk, `run_ctags` must
  not launch the indexing subprocess at all (it would otherwise run
  `ctags -R` with **no path operands**, which recursively scans `cwd` —
  defeating the curated-root restriction entirely). Currently the
  indexing subprocess is always launched once `check_ctags_binary`
  succeeds, regardless of whether `existing_roots` is empty.
- `test_run_ctags_requests_signature_field` — RED. Asserts the ctags
  invocation's `--fields=` argument includes `S` (signature). Currently
  it is hard-coded to `--fields=+n` only, so `CtagsDefinition.signature`
  (consumed by `ctags_parser`/`_merge_collectors`) is silently `None` for
  every real ctags run.
- `test_run_ctags_uses_the_same_caller_supplied_timeout_for_probe_and_index`
  — RED. Asserts both the `--version` probe and the indexing subprocess
  receive the same caller-supplied `timeout=` value. Currently
  `run_ctags` calls `check_ctags_binary(executable)` with no timeout
  override, so the probe always uses the hard-coded
  `CTAGS_VERSION_TIMEOUT_SECONDS` (5.0s) regardless of the caller's
  requested bound, breaking a caller's overall latency budget.

## 4. FTS symbol search must never leak `sqlite3.OperationalError`

File: `tests/indexing/test_storage.py`

- `test_search_symbols_never_raises_for_malformed_query_text` (parametrized,
  16 of 17 cases RED) — quotes, semicolons, boolean keywords (`AND`/`OR`/
  `NOT`), bare `*`, unbalanced parentheses, and column-filter syntax
  (`tcp:input`, `-tcp`) all currently raise `sqlite3.OperationalError` from
  `IndexStorage.search_symbols`, verified directly against current `HEAD`.
  The `^tcp` case does not currently raise and is included as a same-shape
  regression guard (green today, must stay green).
- `test_search_symbols_returns_empty_list_for_empty_query` — RED. Empty
  input currently raises `fts5: syntax error near ""`.
- `test_search_symbols_treats_reserved_boolean_keyword_as_literal_text` —
  RED. States exact semantics: searching `"AND"` against a seeded symbol
  named `and_then_helper` must return exactly `["and_then_helper"]`
  (literal-text match), not raise and not silently return nothing.
- `test_search_symbols_treats_trailing_semicolon_as_literal_punctuation` —
  RED. States exact semantics: `"tcp_retransmit_skb;"` must return exactly
  `["tcp_retransmit_skb"]`.
- `test_search_symbols_preserves_prefix_search_with_trailing_wildcard` —
  **green today**, kept as an explicit regression lock. States exact
  semantics: `"retr*"` must return exactly `["tcp_retransmit_skb"]`, so a
  future sanitization fix cannot break the UI's intentional prefix/
  autocomplete search while it fixes the malformed-input cases above.
- `test_search_symbols_preserves_exact_name_search` — **green today**,
  same purpose: exact well-formed name search must still return exactly
  `["tcp_retransmit_skb"]`.

## 5. Fresh schema indexes for symbol name / edge endpoints

File: `tests/indexing/test_storage.py`

- `test_fresh_schema_indexes_symbol_name_column` — **green today**.
  `idx_symbols_name` already exists in `_SCHEMA_STATEMENTS`
  (`src/netstack_academy/indexing/storage.py`), so this assertion already
  holds. No test previously asserted it explicitly, so it is added now as
  a regression guard per the finding's "if not already covered" condition
  — the *test* was not covered, even though the *schema* already was.
- `test_fresh_schema_indexes_edge_source_and_target_columns` — **green
  today**, same rationale for `idx_edges_source`/`idx_edges_target`.

These two are not RED: inspection of `_SCHEMA_STATEMENTS` confirms all
three indexes (`idx_symbols_name`, `idx_edges_source`, `idx_edges_target`)
already exist on a fresh database. They are included per the finding's
explicit request, documented here as already-satisfied with a new
regression-guarding assertion rather than a gap.

## Summary

| # | Finding | New tests | RED | Green (guard) |
|---|---|---|---|---|
| 1 | Orchestrator → curated roots to both collectors | 3 | 3 | 0 |
| 2 | Fallback default curated + `.h` support | 3 | 3 | 0 |
| 3 | `run_ctags` no-roots / `+S` / timeout coherence | 3 | 3 | 0 |
| 4 | FTS search never raises; exact semantics | 22 (17 parametrized + 5) | 19 | 3 (`^tcp` + 2 preserve tests) |
| 5 | Fresh schema indexes | 2 | 0 | 2 |
| **Total** | | **33** (net new test items) | **28** | **5** |

Full-suite run after this change: `229 passed, 28 failed` (was
`224 passed` before). No file under `src/` was modified.

"""Optional semantic (clangd) symbol-index provider.

Everything in this package is best-effort and additive: the kernel network
stack indexer works fine (via ``ctags``/the regex fallback indexer) without
a semantic provider at all. When available, a semantic provider supplies
higher-confidence call-graph edges (``provenance="semantic"``) that the
orchestrator can merge alongside the heuristic ones.
"""

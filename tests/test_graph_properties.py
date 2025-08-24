from __future__ import annotations

from collections.abc import Hashable

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.strategies import DrawFn

from psd.graph import Graph, _reconstruct_path, find_optimal_path


@st.composite
def dag_graphs(draw: DrawFn) -> tuple[Graph, str, str]:
    n = draw(st.integers(min_value=2, max_value=5))
    nodes = [str(i) for i in range(n)]
    graph: Graph = {node: {} for node in nodes}
    for i in range(n - 1):
        for j in range(i + 1, n):
            if draw(st.booleans()):
                weight = draw(st.floats(min_value=0.1, max_value=10.0))
                graph[nodes[i]][nodes[j]] = weight
    if nodes[-1] not in graph[nodes[0]]:
        graph[nodes[0]][nodes[-1]] = draw(st.floats(min_value=0.1, max_value=10.0))
    return graph, nodes[0], nodes[-1]


def brute_force(graph: Graph, start: Hashable, end: Hashable) -> list[Hashable]:
    best: list[Hashable] | None = None
    best_weight = float("inf")

    def dfs(node: Hashable, weight: float, path: list[Hashable]) -> None:
        nonlocal best, best_weight
        if node == end:
            if weight < best_weight:
                best = path.copy()
                best_weight = weight
            return
        for neigh, w in graph.get(node, {}).items():
            dfs(neigh, weight + w, path + [neigh])

    dfs(start, 0.0, [start])
    assert best is not None
    return best


@settings(max_examples=50, deadline=None, derandomize=True)
@given(dag_graphs())
def test_find_optimal_path_matches_bruteforce(data: tuple[Graph, str, str]) -> None:
    graph, start, end = data
    path = find_optimal_path(graph, start, end)
    assert path == brute_force(graph, start, end)


def test_reconstruct_path_cycle_detected() -> None:
    previous = {"B": "C", "C": "B"}
    with pytest.raises(ValueError):
        _reconstruct_path(previous, "A", "C")

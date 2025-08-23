import sys
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from psd.graph import _reconstruct_path, find_optimal_path  # noqa: E402


def _bruteforce_weight(graph: dict[str, dict[str, float]], start: str, end: str) -> float:
    """Compute the minimal path weight via exhaustive search."""
    if start == end:
        return 0.0
    weights: list[float] = []
    for neighbour, w in graph.get(start, {}).items():
        weights.append(w + _bruteforce_weight(graph, neighbour, end))
    return min(weights)


@st.composite
def dag_strategy(draw: st.DrawFn) -> tuple[dict[str, dict[str, float]], str, str]:
    n = draw(st.integers(min_value=2, max_value=6))
    nodes = [str(i) for i in range(n)]
    graph = {node: {} for node in nodes}
    # Ensure a base path 0->1->...->n-1
    for i in range(n - 1):
        graph[nodes[i]][nodes[i + 1]] = draw(
            st.floats(min_value=0.1, max_value=10, allow_infinity=False, allow_nan=False)
        )
    for i in range(n - 1):
        for j in range(i + 2, n):
            if draw(st.booleans()):
                graph[nodes[i]][nodes[j]] = draw(
                    st.floats(min_value=0.1, max_value=10, allow_infinity=False, allow_nan=False)
                )
    return graph, nodes[0], nodes[-1]


@given(dag_strategy())
@settings(max_examples=50)
def test_find_optimal_path_matches_bruteforce(args: tuple[dict[str, dict[str, float]], str, str]) -> None:
    graph, start, end = args
    path = find_optimal_path(graph, start, end)
    # Weight along returned path
    weight = 0.0
    for a, b in zip(path, path[1:]):  # noqa: B905
        weight += graph[a][b]
    assert weight == pytest.approx(_bruteforce_weight(graph, start, end))


def test_reconstruct_path_cycle_raises() -> None:
    prev = {"A": "B", "B": "C", "C": "B"}
    with pytest.raises(ValueError):
        _reconstruct_path(prev, "A", "C")

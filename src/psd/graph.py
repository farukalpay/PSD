"""Graph utilities for computing shortest paths in directed acyclic graphs.

The implementation targets large graphs by using a topological traversal
instead of Dijkstra's algorithm.  The main entry point is
:func:`find_optimal_path` which validates the input graph, executes the search
and reconstructs the shortest path.  Graphs that contain cycles are rejected
early to avoid potentially expensive computations on invalid inputs.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Hashable
from dataclasses import dataclass
from math import isfinite
from time import perf_counter

logger = logging.getLogger(__name__)


Graph = dict[Hashable, dict[Hashable, float]]
"""Type alias for an adjacency list representing a weighted directed graph."""


@dataclass(frozen=True)
class GraphConfig:
    """Configuration options for :func:`find_optimal_path`.

    Attributes
    ----------
    max_path_weight:
        Maximum allowable weight for any path to guard against overflow.
    """

    max_path_weight: float = 1e12

    def __post_init__(self) -> None:  # pragma: no cover - simple validation
        if not isfinite(self.max_path_weight) or self.max_path_weight <= 0:
            raise ValueError("max_path_weight must be positive and finite")


def _validate_graph(graph: Graph, start: Hashable, end: Hashable, config: GraphConfig) -> None:
    """Validate graph structure and ensure non-negative, finite edge weights.

    Parameters
    ----------
    graph:
        The graph to validate.
    start, end:
        Identifiers for the start and end nodes. Both must be present in
        ``graph``.

    Raises
    ------
    ValueError
        If the start or end node is missing, adjacency lists are not
        dictionaries, or if any edge has a negative or excessively large
        weight.
    OverflowError
        If an edge weight exceeds ``config.max_path_weight`` or is not finite.
    """

    if start not in graph or end not in graph:
        raise ValueError("Start or end node not present in graph.")

    for _node, neighbours in graph.items():
        if not isinstance(neighbours, dict):
            raise ValueError("Graph adjacency lists must be dictionaries.")
        for _neighbour, weight in neighbours.items():
            if weight < 0:
                raise ValueError("Graph contains negative edge weights.")
            if not isfinite(weight) or weight > config.max_path_weight:
                raise OverflowError("Edge weight exceeds safe maximum.")


def _initialize_state(graph: Graph, start: Hashable) -> tuple[dict[Hashable, float], dict[Hashable, Hashable]]:
    """Initialise distance estimates and predecessor map."""

    distances: dict[Hashable, float] = {node: float("inf") for node in graph}
    previous: dict[Hashable, Hashable] = {}
    distances[start] = 0.0
    return distances, previous


def _topological_sort(graph: Graph) -> list[Hashable]:
    """Return a topological ordering of ``graph`` or raise ``ValueError``.

    The function performs Kahn's algorithm while ensuring that all nodes are
    included in the ordering.  A ``ValueError`` is raised if the graph contains
    a cycle which would prevent such an ordering.
    """

    indegree: dict[Hashable, int] = {node: 0 for node in graph}
    for _, neighbours in graph.items():
        for neighbour in neighbours:
            indegree.setdefault(neighbour, 0)
            indegree[neighbour] += 1

    queue: deque[Hashable] = deque([n for n, d in indegree.items() if d == 0])
    order: list[Hashable] = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbour in graph.get(node, {}):
            indegree[neighbour] -= 1
            if indegree[neighbour] == 0:
                queue.append(neighbour)

    if len(order) != len(indegree):
        raise ValueError("Graph must be a directed acyclic graph (DAG).")

    return order


def _reconstruct_path(previous: dict[Hashable, Hashable], start: Hashable, end: Hashable) -> list[Hashable]:
    """Rebuild the path from ``start`` to ``end`` using ``previous`` map.

    Prior to this change the function blindly followed the predecessor map.
    In the presence of cycles it could loop indefinitely when the ``previous``
    dictionary contained a cycle, something that happens when graphs with
    cycles slip through validation.  We now track visited nodes while walking
    backwards from ``end`` to ``start`` and raise ``ValueError`` if a cycle is
    encountered.
    """

    path: list[Hashable] = [end]
    visited = set()
    while path[-1] != start:
        node = path[-1]
        if node in visited:
            # Hitting the same node twice indicates a cycle in ``previous``.
            # Returning an explicit error prevents an infinite loop.
            raise ValueError("Cycle detected while reconstructing path.")
        visited.add(node)
        if node not in previous:
            raise ValueError("No path from start to end.")
        path.append(previous[node])
    path.reverse()
    return path


def find_optimal_path(
    graph: Graph,
    start: Hashable,
    end: Hashable,
    *,
    config: GraphConfig | None = None,
) -> list[Hashable]:
    """Find the shortest path from ``start`` to ``end`` in a DAG.

    The graph is represented as an adjacency list mapping each node to a
    dictionary of neighbouring nodes and their corresponding edge weights.  The
    function assumes the graph is a **directed acyclic graph (DAG)** and uses a
    topological ordering to compute the optimal path in linear time relative to
    the number of nodes and edges, making it suitable for large graphs.

    Parameters
    ----------
    graph:
        Adjacency list representation of a weighted directed graph.
    start:
        Starting node identifier.
    end:
        Target node identifier.

    Returns
    -------
    list
        The sequence of nodes representing the shortest path from ``start`` to
        ``end`` (inclusive).

    Raises
    ------
    ValueError
        If the graph contains negative edge weights, cycles, if ``start`` or
        ``end`` is not present in the graph, or if no path exists between the
        two nodes.
    OverflowError
        If the accumulated path weight exceeds ``config.max_path_weight`` or is
        not finite.
    """

    start_time = perf_counter()
    try:
        cfg = GraphConfig() if config is None else config
        _validate_graph(graph, start, end, cfg)
        order = _topological_sort(graph)
        distances, previous = _initialize_state(graph, start)

        for node in order:
            current_dist = distances.get(node, float("inf"))
            if current_dist == float("inf"):
                continue
            for neighbour, weight in graph.get(node, {}).items():
                new_dist = current_dist + weight
                if not isfinite(new_dist) or new_dist > cfg.max_path_weight:
                    raise OverflowError("Path weight exceeds safe maximum.")
                if new_dist < distances.get(neighbour, float("inf")):
                    distances[neighbour] = new_dist
                    previous[neighbour] = node

        if distances.get(end, float("inf")) == float("inf"):
            raise ValueError(f"No path from {start!r} to {end!r}.")

        return _reconstruct_path(previous, start, end)
    finally:
        duration = perf_counter() - start_time
        logger.info("find_optimal_path executed in %.6f seconds", duration)


__all__ = ["find_optimal_path", "GraphConfig"]

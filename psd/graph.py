"""Graph utilities for computing shortest paths in directed acyclic graphs.

The implementation targets large graphs by using a topological traversal
instead of Dijkstra's algorithm.  The main entry point is
:func:`find_optimal_path` which validates the input graph, executes the search
and reconstructs the shortest path.  Graphs that contain cycles are rejected
early to avoid potentially expensive computations on invalid inputs.
"""

from __future__ import annotations

from collections import deque
from math import isfinite
from time import perf_counter
import logging
from typing import Any, Dict, List, Tuple


logger = logging.getLogger(__name__)


Graph = Dict[Any, Dict[Any, float]]
"""Type alias for an adjacency list representing a weighted directed graph."""

MAX_PATH_WEIGHT = 1e12
"""Maximum allowable weight for any path to guard against overflow."""


def _validate_graph(graph: Graph, start: Any, end: Any) -> None:
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
        If an edge weight exceeds :data:`MAX_PATH_WEIGHT` or is not finite.
    """

    if start not in graph or end not in graph:
        raise ValueError("Start or end node not present in graph.")

    for node, neighbours in graph.items():
        if not isinstance(neighbours, dict):
            raise ValueError("Graph adjacency lists must be dictionaries.")
        for neighbour, weight in neighbours.items():
            if weight < 0:
                raise ValueError("Graph contains negative edge weights.")
            if not isfinite(weight) or weight > MAX_PATH_WEIGHT:
                raise OverflowError("Edge weight exceeds safe maximum.")


def _initialize_state(graph: Graph, start: Any) -> Tuple[Dict[Any, float], Dict[Any, Any]]:
    """Initialise distance estimates and predecessor map."""

    distances: Dict[Any, float] = {node: float("inf") for node in graph}
    previous: Dict[Any, Any] = {}
    distances[start] = 0.0
    return distances, previous


def _topological_sort(graph: Graph) -> List[Any]:
    """Return a topological ordering of ``graph`` or raise ``ValueError``.

    The function performs Kahn's algorithm while ensuring that all nodes are
    included in the ordering.  A ``ValueError`` is raised if the graph contains
    a cycle which would prevent such an ordering.
    """

    indegree: Dict[Any, int] = {node: 0 for node in graph}
    for node, neighbours in graph.items():
        for neighbour in neighbours:
            indegree.setdefault(neighbour, 0)
            indegree[neighbour] += 1

    queue: deque[Any] = deque([n for n, d in indegree.items() if d == 0])
    order: List[Any] = []
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


def _reconstruct_path(previous: Dict[Any, Any], start: Any, end: Any) -> List[Any]:
    """Rebuild the path from ``start`` to ``end`` using ``previous`` map."""

    path: List[Any] = [end]
    while path[-1] != start:
        path.append(previous[path[-1]])
    path.reverse()
    return path


def find_optimal_path(graph: Graph, start: Any, end: Any) -> List[Any]:
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
        If the accumulated path weight exceeds :data:`MAX_PATH_WEIGHT` or is not
        finite.
    """

    start_time = perf_counter()
    try:
        _validate_graph(graph, start, end)
        order = _topological_sort(graph)
        distances, previous = _initialize_state(graph, start)

        for node in order:
            current_dist = distances.get(node, float("inf"))
            if current_dist == float("inf"):
                continue
            for neighbour, weight in graph.get(node, {}).items():
                new_dist = current_dist + weight
                if not isfinite(new_dist) or new_dist > MAX_PATH_WEIGHT:
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


__all__ = ["find_optimal_path"]


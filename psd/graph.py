"""Graph utilities including Dijkstra's shortest path algorithm.

This module provides helper functions that collectively implement Dijkstra's
algorithm. The main entry point is :func:`find_optimal_path` which validates the
input graph, executes the search and reconstructs the shortest path.
"""

from __future__ import annotations

from heapq import heappop, heappush
from time import perf_counter
import logging
from typing import Any, Dict, List, Tuple, Set


logger = logging.getLogger(__name__)


Graph = Dict[Any, Dict[Any, float]]
"""Type alias for an adjacency list representing a weighted directed graph."""

PriorityQueue = List[Tuple[float, Any]]
"""Type alias for the priority queue used by Dijkstra's algorithm."""


def _validate_graph(graph: Graph, start: Any, end: Any) -> None:
    """Validate graph structure and ensure non-negative edge weights.

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
        dictionaries, or if any edge has a negative weight.
    """

    if start not in graph or end not in graph:
        raise ValueError("Start or end node not present in graph.")

    for node, neighbours in graph.items():
        if not isinstance(neighbours, dict):
            raise ValueError("Graph adjacency lists must be dictionaries.")
        for neighbour, weight in neighbours.items():
            if weight < 0:
                raise ValueError("Graph contains negative edge weights.")


def _initialize_state(graph: Graph, start: Any) -> Tuple[Dict[Any, float], Dict[Any, Any], PriorityQueue]:
    """Initialise distance estimates, predecessor map and the priority queue."""

    distances: Dict[Any, float] = {node: float("inf") for node in graph}
    previous: Dict[Any, Any] = {}
    distances[start] = 0.0
    heap: PriorityQueue = [(0.0, start)]
    return distances, previous, heap


def _relax_neighbours(
    graph: Graph,
    node: Any,
    current_dist: float,
    distances: Dict[Any, float],
    previous: Dict[Any, Any],
    heap: PriorityQueue,
) -> None:
    """Relax edges from ``node`` updating ``distances`` and ``previous`` maps."""

    for neighbour, weight in graph.get(node, {}).items():
        new_dist = current_dist + weight
        if new_dist < distances.get(neighbour, float("inf")):
            distances[neighbour] = new_dist
            previous[neighbour] = node
            heappush(heap, (new_dist, neighbour))


def _reconstruct_path(previous: Dict[Any, Any], start: Any, end: Any) -> List[Any]:
    """Rebuild the path from ``start`` to ``end`` using ``previous`` map."""

    path: List[Any] = [end]
    while path[-1] != start:
        path.append(previous[path[-1]])
    path.reverse()
    return path


def find_optimal_path(graph: Graph, start: Any, end: Any) -> List[Any]:
    """Find the shortest path from ``start`` to ``end`` using Dijkstra's algorithm.

    The graph is represented as an adjacency list mapping each node to a
    dictionary of neighbouring nodes and their corresponding edge weights.

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
        If the graph contains negative edge weights, if ``start`` or ``end`` is
        not present in the graph, or if no path exists between the two nodes.
    """

    start_time = perf_counter()
    try:
        _validate_graph(graph, start, end)
        distances, previous, heap = _initialize_state(graph, start)
        visited: Set[Any] = set()

        while heap:
            current_dist, node = heappop(heap)
            if node in visited:
                continue
            visited.add(node)
            if node == end:
                break
            _relax_neighbours(graph, node, current_dist, distances, previous, heap)

        if distances.get(end, float("inf")) == float("inf"):
            raise ValueError(f"No path from {start!r} to {end!r}.")

        return _reconstruct_path(previous, start, end)
    finally:
        duration = perf_counter() - start_time
        logger.info("find_optimal_path executed in %.6f seconds", duration)


__all__ = ["find_optimal_path"]


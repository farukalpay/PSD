import sys
import unittest
from pathlib import Path

# Ensure the src directory is on the path for imports
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from psd import GraphConfig, find_optimal_path  # noqa: E402


class TestFindOptimalPath(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = {
            "A": {"B": 1, "C": 4},
            "B": {"C": 2, "D": 5},
            "C": {"D": 1},
            "D": {},
        }

    def test_shortest_path(self) -> None:
        path = find_optimal_path(self.graph, "A", "D")
        self.assertEqual(path, ["A", "B", "C", "D"])

    def test_negative_weight_raises(self) -> None:
        graph = {"A": {"B": -1}, "B": {}}
        with self.assertRaises(ValueError):
            find_optimal_path(graph, "A", "B")

    def test_disconnected_nodes_raise(self) -> None:
        graph = {"A": {"B": 1}, "B": {}, "C": {}}
        with self.assertRaises(ValueError):
            find_optimal_path(graph, "A", "C")

    def test_missing_node_raises(self) -> None:
        with self.assertRaises(ValueError):
            find_optimal_path(self.graph, "A", "Z")

    def test_cycle_graph_raises(self) -> None:
        graph = {
            "A": {"B": 1},
            "B": {"C": 2},
            "C": {"A": 3},
        }
        with self.assertRaises(ValueError):
            find_optimal_path(graph, "A", "C")

    def test_large_path_weight_raises(self) -> None:
        graph = {
            "A": {"B": 6e11},
            "B": {"C": 6e11},
            "C": {},
        }
        with self.assertRaises(OverflowError):
            find_optimal_path(graph, "A", "C")

    def test_custom_config_overrides_max_weight(self) -> None:
        graph = {"A": {"B": 1e6}, "B": {"C": 1e6}, "C": {}}
        cfg = GraphConfig(max_path_weight=1e6)
        with self.assertRaises(OverflowError):
            find_optimal_path(graph, "A", "C", config=cfg)

    def test_graph_config_validation(self) -> None:
        with self.assertRaises(ValueError):
            GraphConfig(max_path_weight=-1.0)

    def test_large_graph(self) -> None:
        # Construct a linear DAG A0 -> A1 -> ... -> A999 -> A1000
        graph = {str(i): {str(i + 1): 1} for i in range(1000)}
        graph[str(1000)] = {}
        path = find_optimal_path(graph, "0", "1000")
        self.assertEqual(len(path), 1001)
        self.assertEqual(path[0], "0")
        self.assertEqual(path[-1], "1000")

    def test_logs_execution_time(self) -> None:
        with self.assertLogs("psd.graph", level="INFO") as cm:
            find_optimal_path(self.graph, "A", "D")
        self.assertTrue(
            any("find_optimal_path executed in" in message for message in cm.output),
            msg="Expected log message with execution time",
        )


if __name__ == "__main__":
    unittest.main()

import unittest
import sys
from pathlib import Path

# Ensure the package root is on the path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from psd import find_optimal_path


class TestFindOptimalPath(unittest.TestCase):
    def setUp(self):
        self.graph = {
            'A': {'B': 1, 'C': 4},
            'B': {'C': 2, 'D': 5},
            'C': {'D': 1},
            'D': {}
        }

    def test_shortest_path(self):
        path = find_optimal_path(self.graph, 'A', 'D')
        self.assertEqual(path, ['A', 'B', 'C', 'D'])

    def test_negative_weight_raises(self):
        graph = {'A': {'B': -1}, 'B': {}}
        with self.assertRaises(ValueError):
            find_optimal_path(graph, 'A', 'B')

    def test_disconnected_nodes_raise(self):
        graph = {'A': {'B': 1}, 'B': {}, 'C': {}}
        with self.assertRaises(ValueError):
            find_optimal_path(graph, 'A', 'C')

    def test_missing_node_raises(self):
        with self.assertRaises(ValueError):
            find_optimal_path(self.graph, 'A', 'Z')

    def test_logs_execution_time(self):
        with self.assertLogs('psd.graph', level='INFO') as cm:
            find_optimal_path(self.graph, 'A', 'D')
        self.assertTrue(
            any('find_optimal_path executed in' in message for message in cm.output),
            msg="Expected log message with execution time",
        )


if __name__ == '__main__':
    unittest.main()

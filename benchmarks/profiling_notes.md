# Profiling Notes

## Setup
- Function: `rosenbrock_hess` with input dimension 1000.
- Profiling tools: `cProfile` for CPU time, `tracemalloc` for memory.

## Results
| Version | Mean time (ms) | Peak memory (MB) |
|---------|----------------|-----------------|
| Before  | 3.52           | 8.00            |
| After   | 0.67           | 8.04            |

The profiler highlighted repeated Python loops in the original implementation as the primary cost. Replacing these loops with NumPy vectorized operations yielded about a 5x speedup while keeping memory usage roughly constant.

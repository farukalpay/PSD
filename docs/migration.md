# Import Path Migration

The project now adopts a `src/` layout. Library modules live under the
`psd` and `psd_optimizer` packages instead of being available as
implicitly importable top-level modules.

Update any imports accordingly. For example:

```python
- import algorithms
- from optimizer import PSDOptimizer
+ from psd import algorithms
+ from psd_optimizer import PSDOptimizer
```

Ensure the package is installed (e.g. `pip install -e ".[dev]"`) so
that these imports resolve correctly.


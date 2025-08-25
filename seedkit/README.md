# seedkit

`seedkit` provides deterministic sub-seeding utilities for scientific experiments.  A master seed
combined with textual identifiers is hashed using SHA-256 to derive a 64‑bit sub-seed.  This
sub-seed feeds NumPy's counter-based `Philox` bit generator which enables many independent and
reproducible streams.

Hashing decorrelates streams derived from related identifiers, but it does **not** offer any
cryptographic guarantees.  It simply prevents accidental global correlations when deriving
sub-seeds from structured information.

## Usage

### Library
```python
from seedkit.seeding import make_subseed, philox_generator, python_random

sub = make_subseed(42, "trainer", "R1", 0)
np_gen = philox_generator(sub)
py_gen = python_random(sub)
print(np_gen.random(3))
print(py_gen.random())
```

### Command line
```
$ seedkit demo --master 42 --component trainer --run R1 --stream 0 --n 5
Derived subseed: 11789288199025255931
NumPy samples: [0.0116 ...]
Python samples: [0.2327 ...]
Torch samples: [0.4561 ...]  # only if PyTorch is installed
```

Philox is a counter-based generator designed for parallel use.  Each stream consumes from an
independent counter, avoiding overlap between streams.

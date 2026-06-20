# pythonic-algorithms-lab — Suggested Fixes

---

## 1. 🔴 Lint error (blocking CI)

**File:** `benchmarks/plot_big_o.py`

Run this and commit — ruff auto-removes the unused import:

```bash
ruff check . --fix
git add benchmarks/plot_big_o.py
git commit -m "fix: remove unused numpy import (ruff F401)"
git push
```

Or apply manually — remove line 15:

```diff
 import matplotlib.pyplot as plt
 import matplotlib.patches as mpatches
-import numpy as np
 from pathlib import Path
```

---

## 2. 🟡 Node.js 20 deprecation in CI

**File:** `.github/workflows/ci.yml`

GitHub Actions is deprecating Node.js 20 — enforcement started June 2026.
Add `FORCE_JAVASCRIPT_ACTIONS_TO_NODE24: true` to the top-level `env:` block:

```diff
 env:
   PYTHON_VERSION: '3.14'
+  FORCE_JAVASCRIPT_ACTIONS_TO_NODE24: true
```

---

## 3. 🟡 Missing `__init__.py` in `algorithms/cpu/sorting/`

Every other CPU sub-package has an `__init__.py` — sorting is the only exception.
Create an empty one for consistency:

```bash
touch algorithms/cpu/sorting/__init__.py
git add algorithms/cpu/sorting/__init__.py
git commit -m "fix: add missing __init__.py to algorithms/cpu/sorting"
```

---

## 4. 🟡 Empty stub test file

**File:** `tests/test_radix_cupy.py`

Currently contains only a docstring with zero test functions. The `gpu-smoke` CI job
lists it explicitly, so it runs vacuously and gives false confidence.

Replace the stub with a proper redirect comment and a skip marker so it is honest
about what it does:

```python
"""
Radix CuPy tests live in tests/test_gpu_kernels.py::test_radix_cupy_correctness.
This file is intentionally empty — kept so the gpu-smoke job path remains valid.
"""
```

Or simply remove it from the `gpu-smoke` job step and delete the file:

```diff
 - name: Run GPU smoke tests
-  run: python -m pytest tests/test_gpu_kernels.py tests/test_numba_spmv.py tests/test_radix_cupy.py -v
+  run: python -m pytest tests/test_gpu_kernels.py tests/test_numba_spmv.py -v
```

---

## 5. 🟡 Pin loose dependencies

**File:** `requirements.txt`

`plotly` and `dash` have no minimum version, which can cause unexpected breakage on
a fresh install. Add lower bounds:

```diff
-plotly
-dash
+plotly>=5.0
+dash>=2.0
```

---

## 6. 🟡 Add test coverage reporting to CI

**File:** `.github/workflows/ci.yml`

Add `pytest-cov` to the install step and pass `--cov` to the test run:

```diff
     - name: Install dependencies
       run: |
         python -m pip install --upgrade pip
-        pip install -r requirements.txt
+        pip install -r requirements.txt pytest-cov

     - name: Run tests
-      run: python -m pytest -q
+      run: python -m pytest -q --cov=algorithms --cov-report=term-missing
```

---

## 7. 🔵 Add a `conftest.py` with shared fixtures

**New file:** `tests/conftest.py`

Eliminates repeated random-list setup across test files:

```python
import random
import pytest


@pytest.fixture
def random_ints():
    """Return a seeded list of 50 random integers in [-100, 100]."""
    random.seed(42)
    return [random.randint(-100, 100) for _ in range(50)]


@pytest.fixture
def sorted_ints():
    return list(range(50))


@pytest.fixture
def reversed_ints():
    return list(range(49, -1, -1))
```

---

## 8. 🔵 Add docstring to quick sort about recursion limit

**File:** `algorithms/cpu/sorting/quick_sort.py`

```diff
 def quick_sort(arr):
+    """Sort a list using recursive quicksort with middle-element pivot.
+
+    The middle-element pivot avoids O(n²) behaviour on already-sorted input.
+    Note: Python's default recursion limit (~1000) caps practical input size
+    to roughly 5 000–10 000 elements before hitting RecursionError.
+    For larger inputs, use tim_sort or merge_sort instead.
+    """
     a = list(arr)
```

---

## Summary checklist

| # | File(s) | Action | Priority |
|---|---------|--------|----------|
| 1 | `benchmarks/plot_big_o.py` | Remove unused `numpy` import | 🔴 Now |
| 2 | `.github/workflows/ci.yml` | Add `FORCE_JAVASCRIPT_ACTIONS_TO_NODE24` | 🟡 Soon |
| 3 | `algorithms/cpu/sorting/__init__.py` | Create empty file | 🟡 Soon |
| 4 | `tests/test_radix_cupy.py` + `ci.yml` | Remove stub or clean up CI reference | 🟡 Soon |
| 5 | `requirements.txt` | Pin `plotly>=5.0` and `dash>=2.0` | 🟡 Soon |
| 6 | `.github/workflows/ci.yml` + `requirements.txt` | Add `pytest-cov` | 🟡 Soon |
| 7 | `tests/conftest.py` | Create shared fixtures | 🔵 Later |
| 8 | `algorithms/cpu/sorting/quick_sort.py` | Add recursion limit docstring | 🔵 Later |

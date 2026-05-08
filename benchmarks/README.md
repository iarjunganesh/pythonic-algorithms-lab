# Benchmarks

This folder contains runners, smoke-tests, and helpers to generate reproducible benchmark CSVs and to explore results interactively.

Main scripts

- `run_benchmarks.py`: runner supporting per-group runs (`--group sort` — only group currently), `--full` for full sweeps, `--merge-inputs` for merging CSV outputs, `--include-memory` to capture memory metrics, `--plot` to generate a quick matplotlib chart, and environment-aware GPU registration.
- `dashboard_app.py`: Dash app to explore CSV results interactively.
- `gpu_smoke.py`: quick GPU/toolchain validator and smoke tester (safe CPU fallbacks included).

Recommended workflow

1. Run the `sort` group benchmark and save a CSV:

```powershell
python benchmarks/run_benchmarks.py --group sort --sizes 1000 10000 --repeat 3 --out benchmarks/results_sort.csv
```

2. Run a fast smoke sweep (validates runner and GPU detection):

```powershell
python benchmarks/run_benchmarks.py --full --sizes 10 100 --repeat 2 --out benchmarks/results_smoke.csv
```

3. (Optional) Run the canonical sweep and persist results:

```powershell
python benchmarks/run_benchmarks.py --full --sizes 100 1000 5000 10000 50000 100000 --repeat 5 --out benchmarks/results_full.csv
```

4. Merge CSVs into a single file for analysis (default merge output is `results_merged.csv`):

```powershell
python benchmarks/run_benchmarks.py --merge-inputs "benchmarks/results_*.csv" --merge-out benchmarks/results_merged.csv
```

5. Launch the interactive dashboard to explore:

```powershell
python benchmarks/dashboard_app.py --csv benchmarks/results_merged.csv
```

Files you may use

- `benchmarks/results_full.csv` — canonical results file produced by the full sweep; committed to the repo as the authoritative CSV for the dashboard.
- `benchmarks/results_smoke.csv` — produced by the smoke sweep command (step 2 above); not committed.
- `benchmarks/results_merged.csv` — produced by `--merge-inputs`; the runner adds a `_source` column recording the originating filename.

GPU and environment

- The runner conditionally registers GPU-backed kernels when a compatible GPU toolchain is detected (CuPy, Numba). Use `benchmarks/gpu_smoke.py` to verify GPU availability before heavy GPU runs.
- Install a CuPy wheel matching your CUDA runtime (example for CUDA 13.x):

```powershell
pip install cupy-cuda13x
```

Notes and best practices

- The runner uses deterministic synthetic inputs for many wrappers to ensure reproducible comparisons across runs and machines.
- If a given kernel fails during a run it is skipped and the error is printed; this keeps full sweeps robust across heterogeneous environments.
- For environments where Numba/CUDA compatibility is uncertain, prefer `conda`/`mamba` and `conda-forge` packages for `numba` and `cudatoolkit`.

The `benchmarks/` directory is the canonical place for runner documentation.

# Next Iteration Scaffold (CPU vs GPU Algorithms)

Use this document as the operating plan for each iteration.

## 1) Iteration Thesis

Define one measurable thesis for the cycle.

Template:
- Thesis: "For algorithm family X, GPU outperforms CPU beyond input size N because of Y."
- Scope: [algorithm families in scope]
- Out of scope: [what is explicitly excluded]
- End date: [YYYY-MM-DD]

## 2) Value Criteria (Why this iteration matters)

An iteration is valid only if it produces at least one:
- New performance insight (not previously documented)
- New optimization pattern (memory, occupancy, launch config, data layout)
- New crossover boundary (CPU->GPU switch point)
- Reusable benchmark harness improvement

If none of the above is expected, do not start the iteration.

## 3) Candidate Algorithm Intake Rubric

Score each candidate 0-2 on each dimension.

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| Pattern novelty | Already covered | Slightly different | New access/sync/sparsity pattern |
| Benchmark relevance | Toy only | Moderate practical value | Strong real-world relevance |
| GPU fit | CPU-friendly only | Mixed | Strong GPU potential |
| Measurability | Hard to compare | Partial metrics | Clear CPU/GPU metrics |
| Implementation risk | High unknowns | Medium | Low/contained |

Selection rule:
- Minimum total score: 7
- At least one selected algorithm per class:
  - Compute-bound
  - Memory-bound
  - Irregular/graph
  - Sparse

## 4) Definition of Done (Per Algorithm)

A single algorithm is complete only when all checks pass.

Functional:
- CPU and GPU implementations produce equivalent outputs (within tolerance)
- Unit tests added and passing

Performance:
- Benchmarks executed for all required input sizes
- Mean, p50, p95, stdev reported
- CPU and GPU crossover point identified

Profiling:
- Nsight Systems capture saved
- Nsight Compute capture saved
- Bottleneck classification documented: compute-bound, memory-bound, or sync-bound

Documentation:
- One short findings note added in iteration report
- Key optimization decisions recorded

## 5) Benchmark Protocol (Must remain fixed during iteration)

Environment:
- Python version:
- OS:
- CPU model:
- GPU model:
- Driver/CUDA versions:

Execution settings:
- Warmup runs: 5
- Measured runs: 20
- Random seed: fixed
- Input generation: deterministic
- CPU affinity: fixed (if possible)
- Background load: minimal

Metrics to capture:
- Runtime (ms)
- Throughput (ops/sec or elements/sec)
- VRAM peak (MB)
- Memory bandwidth estimate (if available)
- Accuracy delta (if approximate algorithm)

Reporting format:
- CSV in benchmarks/results/
- Aggregated summary table
- Plots: runtime vs size, speedup vs size, VRAM vs size

## 6) Optimization Ladder Template

For each algorithm, implement and measure step-by-step.

1. Baseline implementation
2. Data layout or memory access improvement
3. Shared memory or tiling optimization
4. Launch configuration tuning
5. Final tuned version

For each step, record:
- Change made
- Why it should help
- Measured delta vs previous step
- Any correctness impact

## 7) Weekly Plan (4-Week Sprint)

Week 1: Benchmark and harness hardening
- Freeze protocol and metrics
- Validate deterministic runs
- Add missing test fixtures

Week 2: First optimization wave (4 algorithms)
- Implement ladders and profiling captures
- Publish interim crossover plots

Week 3: Second optimization wave (remaining algorithms)
- Complete ladders and profiling captures
- Consolidate speedup findings

Week 4: Synthesis and release
- Final comparison tables and plots
- Decision guide: when to use CPU vs GPU
- Tag iteration release

## 8) Artifacts Checklist (Release Gate)

All required artifacts before closing the iteration:
- Updated benchmark CSVs
- Plots generated and saved
- Profiling captures attached
- Iteration report completed
- Tests green in CI/local
- README release notes updated

## 9) Iteration Report Template

Copy this block into the report file for each cycle.

### Iteration [ID]
- Date range:
- Thesis:
- Algorithms covered:

### Results Summary
- Best GPU speedup:
- Median speedup:
- Crossover boundaries:
- Main bottleneck categories:

### What worked
- [bullet]

### What failed or regressed
- [bullet]

### Reusable lessons
- [bullet]

### Next iteration candidates
- [bullet]

## 10) Suggested Next Iteration (Starting Point)

Use this exact scope for the next cycle to keep momentum.

Algorithms:
- Compute-bound: matrix multiply (tiled), FFT variant
- Memory-bound: reduction, prefix scan
- Irregular/graph: BFS frontier expansion
- Sparse: CSR SpMV tuning

Hypothesis:
- Memory-bound kernels gain primarily from coalesced access and shared-memory reduction patterns.

Deliverables:
- 8 algorithm ladders completed
- 1 consolidated crossover chart pack
- 1 decision guide for CPU vs GPU selection

---

Owner:
- Name:
- Iteration ID:
- Start date:
- Target release date:

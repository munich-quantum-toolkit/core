# MLIR mapping performance evaluation

The `mqt-core-mlir-mapping-eval` executable times a pass manager containing only
the QCO mapping pass. Program construction, MLIR context initialization, and
module verification happen outside the timed interval. The generated circuits
and mapper seed are deterministic. The default mapper configuration matches the
PR #1930 evaluation: 20 lookahead steps, lambda 0.5, one refinement iteration,
and 18 initial-layout trials.

Configure and build an optimized executable with:

```console
MLIR_DIR=/path/to/llvm/lib/cmake/mlir cmake --preset release \
  -DBUILD_MQT_CORE_BENCHMARKS=ON
cmake --build build/release --target mqt-core-mlir-mapping-eval
```

To compare executables from two worktrees, run:

```console
python eval/mapping_evaluation.py \
  /path/to/baseline/build/release/eval/mqt-core-mlir-mapping-eval \
  /path/to/candidate/build/release/eval/mqt-core-mlir-mapping-eval \
  --output mapping-results.json
```

The runner randomizes the execution order within each pair. Its JSON output
contains the raw nanosecond samples, source-worktree revisions and dirty states,
executable SHA-256 digests, CMake and compiler metadata, medians, median
absolute deviations, 10% trimmed means, and paired median speedups with a
bootstrap 95% confidence interval. Use the same machine without other
substantial workloads for both executables.

Do not use byte-for-byte mapped IR equality as a cross-process correctness
check. Ready operations are stored in a pointer-keyed `DenseMap`, so address
layout can change their iteration order and select an equally valid alternative
route. Validate mapping correctness with focused tests and evaluate route
quality separately from pass execution time.

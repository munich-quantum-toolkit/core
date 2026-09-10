# Compiler-device API overhead

Baseline: `82dc759a8778d026e5231717e2f0515e217fcdef`, native DGX Spark,
Python 3.14 with Clang 23 `MinSizeRel` (`-Os -DNDEBUG`), IPO disabled.
`build/python/MinSizeRel/CMakeCache.txt` confirms these settings and both
DDSIM and SC devices enabled. Native correctness tests separately use the
release ThinLTO preset; the Python timings do not measure that build.
The JSON files record interpreter,
platform, installed extension path and SHA-256, Git HEAD, and every sample.
The after run measures the implementation in
`a59b1c375ce98840a7a7aa31b519238dcf1a4f86`. The measured production sources
were verified to match that commit before removing the duplicate patch.
Installed extension hashes tie the raw results to the measured binaries.
No generated binaries belong in the PR.

Run from the repository root with the installed package matching each revision:

```sh
UV_PROJECT_ENVIRONMENT=.nox/stubs SKBUILD_CMAKE_BUILD_TYPE=MinSizeRel \
  SKBUILD_CMAKE_ARGS=-DBUILD_MQT_CORE_QDMI_SC_DEVICE=ON \
  CC=clang-23 CXX=clang++-23 CMAKE_BUILD_PARALLEL_LEVEL=8 \
  uv sync --no-dev --group build --group test \
  --no-build-isolation-package mqt-core --reinstall-package mqt-core
.nox/stubs/bin/python .agent/benchmarks/device-compiler-api/bench.py > .agent/benchmarks/device-compiler-api/before.json
# Check out the after revision and repeat the same build command, then:
.nox/stubs/bin/python .agent/benchmarks/device-compiler-api/bench.py > .agent/benchmarks/device-compiler-api/after.json
/usr/bin/clang++-23 -std=c++20 -O3 .agent/benchmarks/device-compiler-api/tuples.cpp -o /tmp/mqt-device-api-tuples
/tmp/mqt-device-api-tuples > .agent/benchmarks/device-compiler-api/tuples.csv
uv run --no-project --with matplotlib python .agent/benchmarks/device-compiler-api/plot.py
```

The Python benchmark uses nine samples per case. Submission timing ends before
waiting for completion, then checks the Bell counts and 16-shot total. The raw
submission case includes Python payload extraction. Compilation is warmed by
creating the compiled artifact first. Chain construction uses unrestricted
operations, no distance queries, and preconstructed connectivity; it isolates
target construction rather than device-provider overhead.

The C++ harness uses seven samples of reversed, unique pairs and includes
allocation and sorting for the replacement. It measures the tuple comparison
algorithm, not the complete compiler. Counting comparisons adds work to both
algorithms. This is a worst-order case: identical tuple order uses the existing
linear fast path and should not be assigned the reported helper speedup.

Baseline medians: DDSIM snapshot 15.70 ms, compiled submission 18.05 ms,
raw submission 0.080 ms, source submission 64.34 ms. Chain construction with
256 / 1,024 / 4,096 sites takes 0.370 / 8.169 / 150.552 ms. Raw samples show
spread; these are local measurements, not portable latency guarantees.

At 8,000 reversed tuples, permutation takes 128,000,001 comparisons and a
60.0 ms median; sorted borrowed tuple views take 217,833 comparisons and a
0.103 ms median. The normal same-order path remains linear.

The lazy distance change moves all-pairs work to the first distance query.
Routing still pays its required all-pairs cost; construction and compatibility
checks avoid it. Connectivity is still checked eagerly with one traversal,
and a focused test verifies distances through concurrent target copies.

After medians (same nine-sample harness):

| Case | Before (ms) | After (ms) | After range (ms) |
| --- | ---: | ---: | ---: |
| DDSIM full snapshot | 15.696 | 15.590 | 13.357–49.826 |
| Precompiled submission | 18.051 | 9.318 | 9.240–31.325 |
| Raw submission | 0.080 | 0.061 | 0.017–0.100 |
| Source submission | 64.336 | 44.540 | 42.296–72.913 |
| Chain construction, 256 sites | 0.370 | 0.016 | 0.016–0.038 |
| Chain construction, 1,024 sites | 8.169 | 0.063 | 0.060–0.071 |
| Chain construction, 4,096 sites | 150.552 | 0.251 | 0.244–0.274 |

The host was shared with unrelated LTO workers, so these are indicative local
measurements, not isolated performance guarantees. `after-contended.json`
retains an earlier run overlapping this task's rebuild and Python tests; its
API medians are worse and its spread wider. The final run started after those
operations ended. No unchanged-case speedup is claimed for raw submission or
full snapshots. The strong chain-construction improvement comes from removing
quadratic work entirely, and provider-query regression tests independently
protect the reduction in submission queries.

Remaining intentional costs: a separately compiled artifact must refresh the
current device legality metadata. DDSIM still has 65,535 sites to enumerate.
Compilation retains complete calibration snapshots. The distance cache still
uses quadratic memory when routing first requests distances. CompilerTarget
copies already share immutable storage; adding another cache or deep-copy
optimization would not help this contract.

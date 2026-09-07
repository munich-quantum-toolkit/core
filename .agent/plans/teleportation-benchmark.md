# Teleportation benchmark

Status: complete.

## Goal and scope

Expose a fixed `teleportation` family through C++, Python, JSON, the CLI, and
structured MLIR generation. Teleport `|+>` with measurement-dependent X and Z
corrections. Return only Bob's X-basis measurement: `0` denotes success.

`src/bench/Teleportation.cpp` owns the reference and success score;
`mlir/bench/programs/Teleportation.cpp` owns preparation and feed-forward. The
family has no options. Alice's measurements are internal scalar conditions.

## Decisions

Apply H to Bob before measurement to project onto the prepared `|+>` state. A
Z-basis measurement cannot distinguish `|+>` from an incorrect `|->` state. The
success probability measures overlap with the fixed input; it does not certify
the teleportation channel for arbitrary inputs. In particular, X stabilizes this
input, so the structural test also protects the X correction.

Use the existing evaluation helper with success outcome `0`, and include that
outcome in the manifest. No new simulation or benchmark abstraction is needed.

## Validation

Run `mqt-core-bench-test --gtest_filter='Teleportation.*:BenchmarkJSON.*'` and
`mqt-core-mlir-unittests-benchmark` from their build directories, then
`uv run --no-sync pytest test/python/test_bench.py -k teleportation`. These
checks cover reference metrics, JSON identity, internal correction data flow,
QC/jeff generation, and deterministic DD sampling of Bob's result.

# Coherent DD simulation performance

Status: complete; hosted CI and Windows performance remain unverified.

## Goal and scope

Recursive DD addition in `include/mqt-core/dd/Package.hpp` now preserves
subgraph sharing while retaining small amplitudes in wide coherent states. The
decomposition test retains its widths, phase checks, and numerical tolerance.
`docs/dd_package.md` describes the scaling rule.

## Decisions

Dividing both operands by their largest component at every recursion level
introduced rounding that could prevent shared subgraphs at tight tolerances.
Leave operands unchanged when their largest component is in `[0.5, 2)`;
otherwise extract a common power-of-two scale and restore it on return. Apply
this rule to ordinary and magnitude addition. Removing scaling entirely would
lose the protection against amplitude loss in wide states.

## Validation

Against baseline `1dd85604a`, the release decomposition binary with
`--gtest_filter='*.CoherentStatesMatchAcrossSynthesisBoundaries'` and
`--gtest_repeat=3` took 8.886, 10.243, and 8.640 seconds before the change, and
2.874, 2.887, and 2.975 seconds after it. Median time fell from 8.886 to 2.887
seconds (3.1 times faster) on macOS 26.6.2 ARM64 with AppleClang 21, MLIR
23.1.0, `-O3 -DNDEBUG`, and interprocedural optimization disabled. These
measurements do not establish the speedup on Windows.

All 213 DD tests and 315 decomposition tests passed. Existing wide-state tests
in `test/dd/test_package.cpp` now also cover incoming scales from `1e-200` to
`1e200`, including phase preservation. The benchmark test
`GenerateProgramTest.SamplesRepeatUntilSuccessAgainstReference` passed, checking
the sampling regression that motivated recursive scaling. Repository lint
passed; the required C++ lint session could not run because clang-tidy 23 was
unavailable.

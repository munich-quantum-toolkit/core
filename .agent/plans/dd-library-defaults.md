# Decision-diagram defaults and numerical indexing

Status: complete.

## Outcome and scope

The native DD library uses the measured numerical, allocation, and cache
improvements without new configuration fields. The owning code is under
`include/mqt-core/dd` and `src/dd`; behavioral tests are in `test/dd` and the
public explanation is in `docs/dd_package.md`. The production diff removes
40 lines overall. Experiment harnesses, raw data, and plots stay outside the
repository.

## Decisions

Preserve absolute tolerance 2^-42. Tighter tolerances help some Grover cases
but destroy sharing in larger twisting cases, which is not a suitable general
default. Retain nearest-value interval indexing, relative dominant-phase
selection, balanced cached-vector projection, and stored-weight compensation.
The real index starts at 65,536 buckets and doubles with occupancy up to
1,048,576. Rehashing preserves entry addresses and collection flags.

Start node unique tables with 64 buckets per level and keep their existing
independent growth. Grow the matrix-vector compute cache automatically after
collection, bounded at 1,048,576 buckets and gated by prior reuse. Its target
is the next power of two above four times the surviving vector-node count,
subject to the ceiling. Other compute caches and GC budgets retain their
existing policy. Earlier growth for every real/node table and larger GC
budgets did not give a general benefit.

Use uninitialized slabs and zero fresh entries on acquisition. Preserve
intrusive recycling, stable addresses, tagged real pointers, reference roots,
and reset/collection semantics. Native DDs remain independent of LLVM.
Invalidate all affected caches before growth can allocate. Failed compute-table
resizing preserves the old storage without exposing reclaimed entries.

## Validation and measured limits

The comparison baseline is refreshed upstream main at
`d661b7759fd84961191291565bf5b92eea5ae0f6`. Both variants use normal defaults,
matching Clang release settings, and identical circuits and independent
references. The primary matrix covers 70 family/size pairs across GHZ, QFT,
coherent Hamming counting, one-axis twisting, phase estimation, fixed-excitation
XXZ, dense states, and full Grover search. Of 372 fresh processes, 362 pass
numerical checks and 10 hit the 180-second wall limit. Separate confirmation
runs pass 34/34, and final-build checks pass 16/16. Host contention limits the
precision of timing comparisons; censored runs have no inferred accuracy result.

The three-pair median candidate/main CPU ratios are 0.72 for GHZ/8192,
0.69 for Hamming/256, 0.33 for twisting/64, and 1.09 for dense/14. GHZ/8192
uses 37 MiB instead of 163 MiB. Hamming/256 spends 53 MiB instead of 33 MiB.
Single-trial Hamming/512 and three-excitation XXZ/64 regress about 10-11%.
These tradeoffs are accepted without adding workload-specific switches.

The candidate completes Grover/26, while its Grover/28 and Grover/30 trials
are censored and reach roughly 2.6 GiB and 2.3 GiB of process RSS. The selected
default does not guarantee compact intermediate DDs, polynomial runtime, or a
bound on accumulated numerical error. Smaller tolerance remains an existing
C++ setting, not a replacement for validating a circuit's numerical behavior.

Native `release-no-mlir` builds and CTest pass on both branches: 586 tests on
the candidate and 580 on main, with the same upstream test skipped. All 202
candidate DD tests pass with AddressSanitizer and UndefinedBehaviorSanitizer.
Whole-file clang-tidy passes on all eight changed C++ translation units and
the changed DD headers. `uvx nox -s lint` passes. MLIR, Python integration,
and executable-documentation builds were not run for this change.

# Decision-diagram defaults and numerical indexing

Status: complete.

## Outcome and scope

The native DD library uses the measured numerical, allocation, and cache
improvements without new configuration fields. The owning code is under
`include/mqt-core/dd` and `src/dd`; behavioral tests are in `test/dd` and the
public explanation is in `docs/dd_package.md`. The production diff removes 52
lines overall. Experiment harnesses, raw data, and plots stay outside the
repository.

## Decisions

Preserve absolute tolerance 2^-42. Tighter tolerances help some Grover cases but
destroy sharing in larger twisting cases, which is not a suitable general
default. Retain nearest-value interval indexing, relative dominant-phase
selection, balanced cached-vector projection, and stored-weight compensation.
The real index starts at 65,536 buckets and doubles with occupancy up to
1,048,576. Rehashing preserves entry addresses and collection flags.

Start node unique tables with 64 buckets per level and keep their existing
independent growth. Grow the matrix-vector compute cache automatically after
collection, bounded at 1,048,576 buckets and gated by prior reuse. Its target is
the next power of two above four times the surviving vector-node count, subject
to the ceiling. Other compute caches and GC budgets retain their existing
policy. Earlier growth for every real/node table and larger GC budgets did not
give a general benefit.

Use uninitialized slabs and zero fresh entries on acquisition. Preserve
intrusive recycling, stable addresses, tagged real pointers, reference roots,
and reset/collection semantics. Reuse an already consolidated slab in
`reset(true)`; replacing equal capacity adds allocation churn and can terminate
a constrained process needlessly. Native DDs remain independent of LLVM.
Invalidate all affected caches before growth can allocate. Failed compute-table
resizing preserves the old storage without exposing reclaimed entries.

## Validation and measured limits

The comparison baseline is refreshed upstream main at
`ee7edb68a691e1d9a440dfd16b605f81f05d5c21`. Both variants use normal defaults,
matching Clang release settings, and identical circuits and independent
references. The primary matrix covers 70 family/size pairs across GHZ, QFT,
coherent Hamming counting, one-axis twisting, phase estimation, fixed-excitation
XXZ, dense states, and full Grover search. Of 372 fresh processes, 363 pass
numerical checks and nine hit the 180-second wall limit. All 24 smoke controls
and both separate Grover/28 confirmations pass. Host contention and
pointer-sensitive cache behavior limit timing precision; censored runs have no
inferred accuracy result.

The three-pair median candidate/main CPU ratios are 0.54 for GHZ/8192, 0.73 for
Hamming/256, 0.40 for twisting/64, and 0.12 for Grover/24. GHZ/8192 uses 37 MiB
instead of 163 MiB. Hamming/256 spends 53 MiB instead of 33 MiB. Single-trial
Hamming/512 is 4% slower and uses 96 MiB instead of 56 MiB. Dense/12 is 9%
slower across three pairs. These tradeoffs do not warrant workload-specific
switches.

The candidate completes Grover/28 in all three fresh trials, with CPU ranging
from 5.06 to 13.53 seconds and peak RSS from 22 to 123 MiB. Its Grover/30 trial
is censored at about 2.8 GiB process RSS. Inexact QPE/22 completes but retains
3.9 million DD nodes and uses 698 MiB before reference validation. The selected
default does not guarantee compact intermediate DDs, polynomial runtime, or a
bound on accumulated numerical error. Smaller tolerance remains an existing C++
setting, not a replacement for validating a circuit's numerical behavior.

Native release builds and CTest pass on both branches: 586 tests on the
candidate and 580 on main, with the same upstream test skipped. All 202
candidate DD tests pass with AddressSanitizer and UndefinedBehaviorSanitizer.
The repeated-consolidation regression, constrained-allocation reproducer, and
27,046-lookup independent numeric-index oracle pass. Whole-file C++ lint passes
on all eight changed translation units and the changed DD headers;
`uvx nox -s lint` passes. MLIR, Python integration, and executable-documentation
builds were not run for this change. Use the native build and test entry points
in the root agent guide to reproduce the repository checks.

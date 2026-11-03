# OpenQASM correctness and simplification

Status: complete; findings repaired and locally validated.

## Scope and ownership

The frontend parses and validates the supported OpenQASM subset, produces typed
expressions, and owns mutation and initialization facts. QC emission consumes
those facts. OpenQASM, Qiskit, and jeff exporters preserve their consumer
contracts. This change is based on upstream main
`056bef24722dd0a1f0813f3f7b4197c10f8eb950`.

## Decisions

- Propagate mutations from every potentially executing loop, independently of
  definite initialization. Reject quantum indices that cannot be proved.
- Apply the same machine-width integer promotion to constant negation,
  expressions, and range operands. Keep scalar widths at 1–64 bits and the
  existing nonzero runtime-step precondition.
- Preserve arbitrary-width radix literals for bit registers. Share normalization
  and use the fast `APInt` constructor; the general string conversion performs
  repeated full-width multiplication on long decimal literals.
- Keep validated forward whole-register slices compact. Other selections retain
  their bounds, initialization, and distinctness checks. Compare whole-register
  metadata directly instead of materializing every bit reference.
- Merge switch initialization state incrementally and skip affine proofs for
  unchanged scalars. All-exit branches impose no fallthrough initialization
  constraint. Constant affine constraints avoid copying the Presburger domain
  while preserving vacuous truth for empty domains.
- Skip include expansion when there are no includes, and omit provenance slots
  when there are no custom includes. Context IDs fit in 32 bits under the
  existing one-million include-expansion limit. Pack constant fields to reduce
  each optional cache slot from 48 to 40 bytes.
- Emit loop-exit switches as balanced predicates, keeping negative and wide
  labels compatible with downstream consumers. Emit ordinary multi-label
  switches as label-to-group dispatch followed by one body per source case.
  Singleton labels retain their direct native switch. Shared group constants
  avoid duplicate operations. Unsigned index casts preserve 64-bit selector
  values and match Qiskit's switch input contract.
- Create CFG joins only when reached and carry only live enclosing scalar slots.
  Seed the greedy loop simplifier in preorder so its LIFO worklist simplifies
  children first. This avoids repeatedly rewriting parents with large tuples.
- Preserve float-to-i1 truncation through an integer cast before Boolean
  conversion. Stream validated export without another complete output buffer.
  Use hexadecimal for bit constants with more than 64 active bits.
- Resolve common register provenance through exhaustive Boolean jeff switches.
  Expand register writes using the destination array width, since captured
  integer operands may already have been promoted to a wider jeff type. Before
  Qiskit initialization checks, reuse SCF if/while canonicalization in the
  existing normalization driver; keep resource and snapshot rewrites out.
- Extend existing semantic and consumer regressions. Consolidate duplicate
  fixtures and unused test machinery, preserving numerical limits, diagnostics,
  snapshot behavior, phase-sensitive checks, and operation-budget boundaries.
- Preserve compact standard-library controlled-gate names through the shared
  gate catalog. Use generic modifiers when there is no applicable alias.
- Reuse valid, nonconflicting `NameLoc` hints for scalar qubits and scoped
  classical storage in OpenQASM and Qiskit export. Explicit parameter and
  register names take precedence; missing locations permit generated names
  without changing binding, output labels, or execution.
- Retain gate-spelling and name-collision assertions alongside full matrix
  comparison. Remove the unused arithmetic interpreter in the test helper and
  check dynamic store indices through MLIR's dependency analysis.

## Retained checks and rejected complexity

Final region DCE is necessary: instrumentation found it changes 10 of 31 loops
in the existing frontend suite. Keep it after greedy rewriting. Keep parallel
edge snapshots; skipping stores without tracking simultaneous assignments risks
swaps, and public cleanup already removes the observed identity-only case.

Keep the directly indexed constant cache. Representative indexed gate workloads
populate 50–100% of it, so a sparse hash table would increase storage and lookup
cost. Parsed source already dies before public emission; there is no retention
fix to make. Existing bounded scopes and mutation generations do not justify a
general cache or a new scope data structure. Native switches still require
compatible target legalization before jeff export; the direct conversion does
not provide it. The interchange guide documents this existing boundary.

## Measurement and validation

Measurements use GCC 13 Release and LLVM/MLIR 23.1.0 on shared ARM64 hardware.
CPU-affinity comparisons are diagnostic, not reserved-core benchmark claims. Ad
hoc harnesses and raw results remain outside the repository.

- Whole-register slicing avoids 6.4 million index expressions in the
  100,000-bit/64-assignment probe, reducing analysis peak RSS from about 589 MiB
  to 12 MiB, with analysis CPU time reduced from 449 ms to 0.30 ms.
- Incremental merging reduces analysis of 1,024 cases with 1,024 unchanged
  scalars from about 2.57 s to 11 ms.
- Bottom-up loop simplification reduces the distinct-scalar 256-case emission
  probe from 808 ms to 104 ms, with identical final operation/operand counts.
- Shared multi-label bodies reduce the 64-label/128-gate probe from 8,276 to 218
  operations and emission CPU time from 4.03 ms to 1.05 ms.
- Include-free or standard-library-only 100,000-statement inputs retain about
  1.5 MB less parsing heap. Constant bounds reduce the 100,000-initializer
  analysis probe from 260 ms to 216 ms.
- Wide hexadecimal export reduces the 65,536-bit public export probe from 19,796
  to 16,453 bytes and CPU time from about 112 ms to 8 ms.

All 593 native tests pass. Repository lint, full-file clang-tidy for all 14
changed C++ files, and stub regeneration pass; generated stubs are unchanged.
All 1,981 Python tests pass with Python 3.14.7 and Qiskit 2.6.0. The original 34
contract probes, whole/reversed/strided slices, wide-literal parser and round
trips, and eight measured switch paths pass. Switch probes include Qiskit and
target-lowered serialized jeff with captured 9-bit register writes.

The focused Qiskit 2.5.2 compatibility suite also passes all 705 tests.

Eleven exported controlled gates match independent Qiskit reference matrices,
including phase. Scalar qubits and scoped storage retain their source-name hints
in OpenQASM and Qiskit export. Resource counts, output names, and results also
survive round trips through plain MLIR without locations. A deliberate
constant-index substitution fails the new dependency assertion for both store
branches.

Qiskit's optional OpenQASM source importer rejects the existing `if (true)`
wrapper around output-free programs with either named or generic controlled
gates. Native Qiskit export and the modifier syntax probes pass.

Test consolidation removes seven functions and 171 net test lines. No sanitizer,
additional-platform, or reserved-core run is claimed; hosted CI is separate.

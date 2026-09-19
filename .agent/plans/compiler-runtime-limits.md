# Preserve structured compiler and runtime execution

Status: complete and validated locally. QC/QCO share stable register-slot
identity and disjoint quantum operands. The conversion no longer needs
`qc.take`/`qc.put`, transfer tracking, or a loaded-reference cache.

## Scope and decisions

- QC/QCO helper conversion preserves borrowed-register shapes and returned wire
  order through the shared `FunctionUtils` analysis. Entry points keep
  ownership.
- A register slot retains its qubit identity. Extraction borrows that qubit;
  insertion restores it to the same underlying slot. Gates, including SWAP,
  change states rather than slot identities. Views may renumber slots without
  changing their underlying identities. Quantum operands must be disjoint at
  runtime; dynamic index relationships are preconditions, not proof obligations.
- Registers passed to helpers or through structured region boundaries must be
  complete. Scalar helpers can operate on extracted qubits while the remaining
  tensor stays with the caller. The converter preserves positional arguments.
  The builder uses one complete-register check for calls and structured control
  flow and shared argument tracing for positional results, including standalone
  qubits. Call results reuse the existing value-tracking update; register
  snapshots and reassignment to different input slots are unnecessary.
- Immutable entry-block physical reference buffers become tensors of their
  static qubits. Loaded and direct references share slot provenance, and local
  tensors are released at return. Identity stores after initialization reuse the
  same slot checks as other registers. The shared contract uses underlying
  register indices so future views can renumber slots without changing qubit
  identity; this change does not add view lowering.
- `TranslateQCToOpenQASM3.cpp` emits logical indices directly, immutable
  physical reference arrays as site switches, and constant rank-one f64 tables
  as grouped switches. This uses the existing frontend without new array or
  function syntax. Site dispatch retains its 65,536-case and 256-level nesting
  limits; explicit table data is exempt because its expansion is linear in table
  size and reads.
- Indexed loops survive placement only for unrestricted all-to-all targets and
  payloads with multiway branching. Indexed-loop routing needs scalar-wire
  selection and layout handling; raising the unrolling budget does not fix it.
- DD execution limits conditional loops to 100,000 iterations and rejects
  recursive calls. Counted loops use widened APInt trip counts. Runtime
  assertions are omitted by the frontend; verified IR retains its bounds
  preconditions.
- Fixed-gate powers share constant/runtime lowering with rotations and global
  phase. Direct DD interpretation reuses matrix powering for constant bodies;
  arbitrary runtime body matrices remain unsupported.
- Adaptive QIR reuses Boolean storage for computed CBit outputs. Boolean records
  join DDSIM shot/count strings; measurement-only registers retain Result arrays
  and batch sampling. Base-profile restrictions remain unchanged.

## Validation

Use the native build and checks in [AGENTS.md](../../AGENTS.md), plus
`test/python/qdmi/test_compilation.py` against the rebuilt package and provider.
Owning tests cover borrowed-wire order, physical site identity, emission limits,
constant/runtime full matrices at `5e-13`, and mixed computed/measured outputs.
The stable-slot revision passes 3,569 of 3,570 native CTest entries with one
expected Slurm skip, 50 Python compilation tests against the rebuilt package,
executable documentation, generated-link checks, repository lint, and whole-file
C++ lint on the changed sources. Repeated round trips run QC cleanup between
conversions and compare samples for logical and physical registers. Tests cover
references retained across helper calls, identity stores, dynamic index
expressions, scalar-helper boundaries, terminal DDSIM execution, and known
invalid slot changes rejected before mutation. The documented helper-and-loop
example also converts from QCO to QC and back through `mqt-cc`.

The complexity review removed the builder's duplicate call check, register
snapshots, and slot reassignment machinery. It also reused the default identity
else branch in a synthesis test. The mapping test that returned reversed branch
qubits now restores their original result order after applying the reverse-order
GHZ circuit. Earlier tests that carried an incomplete register with its
extracted qubits now check complete boundaries while retaining their
insertion-order assertions.

The native-synthesis and loop-unrolling tests still exercise non-positional
input, constructed directly after builder finalization. Their original
diagnostic, conformance, and permutation assertions remain intact. Both original
physical identity-store reproducers now compile through `mqt-cc --emit=qco`.

Earlier probes executed Shor 65 and 143 OpenQASM with seeded QIR agreement. The
31-bit composite 2147483645 compiled and reimported as 27.8 MB of OpenQASM;
simulation was not attempted. Shared table functions need frontend support.

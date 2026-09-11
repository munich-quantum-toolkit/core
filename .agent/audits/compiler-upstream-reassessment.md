🤖 *AI text below* 🤖 <!-- rumdl-disable-line MD041 -->

# Compiler placement after the upstream refresh

Status: accepted changes implemented and validated locally. Baseline:
`a66a9b20698d5a7450541002a8122352d8275957` (2026-09-10). Reviewed PR #2506 at
`13f94a9207ec76d199c45c22a525ad767aa7755c`.

## Applied changes

- Keep indexed placement, shared if/for/while scalarization, classical constant
  tensor lowering, and partial-register ownership. Current main does not replace
  these fixes.
- Replace the general QIR pointer worklist with upstream's qubit-store roles.
  Adaptive `ConvertQubitStoreOp` marks its LLVM store with `qir.qubit_store`.
  Initial stores contain the placed static IDs, so metadata need not reconstruct
  subsequent loads, GEPs, selects, or block arguments.
- Extend `includeStaticResource` to count `LLVM::ZeroOp` as static ID zero for
  both qubits and results. The null-resource regression checks both capacities.
- Retain upstream's explicit flat-chain traversal in `discoverComputation` and
  remove the superseded TensorIterator guard. Keep both if and while rejection
  tests, with distinct names and the owning mapper's diagnostic.
- Remove Ctrl/Inv/Pow alternatives from `hasCompleteTensorLifetime`: verified
  modifiers only accept scalar qubits.

## Contracts retained

The flat allocation/extraction/insertion/deallocation chain is a supported-input
boundary of the scalar mapper, not a QTensor semantic requirement.
`placeIndexedAllocations` bypasses that traversal only for all-to-all Adaptive
QIR targets whose native operations have no site-specific tuples. It assigns
allocation slots once; runtime indices select among those placed references.
Capacity, static size, entry-block allocation, native operations, and selected
payload capabilities still have their checks.

PR #2505's opaque classical-effect restriction remains in `MappingPass` because
routing reorders operations. All-to-all placement does not invoke that check.
The positional modifier verifier and QCO-to-QC origin collection that skips
verified modifier interiors also remain. Indexed storage does not remove their
wire-correspondence contracts.

QIR metadata covers compiler-marked qubit stores and known scalar QIS operand
roles. Arbitrary external pointer aliases are not inferred. QCO-to-QC keeps its
load-elision cache and the inlining requirement for tensor function boundaries;
occupancy masks separately track partial dynamic ownership. Complete tensor
lifetimes retain the mask-free path.

## Evidence

The disposable combined-tree experiment made both upstream tests fail when it
retained PR #2506's old pointer analysis:
`MetadataDoesNotCountResultReadsAsQubits` and `QIRCountsPackedStaticQubits`. The
role-based replacement passes both, including X/RX and Base/Adaptive packed
arguments. Sparse indexed placement still reports capacity 20 for sites 7 and
19.

With Adaptive loop capability value zero, current main rejects the saved
constant-slot, standard QPE, and RUS tensor loops at the flat-chain check. The
updated implementation compiles them. This establishes additional support, not a
speedup over a successful current-main compilation. Direct QPE still requires
the classical tensor lowering. Current-main direct partial-release output fails
in DDSIM with `QIR qubit was not dynamically allocated`; corrected output
returns the expected result.

Preserve the merged builder, OpenQASM, QTensor, routing, compiler ownership, and
comparison improvements from PRs #2502, #2505, and #2514–#2517. None substitutes
for these fixes. No general pointer analysis, cross-function tensor ownership,
dynamic-size physical placement, or loop-aware routing algorithm is introduced.

## Validation

The [placement audit](compiler-placement-reassessment.md) records the final
native, Python, lint, and documentation checks. Native regressions cover the
retained compiler contracts. The final ponytail review removed superseded
diagnostic artifacts and found no further justified C++ cuts.

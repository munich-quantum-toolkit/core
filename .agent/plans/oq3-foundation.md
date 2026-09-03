# Direct OpenQASM-to-QC frontend

Status: historical implementation record. Later register ownership is described
in [CBit](first-class-classical-registers.md); later indexing contracts are in
[alias-safe loads](alias-safe-repeated-qubit-loads.md) and
[affine indices](openqasm-affine-quantum-indices.md).

## Outcome and architecture

The staged lexer, parser, and semantic analyzer produce a typed source program
without an MLIR context. A private emitter constructs QC directly. The OQ3 MLIR
dialect and its conversion layer were removed because they duplicated work
without providing a useful intermediate compiler representation.

Source legality and target capability remain separate. Parsing and analysis
retain valid source constructs; emission and later conversions diagnose their
supported subsets. The normal acceptance path is QC to optimized QCO, back to
QC, and then QIR. A jeff round trip is optional and has separate acceptance and
boundary tests. A jeff limitation must not reduce valid QC/QIR source support.

The legacy `QuantumComputation` parser was outside this change.

## Ownership

- `mlir/lib/Target/OpenQASM/Frontend.cpp` owns source buffers and parsing.
  Implementation headers under `mlir/include/mlir/Target/OpenQASM/Detail` and
  their sources own syntax, recovery, and analysis.
- `mlir/include/mlir/Target/OpenQASM/Frontend.h` exposes `ParsedProgram`,
  `TypedProgram`, and diagnostic-bearing parse/analysis results.
  `MLIROpenQASMFrontend` remains independent of QC and MLIR contexts.
- `mlir/lib/Dialect/QC/Translation/OpenQASMToQCEmitter.cpp` owns private QC
  construction. `TranslateQASM3ToQC.cpp` is the small public adapter used by
  `QCProgram::fromQASMString`.
- `GateCatalog.h` and `GateCatalog.cpp` share gate identity and lowering recipes
  between semantic analysis and emission. The `oq3::frontend` namespace names
  the language frontend, not a retained MLIR dialect.

## Decisions and constraints

### Resolve source semantics once

Semantic analysis records overload resolution, promotions, assignment
conversions, gate-angle types, ordered output descriptors, and constant facts.
The emitter consumes these decisions rather than inferring source types again.
Source buffers and include identity remain owned so diagnostics retain
locations.

Use the released OpenQASM conversion rules applicable to the implementation, not
unmerged proposals. Retain standard-library identity so strict mode can check
gate membership while the default importer supports common legacy input.
Measurement belongs in the grammar's statement contexts, not as an unrestricted
primary expression.

Distinguish scalar `bit` from `bit[1]`; storage width alone is insufficient for
register-only builtins. Source outputs have one ordered descriptor sequence.
Without explicit outputs, OpenQASM's implicit-output rule includes global
classical scalars as well as registers. Arbitrary scalar QIR output recording
was left to a separate ABI decision.

### Preserve quantum semantics

Emit ordered modifiers as nested QC operations. Numeric powers must not silently
round integer exponents during f64 conversion: exact binary64 representability
is the boundary. Downstream transformations decide which power bodies they can
lower. Preflight rejects unsupported modifiers before constructing part of an
application.

Use explicit U-family recipes because OpenQASM variants and QC attach different
global phases. Those phases become observable under control. Test the emitted QC
recipe independently of mapping, DD construction, and later conversions.

Mixed physical and declared qubits were valid source but unsupported by the QC
builder's allocation mode. That rejection belongs to emission preflight.

### Bound construction and analysis

Custom-gate validation checks recursion, memoized dependency depth, and
transitive structured-control capability. Unused definitions must not constrain
accepted applications. One composed emission budget bounds custom-gate expansion
and dispatch; separate limits do not bound their product. The recorded budget
was 10,000,000 operations, with overflow-safe projection and a builder listener
counting all constructed operations, including scalar and control-flow
scaffolding. Semantic bit state uses copy-on-write storage.

### Preserve classical values and control flow

Signed arithmetic checks overflow and invalid division; unsigned arithmetic uses
its defined wrapping behavior. Constant ranges use `scf.for`; dynamic ranges use
comparison-driven `scf.while`, preserving inclusive endpoints, negative steps,
and empty ranges without overflowing a trip-count formula.

Both QC/QCO conversion directions preserve classical loop state alongside
quantum state. Terminators need final region-local mappings; a second conversion
phase avoids traversal-order dependence. Regressions use native IR independently
of the OpenQASM parser.

The original emitter used runtime bounds checks and switch-based qubit
selection. The linked index plans supersede that implementation. Likewise, the
original output-register memrefs were replaced by CBit. Preserve the underlying
requirements: aliases must not violate QCO linearity, ordinary local state must
not become output recording, and observable outputs follow function-result order
rather than hash-map order.

### Keep builtin behavior explicit

The implementation added scalar rounding and register population count and
rotation. Constants fold during semantic analysis; runtime forms use standard
Math/LLVM operations. Bit zero is least significant. Rotation assignment reads a
snapshot, normalizes positive or negative distances modulo width, and retains
packed values until unpacking is necessary. Cached facts must be invalidated by
register mutations and control-flow joins.

Retained rounding, population-count, and funnel-shift operations had explicit
QIR/jeff conversion limits. Constant-folded success did not establish support
for their runtime forms. Broader initialization, casts, sized-uint overloads,
bit-string literals, and arbitrary scalar QIR outputs were recorded follow-ups;
this historical document does not claim that they remain unresolved today.

## Validation

The historical corpus exercised direct QC, optimized QCO, reconstructed QC,
Adaptive QIR, and the default pipeline; straight-line cases additionally reached
Base QIR. A separate jeff corpus distinguished supported round trips from
programs rejected at `intoJeff()`. Tests inspected observable results at every
stage, not merely success at the final QIR stage. Intermediate jeff storage was
compared by semantic shape while QC/QCO result types remained exact.

The compiler fixtures live in `mlir/unittests/programs/qasm_programs.cpp`.
Translation tests live under `mlir/unittests/Dialect/QC/Translation/`, and
public pipeline tests under `mlir/unittests/Compiler/`. Parser, semantics, and
emitter tests retain separate responsibilities. Share fixture names and sources;
do not embed expected-failure flags that silently turn unsupported cases into
accepted behavior.

Each downstream correction has a minimized native-IR regression plus the
full-chain fixture that exposed it. Recorded checks covered U-family matrices,
integer powers, construction limits, scalar conversions, rotation results, wide
bit vectors, output order, and retained entry-point metadata. Historical
coverage measurements describe that implementation; they are not current test
results or a substitute for the behavioral checks.

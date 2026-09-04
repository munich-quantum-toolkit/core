# Emit OpenQASM 3 from QC compiler programs

Status: historical implementation record.

## Goal and scope

MQT Core can currently translate OpenQASM 3 into the MLIR QC dialect, optimize
that program through QCO, and emit QIR, but it cannot turn the resulting QC
program back into a standardized circuit language. After this work, C++ and
Python clients can request an `OpenQASMProgram`, and `mqt-cc` can write OpenQASM
text with `--emit=openqasm3`. A user can demonstrate the feature by compiling a
Bell-state program to OpenQASM, parsing the emitted source with the existing
strict frontend, and compiling the reparsed program onward.

The exporter is deliberately independent of the legacy `include/mqt-core/qasm3`,
`src/qasm3`, `include/mqt-core/ir`, and `src/ir` components. It reads the QC,
arith, math, memref, func, and SCF operations already present in an MLIR module.
Programs whose meaning relies on dynamic-index safety checks are rejected
instead of reverse-engineering the importer machinery.

## Constraints

- The compiler API and `mqt-cc` currently implement the same output-stage
  decisions separately. Evidence: `runDefaultPipeline` in
  `mlir/lib/Compiler/Programs.cpp` converts optimized QCO back to QC before QIR,
  while `mlir/tools/mqt-cc/mqt-cc.cpp` constructs the equivalent pass sequence
  directly.

- QC modifiers already implement `qc::UnitaryOpInterface` and their regions are
  verified to contain only the small set of operations valid in a unitary body.
  This permits a uniform exporter without depending on the legacy circuit
  classes.

- OpenQASM helper-gate formals need ordinary identifiers such as `p0`; reusing
  SSA-derived names made strict reparsing unnecessarily fragile. Full-matrix
  Qiskit comparisons also showed that OpenQASM's built-in `U` and the QC
  `u`/`u2` operations agree without an extra global phase.

- The current strict frontend accepts nested `else { if (...) }` control but not
  explicit scalar-cast syntax. The exporter therefore emits standard OpenQASM
  casts and documents cast-containing programs outside the current strict
  round-trip subset.

- Importing an integer `while` condition introduces checked arithmetic and
  safety operations. Direct safety-free, type-preserving `scf.while` is
  exportable, while the importer-produced checked form correctly reaches the
  documented unsupported boundary.

- Removing returned measurements during QC-to-QIR changed a function's result
  arity without realigning result attributes. Preserving OpenQASM output hints
  exposed the latent Func-to-LLVM assertion. Realigning the ordinary MLIR
  result-attribute array at that existing arity-changing operation fixes the
  issue without a metadata side channel or preservation pass.

- A documentation-only top-level CMake configuration currently assumes binding
  targets exist. The final documentation validation generated MLIR reference
  pages with bindings enabled, then ran Sphinx in nitpicky, warnings-as-errors
  mode.

- The OpenQASM standard library defines `sx` and the compatibility alias `u2`,
  but not `sxdg`; the language-level `U` gate covers QC `u`. Emitting
  `inv @ sx`, `u2`, and `U` removes three unnecessary helper definitions.

- OpenQASM switch cases permit multiple constant integer labels and do not fall
  through. Representing them directly as `scf.index_switch` preserves structured
  control and eliminates the exporter's nested-if reconstruction.

- Lowering a source case with several labels creates one `scf.index_switch`
  region per label. The importer's projected-emission preflight must therefore
  multiply the case body cost by its label count; doing so keeps the existing
  operation budget effective before constructing any IR.

- MSVC models `llvm::find_if` over `std::array` as an array iterator, not a
  pointer. Removing the ECR helper dependency lookup eliminated both the
  non-portable pointer deduction and an unnecessary two-RZX decomposition.

- Source output spelling is not needed for the practical export subset. A
  canonical mapping from QC types and direct measurement provenance removes the
  result metadata entirely and makes the accepted semantics explicit.

- Zero-state `scf.while` conditions imported from mutable classical variables
  contain read-only `memref.load` operations. Accepting those loads while
  rejecting writes and other effects preserves practical statement-only loops.

## Decisions

- Emit OpenQASM 3.1 with `stdgates.inc`, using standard gate names where
  possible and self-contained helper gate declarations for the remaining QC
  gates. Rationale: This keeps emitted programs portable and parseable in strict
  mode without MQT-specific language extensions.

- Buffer the complete translation before writing to a caller stream. Rationale:
  Unsupported operations must not leave a syntactically truncated output file.

- Treat dynamic memory indices, `cf.assert`, live `ub.poison`, and checked-index
  scaffolding as unsupported. Rationale: The user explicitly prefers a focused
  practical exporter over reconstruction of importer safety machinery.

- Make `OpenQASMProgram` an owned textual value rather than an MLIR `Program`
  subclass, and accept that value directly as compiler input. Rationale:
  OpenQASM owns source text rather than an MLIR context and module; reparsing it
  at the compiler boundary is straightforward and keeps the value reusable.

- Export from optimized QC in the coordinated pipeline but expose a
  non-consuming direct method on `QCProgram`. Rationale: The compiler output
  should reflect normal optimization, while callers inspecting frontend QC need
  a predictable direct path.

- Do not attach OpenQASM-specific attributes to function results. Canonicalize
  bit memrefs to bit arrays, direct measurements to scalar bits, other `i1`
  values to booleans, integers to signed `int`, and `f64` to `float`. Rationale:
  The smaller contract is predictable, requires no metadata transport, and
  explicitly rejects operations whose unsigned meaning cannot be preserved.

- Run the existing QC cleanup pipeline on a copy before direct `QCProgram`
  export. Rationale: Dead importer scaffolding should be removed by MLIR passes,
  not by a recursive exporter heuristic, while the caller's QC program and the
  low-level translator's strict validation contract remain unchanged.

- Split the modern OpenQASM translations into `MLIRQCOpenQASMTranslation` and
  retain the legacy circuit translations in `MLIRQCTranslation`. Rationale:
  Compiler clients can use either OpenQASM direction without linking
  `MQT::CoreIR`, while existing users retain the aggregate target.

- Parse and emit native `switch`/`case`/`default` statements and map them
  directly to `scf.index_switch`. Rationale: This follows the language construct
  and is simpler than synthesizing nested conditionals. The export subset
  accepts only result-free switches.

- Emit compatibility gates under their catalog names and have the default
  MQT-compatible frontend prefer a matching catalog signature over the helper
  body. Rationale: Strict consumers retain self-contained definitions, while MQT
  round trips recover native QC operations without a duplicate helper name list.

- Restrict structured control to result-free `if` and `switch`, constant `for`
  without iterated state, and zero-state `while`; reject `arith.select`.
  Rationale: This covers practical structured quantum programs while deleting
  result declarations, yield-target plumbing, and carried-state bookkeeping.

## Outcome and validation

The translation API, compiler artifact, bindings, and CLI emit the documented
structured subset deterministically. Canonical output types and measurement
provenance determine outputs; no OpenQASM-specific result metadata is attached
to QC functions. Unsupported shapes fail before buffered output is committed.

Release builds, focused frontend/translation/compiler suites, Python tests,
stubs, strict documentation, changed-unit clang-tidy, and lint passed. Focused
coverage was measured; the complete coverage build did not finish. That
limitation does not invalidate the focused results or establish full coverage.

## Code and ownership

`mlir/include/mlir/Compiler/Programs.h` defines typed compiler artifacts and the
`ProgramFormat` enum. `mlir/lib/Compiler/Programs.cpp` imports OpenQASM,
performs QC/QCO/QIR conversions, and coordinates the default pipeline. The new
textual artifact and pipeline branch belong there.

`mlir/include/mlir/Dialect/QC/Translation/TranslateQASM3ToQC.h` and
`mlir/lib/Dialect/QC/Translation/` contain the modern OpenQASM frontend and its
QC emitter. The reverse translator will live beside them under the separate
public header `TranslateQCToOpenQASM3.h` and will be compiled into
`MLIRQCOpenQASMTranslation`. It may use MLIR and QC libraries already linked by
that target but must not include or link the legacy QASM or circuit libraries.
The existing `MLIRQCTranslation` target remains an aggregate for legacy users.

`bindings/mlir/register_mlir.cpp` exposes compiler artifacts through nanobind.
`bindings/patterns.txt` supplies handwritten overloads to the stub generator.
Generated `.pyi` files are updated only by `uvx nox -s stubs`.

`mlir/tools/mqt-cc/mqt-cc.cpp` is the standalone driver. It imports QASM or
MLIR, runs QC-to-QCO optimization, converts requested outputs, and writes
through LLVM output streams. OpenQASM must be treated as a textual QC-stage
output rather than MLIR bytecode.

The QC dialect represents logical scalar qubits with `qc.alloc`, physical qubits
with `qc.static`, and registers as rank-one memrefs containing `!qc.qubit`.
Gates implement `qc::UnitaryOpInterface`, which exposes parameters, targets, and
controls. Classical structured control uses MLIR SCF operations. The exporter
maps SSA values to deterministic generated OpenQASM variables.

## Acceptance

The public translator succeeds for a representative QC module and returns
strictly parseable OpenQASM 3.1. Passing its text to `translateQASM3ToQC`
produces a verified QC module. Gate tests cover each standard or generated gate
and modifier nesting; helper definitions are compared against the QC unitary,
including global phase.

Programs with measurement, output values, arithmetic gate parameters, reset,
barrier, result-free nested conditionals and switches, constant-range loops
without iterated state, and zero-state while loops emit structured OpenQASM
rather than unrolled or flattened control flow. Tests compare observable
operations and result kinds rather than exact incidental temporary names, while
a separate determinism test checks byte-identical repeated emission.

A module with `cf.assert`, live poison, dynamic memref indexing, an unsupported
type, arbitrary CFG, multiple functions, calls, or an unknown operation fails
with a location-based diagnostic. The destination stream remains empty.

`QCProgram::toOpenQASM3()` runs the existing QC cleanup pipeline on a copy and
returns an `OpenQASMProgram` without consuming or mutating the QC program.
`runDefaultPipeline(..., ProgramFormat::OpenQASM3)` returns the textual program
after the normal QCO optimization round trip. An `OpenQASMProgram` may also be
passed directly as compiler input and remains reusable. Python exposes the same
behaviors and `mqt-cc --emit=openqasm3` writes plain text to stdout and files.

The focused native and Python suites pass, generated stubs match the binding,
documentation builds, lint passes, and `git diff --check` reports no whitespace
errors.

## Interfaces

The final C++ translation interface is:

    namespace mlir::qc {
    LogicalResult translateQCToOpenQASM3(ModuleOp moduleOp,
                                         llvm::raw_ostream& output);
    FailureOr<std::string> translateQCToOpenQASM3(ModuleOp moduleOp);
    }

`OpenQASMProgram` owns a `std::string`, provides `source()` and `write(path)`,
and appears in both `CompilerProgram` and `CompilerInput`. `QCProgram` provides
`std::optional<OpenQASMProgram> toOpenQASM3() const`.

The translation target may link MLIR Arith, ControlFlow, Func, Math, MemRef,
SCF, UB, and QC libraries, plus LLVM Support. It must not link the legacy
`mqt-core-qasm`, `mqt-core-ir`, or circuit targets.

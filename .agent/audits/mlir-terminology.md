# Compiler terminology and public naming

Status: complete. Baseline: `a66a9b20698d5a7450541002a8122352d8275957` (upstream
main).

## Scope and evidence

The audit follows
[issue #2251](https://github.com/munich-quantum-toolkit/core/issues/2251),
including its proposed changes and the linked
[PR #2149 discussion](https://github.com/munich-quantum-toolkit/core/pull/2149#issuecomment-5330271608).
The naming decisions below were approved for implementation without aliases.
Terminology follows `docs/glossary.md` and the
[MLIR glossary](https://mlir.llvm.org/getting_started/Glossary/).

Reviewed the compiler headers and implementations, Python bindings and stub
patterns, OpenQASM frontend and QC importer/exporter, CLI format dispatch,
QC/QCO/QTensor operation and builder descriptions, conversion/pass descriptions,
documentation, and their callers and tests. Existing plan references were
updated with the renamed symbols.

## Applied findings

1. **OpenQASM import names omitted the language name or implied version 3.**
   `OpenQASMSemantics.cpp::analyzeVersion` accepts versionless input, 2.0, 3.0,
   and 3.1. Both `QCProgram` factories and the lower-level importer route
   through that frontend. Rename the complete import surface and document its
   actual version contract. Correct the options documentation to identify
   `gatePolicy`, the field read by the importer.
2. **The CLI advertised MLIR as a specific output checkpoint.**
   `parseOutputFormat` mapped both `mlir` and `qc` to `OutputFormat::QC`. Use
   `qc` exclusively and advertise it as the default. QC, QCO, and the jeff
   dialect all use MLIR; `--input-format=mlir` and isolated-pipeline MLIR output
   remain accurate. Filename inference still recognizes `.qasm`.
3. **Target metadata shadowed MLIR operations.** `CompilerTarget::Operation`
   describes native support and calibration, while `supports` consumes
   `mlir::Operation` instances. Rename the capability class consistently in C++,
   Python, binding patterns, tests, and examples.
4. **Gate catalog identifiers were named after a transformation.**
   `GateLowering` was an alias for `qc::StandardGate`, and `lowering` stored
   that enum. Remove the alias and use the existing type with the field `gate`.
   Catalog lookup, parameter counts, and gate emission are unchanged.
5. **jeff conversion and serialization were conflated in Python.**
   `QCOProgram::intoJeff` runs a dialect conversion; the resulting `JeffProgram`
   retains an MLIR module. Byte/file methods perform serialization. Align the
   Python descriptions and guide with the existing C++ ownership contract.
6. **Operation, pass, target, and ownership descriptions were imprecise.**
   Describe canonicalization patterns separately from passes, barriers as
   operations that preserve quantum state, and dialect changes as conversions.
   Distinguish MQT compiler targets from MLIR conversion targets. Describe QCO
   and QTensor use constraints as linear semantics/ownership, checked by
   `qco::verifyLinearity`, rather than implying type checking alone enforces
   them. Replace the generic Python failure text with `Compiler action failed`
   and name the QC translation in unsupported-input diagnostics.
7. **Gate-count summaries did not identify their static scope.**
   `Programs.cpp::countGatesIf` walks the entry point, skips barrier counts and
   modifier bodies, and does not follow calls. Every SCF region is visited once.
   Clarify that scope in all three C++/Python count methods and the guide,
   consistent with PR #2149. Counting behavior is unchanged.

## Public migration impact

These are unreleased v4 interfaces. Development-branch users must update the
following names and rebuild C++ consumers; there is no compatibility alias or
deprecation period. A deprecated forwarding layer was considered and rejected in
favor of completing the naming changes before v4.

| Previous spelling                                        | Replacement                                                                  |
| -------------------------------------------------------- | ---------------------------------------------------------------------------- |
| `QCProgram::fromQASMString` / `fromQASMFile`             | `fromOpenQASMString` / `fromOpenQASMFile`                                    |
| `QCProgram.from_qasm_str` / `from_qasm_file`             | `from_openqasm_str` / `from_openqasm_file`                                   |
| `translateQASM3ToQC`                                     | `translateOpenQASMToQC`                                                      |
| `TranslateQASM3ToQC.h`                                   | `TranslateOpenQASMToQC.h`                                                    |
| `QASM3ImportOptions`                                     | `OpenQASMImportOptions`                                                      |
| `mlir::oq3::frontend`                                    | `mlir::openqasm::frontend`                                                   |
| `CompilerTarget::Operation` / `CompilerTarget.Operation` | `CompilerTarget::OperationCapability` / `CompilerTarget.OperationCapability` |
| `GateLowering` / `GateCatalogEntry::lowering`            | `qc::StandardGate` / `GateCatalogEntry::gate`                                |
| `--input-format=qasm`                                    | `--input-format=openqasm`                                                    |
| `--emit=mlir`                                            | `--emit=qc`                                                                  |

In-tree consumers include compiler pipelines, DDSIM import, target adapters,
Python bindings, compiler/translation tests, and documentation examples. All are
updated together. Existing unreleased changelog and upgrade-guide examples use
the final names; no separate upgrade section is added for unreleased APIs.

GitHub code searches of indexed MQT repositories found no affected downstream
calls. The QMAP compiler-target consumer found by search uses `from_device_id`,
which is unchanged. Searches for the old importer in the MQSC organization found
no matches. These checks do not cover private/unindexed code or nondefault
branches. Open Core PRs #2260 and #2506 touch compiler inspection or pipeline
surfaces and may require mechanical conflict resolution after this change.

## Deliberately retained terminology

- OpenQASM 3 export names and QDMI's external `QASM3` format constants identify
  a versioned output contract and remain unchanged.
- Python convenience functions use the `OPENQASM` header to recognize source
  strings. Versionless text uses the explicit `QCProgram` factory; `.qasm` file
  import also accepts it. Document this boundary without changing source
  detection or guessing whether a string is a path.
- `LinearType`/`isLinearType` classify quantum types to which ownership rules
  apply; they do not claim that MLIR's type system proves linearity.
- QIR and CBit lowering genuinely move toward a lower-level representation. QC
  target operands are gate operands, distinct from compiler targets.
- Static qubit identifiers and dynamic resource allocation retain their
  established names. The glossary distinguishes them from known quantum states
  and compile-time allocation sizes.
- The reported cast phrase about "different value semantics" no longer exists at
  the baseline. Current cast descriptions specify numeric behavior.

## Validation

- Native Clang 23/LLVM-MLIR 23.1 release build with ThinLTO: passed.
- `ctest --preset release-clang-ipo`: 3,505 passed, one existing sample-device
  job-ID query skipped. Includes compiler, OpenQASM, conversion, DDSIM, and CLI
  coverage. The final target-test formatting correction also passed all 19
  `CompilerTargetTest` cases.
- `uvx nox -s tests` with `test_mlir.py`, `test_mlir_integer_interchange.py`,
  `test_mlir_loops.py`, `test_mlir_qiskit_translation.py`, and `test_qco_dd.py`
  under `test/python/`: 653 tests passed on each of Python 3.11, 3.12, 3.13, and
  3.14. The final versionless-input example passed the four focused
  import-version cases.
- `uvx nox -s stubs`: passed; regenerated the MLIR stub from the bindings.
- `uvx nox --non-interactive -s docs`: passed, including executable examples and
  generated local-link checks.
- `uvx nox -s lint`: passed.
- `uvx nox -s cpp-lint -- a66a9b20698d5a7450541002a8122352d8275957`: passed with
  zero findings across the complete changed C++ files.
- The final symbol search found old public spellings only in the audit's
  migration discussion and CLI rejection tests. CLI help advertises `openqasm`
  input and the default `qc` output checkpoint.

These are local results; hosted CI is separate.

## Complexity review

The Ponytail review found one redundant forwarding helper in the touched gate
emitter. Removed `emitPrimitive` and called `qc::emitStandardGate` directly at
all three call sites, saving six lines. No new abstraction or dependency is
needed. The final pass found no further complexity to remove.

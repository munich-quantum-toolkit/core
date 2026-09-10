# Audit: MLIR tests, diagnostics, and debugging

Status: proposed findings; experimental changes are not applied. Baseline:
`d994fe6833b6b7a9b1bccdeccc64e31c5c09ffd1`, current main when this audit began.
Date: 2026-09-10. There were no in-scope uncommitted changes at the baseline.

This audit addresses [issue #2254][issue] and the release criteria in its
[parent issue #2250][parent]. It covers MLIR test infrastructure, dialect and
conversion tests, compiler and frontend tests, Python MLIR boundaries,
diagnostics, and the documented `mqt-cc` debugging workflow. Source references
and line numbers below refer to the pinned baseline.

Before publication, main was checked again at
`4faf68e3ae5b31bc5dbbd434e5e3ce5795096088`. The intervening commit, PR #2501,
changes mathematical notation in comments and documentation. Its overlapping
changes do not alter the reported behavior. Executed results remain tied to the
baseline above; they are not a new run against the publication-time main.

## Result

The existing suite is green, but focused negative probes expose two process
crashes and an unsound shared test oracle. Fix the crashes before v4. Strengthen
the oracle before relying on it to approve semantic compiler changes.

| Rank | Finding                                                            | Impact                                                       | Confidence                                                   |
| ---- | ------------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1    | Program move assignment destroys its context before its old module | Valid-program C++ crash; pre-v4 blocker                      | Reproduced; lifetime-order experiment passes                 |
| 2    | Several consumed-program Python methods abort the interpreter      | Public Python error handling can terminate the process       | Three independent subprocess reproductions                   |
| 3    | Shared IR comparator accepts seven semantic changes                | False-positive conversion and optimization tests             | Seven verified input pairs reproduce it                      |
| 4    | Driver drops diagnostic notes, including requested stack traces    | Advertised debugging option produces no trace                | CLI reproduction and exact LLVM source                       |
| 5    | Driver and library expose different, partly broken pass interfaces | Common pipeline names fail; individual flags misbehave       | CLI reproductions                                            |
| 6    | Driver has no complete isolated-pass/reproducer replay workflow    | Reduced cases cannot be run with exactly the saved pipeline  | CLI trace, saved reproducer, source ownership                |
| 7    | Initial jeff conversion bypasses pass-manager options              | Import failures fall outside normal pass debugging           | Successful jeff import produces no requested dump            |
| 8    | QC-to-QCO repeats modifier verification owned by QC                | Extra whole-IR walk and misplaced negative tests             | 545 relevant tests pass after removal and retargeting        |
| 9    | A conversion rejection fixture violates QCO linearity              | Test does not establish its intended valid-input boundary    | Corrected fixture passes both verifiers and fails conversion |
| 10   | Handler-restoration test performs no subsequent diagnostic         | A success-only check cannot establish restored error routing | Added negative action emits through the restored handler     |

The full baseline C++ run reports 3,454 CTest entries, with one intentional QDMI
query skip. The four Python MLIR test files pass all 606 cases. These are
baseline validation results, not proof that the reported defects are absent. QC
and QCO IR each group many GoogleTests into one CTest entry.

## Contracts and coverage

The important contracts are:

- `docs/mlir/development.md:55-74`: passes accept verified input, successful
  output verifies, operation verifiers own their invariants, and unsupported
  valid input gets an actionable diagnostic.
- `docs/mlir/development.md:78-95`: QCO linearity is a separate whole-IR
  invariant. Ordinary operation verification does not establish it. Linearity
  also does not establish positional wire correspondence.
- `docs/mlir/development.md:117-151`: direct GoogleTest/CTest coverage, semantic
  assertions, meaningful valid/invalid cases, and the debugging workflow named
  in the issue.
- `mlir/include/mqt/Compiler/Programs.h:66-89`: a program owns its module and
  keeps the context alive for that module's full lifetime; consumption is an
  explicit state transition.
- `docs/mlir/OpenQASM.md:54-70` and
  `mlir/include/mqt/Dialect/QC/Translation/TranslateQASM3ToQC.h:38-60`: frontend
  errors retain source information and use the context's diagnostic engine.
  Export failure leaves the supplied output stream unchanged.
- `docs/mlir/target_compilation.md:220-229`: failed in-place target compilation
  may leave changed IR; its Python exception includes diagnostics. Rollback is
  not a general contract.

The inventory contains 73 C++ test source files and 1,617 `TEST`, `TEST_F`, or
`TEST_P` declarations under `mlir/unittests`, before parameter expansion. The
audit also reviewed the shared support and program-builder files, test CMake,
the four `test/python/test_mlir*.py` files, bindings, driver, and documentation.
It covered all dialect/conversion families and test entry points. Investigation
was deeper around shared oracles, error boundaries, and observed failures; this
is not a claim that every assertion has an independent correctness proof.

Related work was checked on GitHub and in the existing audit records. The
applied build/test, dialect-canonicalization, QC/QCO infrastructure, and
documentation improvements are not reported again. Open PR #1555 concerns MLIR
getting-started documentation; coordinate debugging examples with that work.
Open PR #2495 concerns device-directed compilation and overlaps compiler entry
points, but its unmerged code is outside this baseline.

## Findings

### 1. Preserve context lifetime during move assignment

`Program::operator=(Program&&)` is defaulted at `Programs.h:76`. Its storage
declares the context before the module at `Programs.h:92-95`. Memberwise move
assignment therefore releases the destination context first. LLVM 23.1's
`OwningOpRef::operator=` then erases the destination's old module.

Two independently imported, valid programs reproduce a segmentation fault:

```cpp
auto first = QCProgram::fromQASMString("OPENQASM 3.0; qubit q; h q;");
auto second = QCProgram::fromQASMString("OPENQASM 3.0; qubit q; x q;");
*first = std::move(*second);
```

The probe reaches the assignment and terminates with SIGSEGV. It does not need
malformed IR or an unusual compiler pipeline. Existing ownership tests cover
construction, copying, adoption, and consumption, but miss assignment between
independent contexts.

**Change:** explicitly assign/destroy the old module while its context is still
alive, then transfer the context. Handle self-assignment. The disposable
[lifetime-order experiment][lifetime-patch] makes the new case and all 204
existing compiler cases pass: 205 tests total.

**Risk:** changing declaration order is insufficient: it would break ordinary
destruction order. Keep the context/module lifetime relationship explicit in
assignment and test independent contexts. The experiment proves this crash's
cause; it is not a complete audit of every ownership transition.

### 2. Guard consumed programs at the Python boundary

Python's `ir`, conversions, target compilation, and Qiskit export already turn
use after consumption into `RuntimeError`. Direct `copy` bindings, generic
method adapters, and high-level typed-input conversion do not all use that
guard. Relevant sites include `bindings/mlir/register_mlir.cpp:118-148`,
`:283-296`, `:1105-1111`, `:1217-1226`, and `:1339-1352`.

Each of these was run in a fresh subprocess against the rebuilt baseline:

```python
from mqt.core.mlir import QCProgram, compile_program

p = QCProgram.from_qasm_str("OPENQASM 3.0; qubit q;")
p.to_qco()
assert not p.is_valid
# Run each following action separately:
p.copy()
p.cleanup()
compile_program(p)
```

All three actions abort with SIGABRT at `Programs.cpp:167`, the assertion
against accessing a consumed module. The corresponding `p.ir` control raises the
expected `RuntimeError` instead.

**Change:** use the existing validity check in program-backed binding adapters
and the typed-input boundary, and wrap direct bindings that bypass them. The
generic Boolean adapter also wraps `OpenQASMProgram::write` at `:1369`; that
class is not a consumed-module `Program`, so do not impose this guard on every
adapter instantiation. Add a small parameterized regression over public
operations on consumed program kinds. Do not scatter a second state
representation across Python wrappers.

**Risk:** preserve ordinary C++ ownership and consumption semantics. The
assert-enabled build exposes the abort; a build without assertions is not a safe
workaround. This finding concerns controlled error handling for the existing
invalid state, not making consumed objects reusable.

### 3. Replace permissive comparison rules with explicit equivalence rules

`mlir/unittests/Support/IRVerification.cpp` has 838 lines at the baseline and 85
direct comparison call expressions across 16 test source files. It acts as a
small custom IR-equivalence engine. Its broad name hides several exclusions:

| Exclusion                                                                     | Baseline source        | Reproduced false equivalence                           |
| ----------------------------------------------------------------------------- | ---------------------- | ------------------------------------------------------ |
| Unrecognized attribute kinds compare equal                                    | `:288-381`             | Dense constant `1` versus `2`                          |
| Function type attributes and block argument types are not compared completely | `:293-381`, `:597-660` | Function argument `i32` versus `i64`                   |
| Missing attributes are ignored                                                | `:400-411`             | Added/removed `llvm.emit_c_interface`                  |
| Readiness uses SSA dependencies, not side-effect ordering                     | `:500-592`             | `qc.h; qc.x` versus `qc.x; qc.h` on the same reference |
| QCO condition is omitted                                                      | `:437-441`             | `qco.if` uses argument `%a` versus `%b`                |
| QCO switch selector is omitted                                                | `:443-450`             | `qco.index_switch` uses `%a` versus `%b`               |
| Successor destinations are omitted                                            | `:389-396`, `:477-494` | Swapped `cf.cond_br` successors returning `1` and `0`  |

All seven input pairs parse, pass ordinary verification, and pass QCO linearity
verification. All seven negative comparisons fail because the helper returns
`true`. The [probe patch][oracle-patch] contains complete inputs.

**Immediate change:** make attributes/types, control predicates, successors, and
reference-side-effect order significant. Add negative tests alongside positive
equivalence cases. An unknown attribute should compare exactly unless a specific
contract says otherwise; missing metadata should not silently disappear. Compare
classical constants exactly; apply numerical tolerance only in semantic oracles
whose contracts permit it.

**Simplification experiment:** replace the attribute-kind whitelist with exact
attribute equality plus the existing floating/array exceptions. This removes 76
lines from the helper. It is an interim strengthening, not a sound final
structural comparator. The first full-suite experiment exposed one fixture
mismatch: `QCOTest.IndexSwitchParser` allocates an undefined CBit register at
`test_qco_ir.cpp:1530`, while its builder reference initializes to zero
(`QCOProgramBuilder.h:330-332`). Both write every bit before returning it, so
this parser test need not exercise different initialization policies. Align that
one fixture's initialization; do not restore a blanket attribute bypass. With
the [combined patch][attribute-patch], all 579 QCO cases and the new dense
constant and function-type negative cases pass. The full suite was run with the
helper-only change; the final fixture adjustment was checked with those focused
binaries.

The remaining floating exception needs a separate decision. From the source,
`0.0` and `1e-16` fall within its `1e-15` absolute tolerance. Used as constants
in `arith.cmpf oeq` against a function argument, they yield different booleans
when that argument is zero. This additional counterexample is source-derived,
not an executed eighth probe. Existing approximate quantum comparisons do not
justify approximate classical control decisions.

**Larger opportunity:** use upstream `OperationEquivalence` with locations
ignored for structural round trips, and the existing full-unitary or execution
oracles plus required normal-form checks for transformations. Restrict any
remaining permutation helper to explicitly supported pure QCO shapes. Do not
treat arbitrary QC reference operations or control flow as freely reorderable.
The current readiness algorithm also repeatedly scans the remaining operations;
a chain with one newly ready operation per round has quadratic scan work. That
is a source-level complexity argument, not a measured speedup.

**Risk:** a wholesale replacement by exact textual equality would reject
legitimate independent-operation and tensor permutations. Full-unitary checks
are unsuitable for unrestricted dynamic, measured, or large programs. Preserve
phase, wire identity, metadata, numerical limits, and resource bounds while
migrating callers. The 76-line experiment only addresses attributes; it does not
fix the remaining semantic exclusions or justify deleting all 838 lines.

### 4. Keep a source-aware diagnostic handler alive through compilation

The driver registers `--mlir-print-stacktrace-on-diagnostic`, but the verified
unsupported CFG input produces only its primary location/error, with no trace.
The same loss affects attached operation notes.

This is explained by exact [LLVM 23.1 diagnostic source][llvm-diagnostics]:
`emitDiag` attaches the trace as a note, while the fallback
`DiagnosticEngineImpl::emit` prints only the primary diagnostic.
`SourceMgrDiagnosticHandler` prints notes and source excerpts. The driver has no
such handler around its complete compilation. Its QASM and MLIR loaders own
short-lived source managers at `mqt-cc.cpp:262-291`.

**Change:** create one source manager and upstream diagnostic handler beside the
context, pass that manager into the input loaders, and retain them through all
passes and output. Add a CLI regression that asks for a stacktrace and observes
a diagnostic note, without pinning addresses, frame counts, or exact platform
stack formatting. Also check an ordinary source location and message.

**Risk:** merely adding a handler inside a loader misses later pass failures. An
empty manager can restore notes but cannot retain stdin/include buffers. Use the
upstream handler rather than writing another note renderer.

### 5. Give CLI and library pipelines one deliberate interface

The library's `runPassPipeline` registers both MQT and upstream transform passes
(`mlir/lib/Support/Passes.cpp:113-117`). The driver registers only MQT passes
(`mqt-cc.cpp:378`). Consequently,
`--pass-pipeline='builtin.module(canonicalize,cse)'` fails with an unknown-pass
diagnostic, despite the shared pass-name interface documented at
`docs/mlir/python_compiler_collection.md:197-208`.

The named `PassPipelineCLParser("passes", ...)` also advertises individual
passes that cannot be invoked as shown. Executed examples:

| Invocation                                         | Baseline outcome                                 |
| -------------------------------------------------- | ------------------------------------------------ |
| `--pass-pipeline=builtin.module(hadamard-lifting)` | Runs the pass, then cleanup                      |
| `--passes=hadamard-lifting`                        | Exit 1; interprets the pass name as pass options |
| `--measurement-lifting`, `--reuse-qubits`          | Exit 1; unknown argument                         |
| `--hadamard-lifting`                               | Exit 0 but prints help, not compiled IR          |

**Change:** register the shared supported pass set once and prefer the already
documented textual pipeline interface. A textual `--pass-pipeline`, optionally
with a deliberate `--passes` alias, can use native MLIR parsing without the
broken individual-pass surface.

**Risk:** simply changing the parser name to an empty string introduces a
collision with the existing convenience flag `--decompose-multi-controlled`.
Decide supported syntax explicitly and check its help, failures, and actual pass
execution. A zero exit code or presence in `--help` is not enough.

### 6. Document and provide an exact debugging/replay route

Textual custom pipelines replace the existing `OpPassManager` contents in
[LLVM's `PassPipelineCLParser::addToPipeline`][llvm-pass-registry]. Thus the
driver's previously populated leading cleanup, and its pre-QIR inliner, are
discarded. The cleanup appended after parsing remains (`mqt-cc.cpp:584-610`). A
trace of a two-Hadamard QCO function shows HadamardLifting first, followed by
six cleanup passes.

The initial source-only suspicion that every textual pipeline runs cleanup on
both sides was rejected by execution. Do not use it as a finding.

There is also no direct raw-pass mode: `--emit=qco` and `qc-import` reject a
custom pipeline, while other output modes impose further stages. Crash
reproducer generation works for the tested conversion failure, but normal driver
parsing does not apply the saved reproducer resource's pipeline, verification,
and threading settings. Core conversion passes such as `qc-to-qco` are not
registered for textual replay either.

**Change:** keep ordinary compiler preparation explicit and separate from custom
pipeline parsing. Provide or clearly document an isolated `.mlir` debugging
route. Reuse upstream `PassReproducerOptions::attachResourceParser` and `apply`
if replay is supported. Avoid introducing a second custom resource parser or
`lit`/FileCheck runner.

The debugging page should contain one runnable reduced-input example and state
these boundaries:

- How to build/locate `mqt-cc`, run the chosen pipeline, and inspect its stages.
- Generic printing and before/failure dumps, with stderr distinguished from
  output IR. With the current driver, non-stdout MLIR output is bytecode;
  redirect stdout when a text `.mlir` file is required.
- `--mlir-disable-threading` for readable output and for upstream options that
  require it, including local reproducer generation.
- `--debug-only=dialect-conversion` and the requirement for debug logging in the
  relevant LLVM/MQT code. Build-type names alone do not establish logging
  support; the tested LLVM installation enables assertions.
- How to generate and replay a reproducer, and which driver modes cannot provide
  an isolated run today.

**Risk:** do not remove normal cleanup or required inlining globally merely to
make a debugging example work. Pipeline replacement can discard required stages;
test a reusable-function input before changing QIR scheduling.

### 7. Apply pass-manager options during jeff import

`loadJeffFile` creates its own pass manager at `mqt-cc.cpp:315-321`, without
`applyPassManagerCLOptions`. Later stages use the configured manager helper at
`:556-565`. A small QASM program was exported to jeff and reimported with
`--emit=qco --mlir-print-ir-before-all`. Import succeeds, but stderr is empty.
This mode skips later optimization and isolates the missing initial dump.

**Change:** apply the same upstream CLI options to the initial conversion
manager and propagate application failure. Add a small driver test checking that
the initial jeff conversion is included in a requested dump. Retain the
deserialization, conversion, and linearity checks.

**Risk:** do not apply process-global CLI options automatically inside all
library helpers. This is a driver-owned boundary. Missing IR dumps are executed
evidence; other instrumentation omissions follow from the same missing call, but
each crash/reproducer failure path was not injected separately.

### 8. Remove repeated modifier validation from QC-to-QCO

`validateModifierBodies` at `QCToQCO.cpp:683-694` walks every operation only to
call QC modifier verifiers. The conversion invokes it at `:1993-1996`. Those
rules already belong to QC's `ModifierUtils.cpp:32-60` and its inv/ctrl/pow
verifiers. The documented pass contract permits verified input.

Four conversion test families build malformed modifiers, disable the pass
manager verifier, and assert the duplicate rejection:
`test_qc_to_qco.cpp:1337-1358`, `:1401-1428`, `:1500-1537`, and `:1582-1614`.
They cover structured operations, register loads, CBit operations, and qubit
captures. Owning QC tests cover overlapping classes at
`test_qc_ir.cpp:1110-1155` and `:1191-1217`.

The [experiment][modifier-patch] removes the repeated walk and changes those
four call sites to invoke `mlir::verify` directly, retaining their original
input matrices and diagnostic assertions. All 367 QC tests and 178 QC-to-QCO
tests pass. This removes 14 production lines and 12 test setup lines, plus one
whole-IR traversal. No runtime percentage is claimed.

**Change:** place these negative cases under the owning QC verifier tests,
merging overlapping inputs only after checking their distinctions. Preserve SCF
shapes, nested captures, and register-backed cases; historical regression
coverage from PR #1987 must survive. Keep converter checks for unsupported valid
IR, including wire correspondence, aliasing, and register limitations.

**Risk:** public borrowed modules are mutable, so invalid input is
constructible. The simplification rests on the declared verified-pass-input
contract, not an assumption that all callers are incapable of producing
malformed IR. It does not remove validation from parsing or public program
adoption.

### 9. Start the positional-result rejection test with valid QCO

`QCOToQCRegressionTest.RejectsMissingPositionalQubitResults` at
`test_qco_to_qc.cpp:434-448` leaves its qubit argument unused. It therefore
violates QCO exactly-one-use before testing the converter's result restriction.
It only asserts failure, without identifying the reason.

The [corrected fixture experiment][valid-input-patch] sinks the qubit, verifies
ordinary IR and linearity, and checks the rejection substring
`must return one trailing qubit for each qubit argument`. All 153 QCO-to-QC
tests pass. This distinguishes a valid consumed-qubit function from the
converter's narrower supported subset.

**Change:** retain this valid input and reason check. There is no production
deletion associated with this finding, and no need to widen conversion support.

### 10. Exercise diagnostic routing after handler teardown

`test_target_compilation_preserves_diagnostics` in `test_mlir.py:571-599`
correctly checks a target-compilation error. Its subsequent successful
compilation, described as proving handler restoration, emits no diagnostic.

After the first caught error, call `valid.run_pass_pipeline("not-a-pass")` on
the context-sharing copy, expect `RuntimeError`, and capture stderr. Check
`failed to parse pass pipeline`, emitted by `mod.emitError` in
`mlir/lib/Support/Passes.cpp:125-127`; checking only the LLVM parser's text
would not establish context-handler routing. The executed negative action
produces that diagnostic with the current implementation.

**Change:** add this negative action to the existing test. The production scoped
handler is correct in this probe. Do not add a new abstraction or claim that a
handler lifetime defect was found.

## Retained behavior and unresolved candidates

- Keep the existing CLI subprocess tests. QIR filename/encoding selection,
  target payload precedence, driver parsing, and driver linearity are genuine
  process-boundary contracts. Library tests do not replace them. The audit found
  no reason to introduce `lit` or FileCheck.
- Keep exact-phase, borrowed-qubit restoration, numerical-boundary, gate-count,
  and determinism assertions where they guard named contracts. Do not merge
  numerical matrices merely because some cases share code coverage.
- Keep OpenQASM source/include ownership coverage: the existing parser test
  destroys the supplied source manager before analysis. Extra lifetime tests of
  the same path would duplicate that protection.
- Keep failed-export buffering and valid-input unsupported-subset checks. Failed
  in-place target compilation does not imply rollback, while rejection before
  consumption is a distinct supported boundary.
- `DeferredPrinter::record` eagerly renders and copies snapshots on passing
  tests (`TestCaseUtils.h:154-166`). An opt-in-only printer could remove that
  work but would give up automatic before-state snapshots on failure. No
  measured benefit justifies that tradeoff here; it remains a candidate.
- Generic wrapper diagnostics after parser/frontend failures may duplicate
  already detailed errors (`Programs.cpp:93-96`, `:238-259`). Removing them
  could reduce noise, but their stage context may be useful. Uniform Python
  exception payloads are not a documented universal contract. These are design
  decisions, not confirmed deletions.
- The CLI QIR test's `; ModuleID` prefix check is more specific than valid
  textual LLVM IR. Its `llvm-as` checks may suffice, but a replacement must
  still distinguish bitcode from text. No change is recommended without that
  format probe.
- Moving the driver's test-only default-build override into test CMake may
  improve ownership. There is no demonstrated build-time saving, and the tool
  must still be built for the retained CLI tests.

## Validation and reproduction

The local environment used macOS ARM64, AppleClang 21, CMake 4.4.3, LLVM/MLIR
23.1.0 with assertions enabled, Release configuration, Python 3.13.7, nanobind
3.0.1, and Qiskit 2.5.2. jeff-mlir was pinned to
`eec13f1f41da0da03ad55090adce4c6b4afecd21`; jeff to `3bf34d2` and QDMI to
`7cf3dfd3a2337d0156dcc739cb96bf760828f0ec`.

Use an isolated checkout of the baseline, a compatible LLVM/MLIR installation,
and the repository's normal build commands:

```sh
cmake --preset release -DMLIR_DIR="$MLIR_DIR" -DBUILD_MQT_CORE_TESTS=ON
cmake --build --preset release -j 8
ctest --preset release -j 8
```

For Python, use the documented development installation and run:

```sh
uv run --no-sync pytest test/python/test_mlir.py \
  test/python/test_mlir_loops.py test/python/test_mlir_integer_interchange.py \
  test/python/test_mlir_qiskit_translation.py -n 4
```

This audit instead used an isolated interpreter with cached dependencies and the
exact rebuilt extensions, pure Python source, and bundled QDMI runtime files
staged under the build directory. The loaded extension path was checked. It did
not test an unrelated previously installed Core binary. Other operating systems,
Python/Qiskit versions, sanitizers, coverage deltas, and hosted CI were not run.

After restoring every experimental production and test edit, the full build and
all 3,454 CTest entries were rerun successfully, with the same intentional QDMI
query skip. The contribution contains only the audit report and its evidence
directory.

`uvx nox -s lint` passes. The audit files were also checked before staging
explicitly with `uvx prek run --files ...`; those hooks pass after formatting.
All five retained patches pass `git apply --check` against the restored
baseline. No C++ source change remains for `cpp-lint`, and no binding change
remains for stub regeneration.

The [reproduction instructions][reproduction] give exact patch/probe commands.
[Results][results] retain relevant output, with machine paths normalized.
Experimental patches are evidence, not proposed complete fixes. Restore them and
rebuild between experiments. No test or production change from the probes is
part of this audit's final working-tree changes.

[issue]: https://github.com/munich-quantum-toolkit/core/issues/2254
[parent]: https://github.com/munich-quantum-toolkit/core/issues/2250
[oracle-patch]: mlir-tests-diagnostics/oracle-and-lifetime.patch
[lifetime-patch]: mlir-tests-diagnostics/lifetime-order.patch
[modifier-patch]: mlir-tests-diagnostics/modifier-ownership.patch
[valid-input-patch]: mlir-tests-diagnostics/valid-conversion-input.patch
[attribute-patch]: mlir-tests-diagnostics/attribute-comparison.patch
[reproduction]: mlir-tests-diagnostics/README.md
[results]: mlir-tests-diagnostics/results.txt
[llvm-diagnostics]: https://github.com/llvm/llvm-project/blob/llvmorg-23.1.0/mlir/lib/IR/Diagnostics.cpp
[llvm-pass-registry]: https://github.com/llvm/llvm-project/blob/llvmorg-23.1.0/mlir/lib/Pass/PassRegistry.cpp

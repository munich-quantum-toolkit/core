# Contract audit: OpenQASM import and export

Status: complete; accepted findings implemented and locally validated. Audit
baseline: `33dbc843e589d9e9166308084e825c9f8b2ff89d`, initially clean checkout.
Date: 2026-09-08. Environment: native ARM64, release build, LLVM/MLIR 23.1.0.

## Implementation disposition

The implementation preserves affine quantum indexing. Static-only quantum
indexing was rejected. The accepted changes are:

- Bound affine reconstruction per proof and use set-based static distinctness.
- Replace projected emission costs with actual insertion accounting and bounded
  cancellation. Keep semantic target restrictions at their owning lowering.
- Preserve ordered outputs and zero-valued data; use void for absent outputs and
  diagnose repeated register outputs instead of silently dropping aliases.
- Share frontend name validation; align floating inequality with unordered
  semantics; avoid storage for unused measurement results.
- Build syntax IDs directly, use one bit-vector comparison representation, and
  materialize entry-function snapshots at their definition.
- Require full-register initialization for dynamic reads. Static initialization
  tracking and affine scalar generations remain.
- Accept exact-width bit strings and float casts; use compact zero initializers.
  Document the existing working-directory/include-directory lookup policy.
- Lower catalog definitions through strict mode for matrix tests, including
  controlled forms and exact global phase. These tests exposed native `qc.u` and
  `qc.u2` export phase errors. One shared helper applies `gphase(-theta/2)`
  before OpenQASM 3 `U`, retaining the native matrix under controls.

Local parser measurements on 100,000 distinct declarations reduced peak RSS from
about 137,600 KiB to 88,000 KiB. Three-run median wall time was 0.66 s versus
0.45 s; other host workloads make timing noisy. A 20-assignment affine DAG took
17.8 s before and 0.00072 s after under host contention. A 64,000-operand static
control probe took 7.21 s before and 0.145 s after. These observations establish
local scaling improvements, not cross-machine timing promises.

The release build and all 3,215 MLIR CTests pass, including 190 OpenQASM target
and 209 QC translation tests. Repository lint and full-file C++ lint pass.
Hosted CI is separate from these local results.

The sections below retain the original audit evidence and proposals, whose
source line numbers refer to the audit baseline. The disposition above records
which proposals were implemented.

## Result

The most useful work is to bound affine analysis, remove a stale emission-cost
assumption, and repair exporter result and expression contracts. The larger
deletions are possible only after choosing explicit internal representations.
The current safety checks are not generally redundant.

| Rank | Finding                                                         | Impact                                        | Confidence                                       |
| ---- | --------------------------------------------------------------- | --------------------------------------------- | ------------------------------------------------ |
| 1    | Repeated scalar assignments cause exponential affine analysis   | Small input can stall compilation             | Confirmed by timing and source                   |
| 2    | Dynamic writes still pay for an obsolete register-wide lowering | Valid small programs fail the emission limit  | Confirmed reproducer                             |
| 3    | Export loses output order, multiplicity, and zero-valued data   | Changes the result interface                  | Confirmed reproducers; status rule is documented |
| 4    | Floating inequality uses inconsistent predicates                | Valid imported comparison cannot be exported  | Confirmed reproducer and predicate definitions   |
| 5    | Exporter name validation duplicates an incomplete keyword list  | Successful export produces rejected source    | Confirmed reproducers                            |
| 6    | Unused measurements allocate observable classical storage       | Reimport introduces an implicit output        | Confirmed reproducer                             |
| 7    | Static gate-operand distinctness uses pairwise checks           | Quadratic analysis for large controlled gates | Confirmed timing and source                      |

The source scope covers the lexer, parser, syntax storage, semantic analysis,
typed frontend, QC emitter, QC builder, exporter, gate catalog, relevant driver
entry points, documentation, and both OpenQASM/QC translation test targets. This
is an investigation of the current implementation, not a claim of full OpenQASM
language conformance or exhaustive downstream interoperability.

## Confirmed findings

### 1. Store affine facts without repeatedly expanding their expression DAG

**Change and benefit.** In `mlir/lib/Target/OpenQASM/OpenQASMSemantics.cpp`,
replace repeated recursive reconstruction in `buildAffineForm` (line 633) and
`expandAffineScalarValues` (line 710) with bounded analysis of shared
expressions. Prefer retaining normalized affine facts at assignment time. At
minimum, avoid revisiting a shared expression within one proof and bound proof
work.

**Evidence.** Analyze `OPENQASM 3.1; int a = 1;` followed by N copies of
`a = a + a;`. No quantum indexing is needed to trigger the cost.

| N   | Typed expressions | Parse and analysis time |
| --- | ----------------- | ----------------------- |
| 8   | 33                | 0.00161 s               |
| 12  | 49                | 0.01886 s               |
| 16  | 65                | 0.29367 s               |
| 20  | 81                | 2.339 s                 |
| 24  | —                 | Exceeded a 5 s timeout  |

These are local observations, not cross-machine performance promises. The
expression arena grows modestly; recursively following both operands repeatedly
revisits the same DAG. The parser's expression-depth bound does not bound this
work across assignments.

**Contract and limits.** Preserve proof of intermediate integer operations and
overflow, induction domains, scope, and branch invalidation. A global cache
keyed only by expression ID would be unsafe because the proof context changes.
When an optional scalar fact exceeds a budget, forget the fact; reject a quantum
index only when its required proof then fails. Do not reject ordinary classical
arithmetic merely because an optional affine optimization is expensive.

### 2. Remove the obsolete register-width charge for dynamic stores

**Change and benefit.** In
`mlir/lib/Dialect/QC/Translation/OpenQASMToQCEmitter.cpp`, the
projected-emission checks at lines 781–788 and 830–834 charge `9 + 3 * width`
for a dynamic bit store. `assignBit` at line 2251 now emits a checked index and
a native `cbit.store`. Correct both ordinary-assignment and measurement-target
accounting.

**Evidence.** This accepted-width source fails with the projected 10-million-op
limit around line 35, although only 40 stores are requested:

```python
source = "OPENQASM 3.1;\nbit[99999] c;\noutput int i;\ni = 0;\n"
source += "c[i] = false;\n" * 40
```

Using width 10 instead succeeds and emits 40 `cbit.store` operations. The
explicit scalar output keeps the partially initialized register from becoming an
implicit output, isolating the emission-budget problem.

**Larger simplification.** The roughly 600-line predictive cost implementation
duplicates lowering details while `EmissionBudget`, an `OpBuilder` listener,
already counts actual emitted operations. This is a concrete maintenance cost:
the two have drifted. Prefer one authoritative actual-emission budget if
emission can abort promptly. Do not delete the preflight implementation
unchanged: the listener currently marks exhaustion, and recursive expression
construction and bulk statement paths need cancellation propagation. Preserve
all resource bounds and rejection tests; replace predicted-count assertions with
bounded-emission checks where the distinction is not public behavior.

### 3. Represent outputs as ordered results, with explicit status semantics

**Change and benefit.** In
`mlir/lib/Dialect/QC/Translation/TranslateQCToOpenQASM3.cpp`,
`collectProgramShape` (line 382) separates returned registers into a set and
scalars into a vector. `emitDeclarations` (line 515) prints registers in
allocation order before scalar outputs. That representation cannot preserve the
entry function's ordered result list or repeated register results.

**Evidence.** Direct import/export of:

```qasm
OPENQASM 3.1;
output int a;
output bit b;
a = 1;
b = false;
```

prints `output bit[1] b;` before `output int _mqt_out0;`. Reimport changes the
ordered output types. A verified function returning `%c, %c` as two
`!cbit.reg<1>` results exports only one output declaration. Independently,
`output int answer; answer = 0;` loses its sole output through the CLI, while
the same program with value 1 retains an output.

The last behavior follows `isCanonicalStatus` (line 480), which recognizes any
lone constant-zero `i64` result. `docs/mlir/OpenQASM.md` explicitly documents
it; this is a brittle documented convention, not an undocumented regression.

**Contract and limits.** Use an ordered result description and choose an
explicit policy for aliases: preserve each result with a distinct output or
reject repeated register results. Distinguish status from data through an
explicit contract; `QCProgramBuilder` already has empty result-type/value
overloads, but changing the default status ABI requires checking all builder and
compiler consumers. A documented restriction on mixed or aliased outputs would
be a smaller acceptable change than silently changing their shape. Scalar names
and fixed-angle spelling are already documented as lossy and are separate from
losing actual results.

### 4. Align floating inequality with the imported predicate

**Change and benefit.** The importer maps `!=` to `arith.cmpf UNE` in
`OpenQASMToQCEmitter.cpp::emitComparison` (line 1919). The exporter maps `ONE`
to `!=` and rejects `UNE` in `TranslateQCToOpenQASM3.cpp::floatPredicate` (line
1227). Map `UNE` consistently; reject `ONE` unless its ordered semantics can be
preserved or non-NaN operands are established.

**Evidence.** Strict frontend analysis accepts this source, but direct
import/export fails with an unsupported floating-comparison diagnostic:

```qasm
OPENQASM 3.1;
float f = 1.0;
output bool b;
b = f != 2.0;
```

`ONE` and `UNE` differ for NaNs according to the
[MLIR comparison definitions](https://mlir.llvm.org/docs/Dialects/ArithOps/#arithcmpf-arithcmpfop).
Checking finite literals does not establish finite runtime computations. The
existing `EmitsSignedAndFloatingComparisonFamilies` string check accepts the
wrong `ONE` mapping. Replace that expectation with predicate-sensitive coverage.
The finite-value roundtrip failure is reproduced; NaN behavior here is
established from predicate semantics, not from a hardware execution experiment.

### 5. Share source-name validation with the frontend

**Change and benefit.** Exporter `isReserved` (line 111) duplicates only part of
the lexer keyword and semantic builtin-constant policy. Centralize that policy
for names that will be emitted as source. Keep exporter-specific
generated-prefix and gate-collision rules separate.

**Evidence.** Export succeeds for the following verified IR, but strict analysis
of its output fails:

```mlir
module {
  func.func @main() -> !cbit.reg<1> attributes {mqt.entry_point} {
    %c = cbit.alloc(#cbit.init<zero>) {mqt.register_name = "void"}
      : !cbit.reg<1>
    return %c : !cbit.reg<1>
  }
}
```

The same happens for `im`, `pragma`, and `pi`. `sin` and `valid_name` pass,
providing controls against an overly broad blacklist. Compare
`OpenQASMLexer.cpp::keywordKind` and `OpenQASMSemantics.cpp::builtinConstant`.
Preserve deterministic renaming and catalog collisions; accepting every Unicode
source identifier is not necessary to fix this inconsistency.

### 6. Emit unused measurements without a scratch bit

**Change and benefit.** `TranslateQCToOpenQASM3.cpp::emitMeasurement` (line
1315) allocates a scratch bit for an unfused measurement even when its classical
result is unused. Emit `measure q;` for an unused result.

**Evidence.** Direct import/export of `OPENQASM 3.1; qubit q; measure q;`
produces `bit _mqt_b0;` and `_mqt_b0 = measure _mqt_q0;`. Reimport treats the
new global bit as an implicit output. The original has no classical outputs. The
quantum measurement and its position must survive; only its unused classical
storage is removable. Keep scratch storage for values with consumers.

### 7. Reuse static deduplication before affine distinctness proofs

**Change and benefit.** `OpenQASMSemantics.cpp` gate-application validation
(line 4955) compares every operand with every previous operand, including static
indices. `analyzeBarrier` already demonstrates set-based static deduplication.
Reuse that pattern, partitioned by reference kind and register identity; reserve
pairwise affine proofs for cases that need them.

**Evidence.** Analyze `qubit[N] q; ctrl(N-1) @ x q[0], ..., q[N-1];` after the
version declaration. Median parse-and-analysis times over three local runs:

| N      | Time       |
| ------ | ---------- |
| 8,000  | 0.028016 s |
| 16,000 | 0.105600 s |
| 32,000 | 0.705165 s |
| 64,000 | 5.494332 s |

The affine solver's comparison cap does not limit the cheap static pair count.
Keep rejection of duplicate targets/controls, hardware-qubit identity, and
uncertain affine overlap. No revised implementation was benchmarked.

## Structural simplifications and explicit limits

These are proposals, not measured line savings or applied changes.

### Collapse the special register-comparison representation

`ConditionKind::RegisterComparison` and `BitVectorComparison` now converge on
whole-register reads, constants, and integer comparisons in `emitCondition`
(lines 1970–1994). Convert register-versus-literal comparisons to the existing
bit-vector expression representation during analysis, then remove the redundant
typed fields, emitter branch, and cost-model branch.

Preserve `OpenQASMSemantics.cpp` lines 4567–4630: comparisons against
out-of-range literals can fold without truncation, and swapping literal/register
sides changes relational predicates. Initialization diagnostics must also
survive. The typed frontend is publicly declared, so no in-tree external user is
insufficient to prove source compatibility. This is a narrow consolidation, not
deletion of the comparison semantics.

### Make the parser concrete; consider building syntax IDs directly

`OpenQASMParser.h` defines a `QASMSink` concept and a large parser template, but
`Frontend.cpp` instantiates it only with `SyntaxBuilder`. Removing that unused
internal extension point would simplify the interface. A second, larger change
could construct syntax expression IDs directly:
`OpenQASMSyntax.cpp::copyExpression` (lines 83–122) currently traverses
transient expression objects into an ID arena using a worklist and map while the
transient allocator remains alive during parsing.

Do not merge parsing with semantic analysis. Reusable `ParsedProgram` analysis,
different gate policies, source-buffer lifetime, diagnostics, and iterative
handling of long expression chains are useful contracts. Removing the transient
tree needs a peak-memory and parse-time comparison before claiming a speedup.

### Prefer explicit scalar snapshots to recursive exporter reconstruction

The exporter tracks operation positions and writes to reject stale or
cross-region classical snapshots, and caps recursive expression expansion at
4,096 nodes and depth 256. These checks protect correctness and bounded output.
A more explicit entry-function export normal form could materialize reads at
their definition and shared scalar expressions once, using the existing
`materialize` machinery.

This could remove much of the need to reconstruct temporal safety and could
export presently rejected programs. It is not safe to delete snapshot checks
first. Preserve evaluation order, simultaneous SCF edge assignments, and loop
semantics. Gate bodies have stricter allowed statements, and registers wider
than 64 bits need a different snapshot representation. Start with entry-function
scalar values. Replace stale-snapshot and expansion rejection tests only when
semantic snapshot and bounded-output tests cover the new implementation.

### State partial-initialization support before simplifying its analysis

Dynamic bit facts, expression equivalence, dependencies, and generation tracking
allow a partially initialized register to support `c[i] = ...; read c[i]`
without requiring every bit to be initialized. Requiring full initialization
before every dynamic read could delete substantial machinery, but would remove
tested behavior. That is a product decision, not redundant-code cleanup. Keep
the copy-on-write initialization state used across branches.

Similarly, requiring constant quantum indices would remove affine analysis but
also useful documented loops. The
[OpenQASM type specification](https://openqasm.com/language/types.html) permits
implementations to restrict nonconstant quantum indexing; permission to restrict
it does not make this project's existing feature dispensable. Bound the current
proof work before considering that larger reduction.

### Compact zero initialization within the supported expression profile

`emitDeclarations` writes one assignment per bit for zero-initialized registers.
A width-10,000 probe produced 10,005 lines and 248,959 bytes. For widths up to
64, the frontend accepts a whole-register assignment such as
`c = bit[64](uint[64](0));`, which can replace the per-bit loop.

Do not assume that an arbitrary-width bit string is an available replacement:
`bit[3] c = "000";` fails in the current parser. Exact-width bit strings are
described by the language and mentioned by the local documentation, but
`parsePrimary` does not accept them. Supporting them is a separate frontend
change. Keep wide-register initialization correct until that support exists.

### Make the remaining syntax and include boundaries explicit

The exporter can emit `float(int[64](1))`, while parser `parsePrimary` does not
accept the float cast. This reproduces a reimport failure, but float/integer
conversions are explicitly excluded from the documented roundtrip subset. Either
close the parser gap through the existing cast representation or make the
earlier broad statement that emitted expressions are accepted more precise.

CLI relative include lookup depends on the process working directory rather than
automatically using the source file's directory. A source beside `repeated.inc`
failed to find it when invoked from another directory. `openSourceMgr` in the
driver and `SourceMgr::OpenIncludeFile` in the frontend own this boundary.
Document the search policy or provide the source directory explicitly. The
[OpenQASM include rules](https://openqasm.com/language/comments.html) describe
source inclusion; they do not by themselves establish a filesystem search order.
This is a usability/contract choice, not a claimed language violation.

## Test coverage and rejected deletions

`EmitsCatalogHelpersUnderTheirNativeNames` in `test_openqasm3_emission.cpp`
(around line 658) runs strict analysis, then translates through the default
compatibility policy. Compatibility mode deliberately ignores a matching catalog
definition body and emits the native operation. The test therefore does not
prove that exported helper bodies implement those native operations.

Add strict-policy lowering and compare helper unitaries, including controlled
global phase. Reuse the existing matrix oracle in `test_qasm3_translation.cpp`
where suitable. That oracle is independent reference behavior, not removable
duplication of production lowering. No incorrect helper matrix was demonstrated
by this audit.

Other tempting deletions are rejected:

- Exact fixed-angle quantization protects widths, wrapping, and ties-to-even;
  ordinary floating rounding is not an equivalent replacement.
- Runtime integer-power overflow checks protect a distinct documented contract;
  modular addition semantics do not justify removing them.
- Resource limits, snapshot checks, uncertain-alias rejection, and
  initialization checks are necessary until an alternative representation
  enforces the same guarantees.
- Equal emitted strings or overlapping coverage do not prove tests redundant. No
  broad test deletion is justified by this investigation.

## Related work

Live GitHub inspection found relevant boundaries in issues
[#2414](https://github.com/munich-quantum-toolkit/core/issues/2414) (naming),
[#2413](https://github.com/munich-quantum-toolkit/core/issues/2413) (gate
bodies), and
[#2427](https://github.com/munich-quantum-toolkit/core/issues/2427),
[#2428](https://github.com/munich-quantum-toolkit/core/issues/2428), and
[#2429](https://github.com/munich-quantum-toolkit/core/issues/2429) (register
subroutines). Coordinate changes to naming or function/result contracts with
that work. Runtime angles, general arrays, and slices remain separate feature
scopes. No issues, pull requests, or remote branches were changed.

## Validation and reproduction

Built successfully, exit 0:

```sh
cmake --build --preset release --target \
  mqt-core-mlir-unittest-openqasm-target \
  mqt-core-mlir-unittest-qc-translation mqt-cc -j8
```

Ran both binaries directly and recorded their actual exit statuses:

```sh
build/release/mlir/unittests/Target/OpenQASM/mqt-core-mlir-unittest-openqasm-target
build/release/mlir/unittests/Dialect/QC/Translation/mqt-core-mlir-unittest-qc-translation
```

Results: 185/185 and 199/199 passed, each exit 0. These baseline tests do not
disprove the reproduced gaps. Logs are in `/tmp/openqasm-audit-target.log` and
`/tmp/openqasm-audit-translation.log` for this session.

`uvx nox -s lint` passed. The explicit report check,
`uvx prek run --files .agent/audits/openqasm-import-export.md`, also passed.

The temporary C++ probe and input files are under `/tmp/mqt-openqasm-audit`. The
probe uses the built libraries and these API sequences:

- `analyze` / `strict`: load a `SourceMgr`, call `parseOpenQASM`, then
  `analyzeOpenQASM` with `MQTCompatibility` / `Strict`; report diagnostic
  failure as exit 1 and success as exit 0. Timings include parsing and analysis.
- `import`: call `translateQASM3ToQC` and print the resulting module.
- `roundtrip`: call `translateQASM3ToQC`, then `translateQCToOpenQASM3`
  directly, without cleanup passes that could fold away the tested operation.
- `export`: parse the input MLIR and call `translateQCToOpenQASM3`.

Example commands with the retained inputs:

```sh
/tmp/mqt-openqasm-audit/probe strict /tmp/mqt-openqasm-audit/float-ne-input.qasm
/tmp/mqt-openqasm-audit/probe roundtrip /tmp/mqt-openqasm-audit/float-ne-input.qasm
/tmp/mqt-openqasm-audit/probe export /tmp/mqt-openqasm-audit/name-void.mlir
/tmp/mqt-openqasm-audit/probe analyze /tmp/mqt-openqasm-audit/affine-dag-20.qasm
```

The first command succeeds; the second fails; the third succeeds but produces
source that strict analysis rejects; the fourth succeeds after excessive work.
The source fragments and generators above preserve the essential reproductions
without depending on temporary files surviving.

No production mutation, full-project test run, downstream simulator execution,
hosted CI run, or benchmark of a proposed replacement was performed. This report
does not claim a measured net line reduction or dependency removal. Implement
the small reproduced corrections first; evaluate the parser, budgeting, and
snapshot representations as separate changes with their retained contracts.

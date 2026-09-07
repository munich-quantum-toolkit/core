# Contract audit: MLIR lowering and import boundaries

Status: complete; five confirmed findings proposed for focused fixes. Baseline:
`6cc98f059b73ebb3ca4d39fbd318b4cacffa31c8` (`main`). Date: 2026-09-07. No
production or test changes accompany this report. Dependencies: LLVM/MLIR
23.1.0; jeff-mlir `4732c4f12047e8cbf1890c79e90e30110e6588e2`.

## Result

1. **High impact, reproduced:** QIR Base lowering moves a gate before an earlier
   measurement of the same qubit, changing the recorded result.
2. **High impact, reproduced:** QC-to-QCO conversion aborts on valid branches
   that use the same dominating qubit.
3. **High impact, reproduced:** a malformed JeFF file aborts the importing
   process instead of returning an import error.
4. **Medium impact, reproduced:** QIR resource metadata declares too few qubits
   when a valid program uses sparse static IDs.
5. **Medium impact, reproduced:** Adaptive QIR lowering moves a branch-local
   qubit release outside the allocation's scope and fails verification.

The review revisited the prior audit's compiler, conversion, verifier, modifier,
mapping, numeric, resource, and metadata claims against current code and related
pull requests. Only these five have both a concrete failure and sufficient
impact for this report. This is not a claim that the other subsystems are free
of defects.

Passes may assume verified input, as required by `docs/mlir/development.md`.
Operation verifiers own IR invariants; importers and conversions own external
input and target restrictions. Each finding below belongs to one such boundary.

## Findings

### 1. Reject QIR Base lowering that changes measurement order

**Change and benefit.** Diagnose valid source programs whose measurement order
cannot be represented by the Base profile. Check this in the Base conversion,
before moving measurements. This prevents silently changing program results.

**Contract and source.** In
`mlir/lib/Conversion/QCToQIR/QIRBase/QCToQIRBase.cpp`,
`ConvertQCMeasureOp::matchAndRewrite` emits every measurement in
`state.measurementsBlock`. `QCToQIRBase::ensureBlocks` places that block after
the body containing the gates. No check prevents a later gate on the measured
qubit from moving before its measurement. The
[QIR Base profile](https://github.com/qir-alliance/qir-spec/blob/main/specification/profiles/Base_Profile.md#qubit-and-result-usage)
forbids using a qubit after an irreversible operation; satisfying that output
restriction must preserve the source semantics or fail conversion.

**Reproducer.** Save this as `measure-before-x.qasm`:

```qasm
OPENQASM 3.0;
include "stdgates.inc";
qubit q;
bit c;
c = measure q;
x q;
```

```sh
build/release/mlir/tools/mqt-cc/mqt-cc \
  --emit=qir-base -o - measure-before-x.qasm
```

Observed: exit 0; the QIR calls `__quantum__qis__x__body` before
`__quantum__qis__mz__body`, then records the measurement. The source records 0
from the initially zero qubit; the emitted instruction sequence records 1. This
conclusion follows from the emitted operations, not a QIR execution test.

**Limits and disposition.** Proposed. Preserve support for programs whose
measurements can legally move past independent operations. A fix needs the
same-qubit regression and a valid independent-qubit case; it does not need
whole-module or QCO-linearity checks at every pass entry.

### 2. Diagnose unsupported QC control flow without aborting

**Change and benefit.** Make QC-to-QCO conversion diagnose control-flow shapes
that its qubit-state mapping cannot represent. If general CFG support is
required, propagate quantum values along each edge instead. Either approach must
avoid aborting the compiler on valid QC input.

**Contract and source.** QC uses reference semantics, so mutually exclusive
blocks may use the same dominating qubit. In
`mlir/lib/Conversion/QCToQCO/QCToQCO.cpp`, `lookupMappedQubit` assumes that the
region-local state still contains every referenced qubit. Converting the first
returning branch exhausts this state; conversion of the other branch asserts.
This violates the valid-input pass contract in `docs/mlir/development.md`.

**Reproducer.** Save this as `qc-branches.mlir`:

```mlir
module {
  func.func @main() attributes {mqt.entry_point} {
    %q = qc.static 0 : !qc.qubit
    %c = arith.constant true
    cf.cond_br %c, ^then, ^else
  ^then:
    qc.x %q : !qc.qubit
    return
  ^else:
    qc.z %q : !qc.qubit
    return
  }
}
```

```sh
build/release/mlir/tools/mqt-cc/mqt-cc \
  --input-format=mlir --emit=qco -o - qc-branches.mlir
```

Observed: exit 134 and the assertion `QC qubit not found` in
`lookupMappedQubit`. Input parsing and verification succeed before conversion.

**Limits and disposition.** Proposed. This reproduces through public MLIR
import, not the OpenQASM frontend. The smallest fix may reject this supported
input dialect's unsupported conversion shape; full CFG support is a separate
scope decision. Do not add generic verification to every pass.

### 3. Return JeFF import errors without aborting the host process

**Change and benefit.** Make the shared JeFF deserializer return a diagnostic
and failure for malformed input. Prefer fixing the pinned upstream dependency;
propagate its failure through the public compiler import methods and CLI. This
allows an application to reject a bad file without terminating.

**Contract and source.** `JeffProgram::fromBytes` and `JeffProgram::fromFile` in
`mlir/lib/Compiler/Pipeline.cpp` expose fallible imports but call the
deserializer directly. `loadJeffFile` in `mlir/tools/mqt-cc/mqt-cc.cpp` uses the
same dependency. In the pinned jeff-mlir `lib/Translation/Deserialize.cpp`,
`deserialize` calls `llvm::report_fatal_error("No functions found in module")`
when the serialized module lacks a functions list. This is an external file
boundary, before there is verified MLIR input.

**Reproducer.** The following bytes encode a Cap'n Proto JeFF 0.3 module with
the functions list absent:

```sh
printf '%s' \
  'AAAAAAgAAAAAAAAAAgAFAAAAAAAAAAAAAwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA' \
  | base64 --decode > missing-functions.jeff
build/release/mlir/tools/mqt-cc/mqt-cc \
  --input-format=jeff --emit=qco -o - missing-functions.jeff
```

Observed: exit 134 (`SIGABRT`) and `LLVM ERROR: No functions found in module`.
The stack reaches `deserializeFromFile` from `loadJeffFile`. The CLI failure was
executed; the byte/file library methods' use of the same fatal dependency was
checked in source, not in a separate host-process experiment.

**Limits and disposition.** Proposed, preferably upstream. Keep a small import
regression once failure is recoverable. This finding does not justify the old
schema-wide preflight implementation, a process-global fatal-error bridge, or
validation in unrelated transformation passes.

### 4. Cover sparse static IDs in QIR resource capacity

**Change and benefit.** Compute required static qubit capacity from the largest
referenced ID plus one, rather than the number of distinct IDs. This produces
metadata that consumers can use to size resources and validate the program.

**Contract and source.** `qc.static` supports nonnegative static indices,
including hardware-mapped indices. In
`mlir/lib/Dialect/QIR/Transforms/AttachQIRAttributes.cpp`, `getNumQubits` counts
distinct static pointers. That is insufficient for a sparse index set. The
[QIR Base profile's data-type contract](https://github.com/qir-alliance/qir-spec/blob/main/specification/profiles/Base_Profile.md#data-types-and-values)
requires static IDs to lie below the declared resource count.

**Reproducer.** Save this as `sparse-static.mlir`:

```mlir
module {
  func.func @main() attributes {mqt.entry_point} {
    %q = qc.static 7 : !qc.qubit
    qc.x %q : !qc.qubit
    return
  }
}
```

```sh
build/release/mlir/tools/mqt-cc/mqt-cc \
  --input-format=mlir --emit=qir-base -o - sparse-static.mlir
```

Observed: exit 0; `x` receives `inttoptr (i64 7 to ptr)`, but the entry point
declares `required_num_qubits="1"`. Preserving ID 7 requires a capacity of at
least 8. The emitted program therefore violates the target contract.

**Limits and disposition.** Proposed. Preserve physical-qubit identity; do not
silently renumber mapped qubits. The regression should check capacity and the
retained ID. Sparse result IDs and other pointer-provenance cases were not
separately reproduced and are not additional findings here.

### 5. Keep Adaptive QIR releases on the allocation's control-flow path

**Change and benefit.** Lower a dynamic qubit release where its allocation
dominates it and only on paths where that allocation ran. This permits
branch-local dynamic allocation to compile to valid Adaptive QIR.

**Contract and source.** `QCToQIRAdaptive.td` describes SCF lowering and the
translation of QC operations to equivalent QIR operations. In
`mlir/lib/Conversion/QCToQIR/QIRAdaptive/QCToQIRAdaptive.cpp`,
`ConvertQCDeallocOp::matchAndRewrite` instead inserts every release in
`state.outputBlock`, even if the allocation exists only inside a branch.

**Reproducer.** Save this as `conditional-allocation.mlir`:

```mlir
module {
  func.func private @condition() -> i1
  func.func @main() attributes {mqt.entry_point} {
    %c = func.call @condition() : () -> i1
    scf.if %c {
      %q = qco.alloc : !qco.qubit
      %out = qco.x %q : !qco.qubit -> !qco.qubit
      qco.sink %out : !qco.qubit
    }
    return
  }
}
```

```sh
build/release/mlir/tools/mqt-cc/mqt-cc \
  --input-format=mlir --emit=qir-adaptive -o - conditional-allocation.mlir
```

Observed: exit 1 with `operand #0 does not dominate this use` at the lowered
sink. The input satisfies QCO linearity; the release moved into the epilogue
uses an allocation from only one branch.

**Limits and disposition.** Proposed. The post-pass verifier catches the
failure, so this is a valid-program compilation failure, not an observed runtime
error. Add a branch-local allocation regression. Keep static-qubit release
policy, discussed in closed #2321, outside this fix.

## Reconciliation and exclusions

- Merged work is already in `main`: #2290, #2291, #2293, #2294, #2295, #2296,
  #2300, #2301, #2302, #2304, #2307, #2308, #2320, and #2322. Their old branch
  copies and tests are removed from this audit's diff.
- #2303, #2305, #2306, #2309, #2318, and #2321 are closed without merge. Their
  rejected or superseded changes are not revived. #2319 remains under review;
  its disputed pass-local effect guards and invalid modifier fixture do not
  establish an additional finding here.
- Repeated-register-load aliasing did not reproduce through the current public
  QIR pipeline: QC-to-QCO-to-QC conversion canonicalized the repeated
  references.
- Generic pass-entry verification, invalid-QCO tests, arbitrary nesting caps,
  manual replacements for MLIR walks, and blanket failure-atomicity requirements
  are excluded. No supported-path need was established for those additions.
- Wide dense-matrix and loop-expansion limits remain candidates for a separate
  bounded investigation. Source-level growth alone does not establish a useful
  project-wide cap. No new cap or claimed practical performance failure is part
  of this report.

The old implementation and audit history remain recoverable from branch commit
`cb9fdea03eefa73e97a2512220119c3496723658`. They are historical leads, not
current findings or an implementation proposal.

## Validation

The compiler and its unit-test target were built with the release preset and
LLVM/MLIR 23.1.0. All 163 compiler tests passed. Each of the five public CLI
reproducers above was executed against the current source. Their failures remain
unpatched; this branch contains the report only.

The review started at `f4093cbdc` and was refreshed to `6cc98f059`. The
intervening commit changes DD caching and leaves the five affected paths
unchanged.

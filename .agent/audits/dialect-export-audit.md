# Dialect export preparation audit

Status: the two accepted performance findings are implemented and validated.
Audit baseline: `0c3fac2cb2861e9e25262857a84650cba651f466` (2026-09-08).
Implementation base: `ce608b0822c25dcfc5554dd367711e1fdd053f80` (2026-09-09).

## Decisions

The accepted scope is repeated OpenQASM preparation and repeated QTensor
fresh-slot reset analysis. Direct QCO exports are deferred. No shrink-pass
cleanup, exporter framework, new pass, or persistent analysis cache is included.

QC remains a supported frontend and interchange representation. Returning to QC
centralizes quantum SSA-to-reference binding, including positional region and
function correspondence and actual tensor-slot updates. Direct export could
avoid constructing QC operations, but must preserve this reasoning and the
classical snapshot/output contracts. Linearity alone does not prove wire order.

## OpenQASM preparation

`Compiler/Pipeline.cpp::runDefaultPipelineImpl` already owns cleaned QC. Emit
that module directly rather than cloning it and running export preparation again
through `QCProgram::toOpenQASM3`. Preserve the public const export API's
preparation and non-consuming behavior.

The regression compares the default OpenQASM output with export through cleaned
QC, including a global phase, sparse register accesses, terminal measurements,
and an unused gate definition. Existing compiler tests retain invalid-input and
round-trip coverage.

Pinned alternating native Release probes reduced the full OpenQASM pipeline from
27.36 to 24.29 ms for 1,000 alternating RY/controlled-X gates on 16 static
wires, and from 313.39 to 272.22 ms for 10,000 gates. All wires are measured and
returned so the circuit remains live. Input parsing/copying is outside timing.

## QTensor reset analysis

`QTensor/IR/Operations/ExtractOp.cpp::RemoveResetAfterExtract` previously proved
the allocation origin separately for each reset, repeatedly walking the same
chain. A successful proof now removes fresh-slot resets together in the root
reset's block. Constant indices are tracked only during that rewrite; the scan
stops at unknown indices or unsupported tensor users. Earlier accesses and
writes prevent a slot from being treated as fresh. The immediate
allocation/extract/reset case still folds without a forward scan.

Existing same-index and commuting-index tests are retained. New tests cover
1,024 fresh slots and an unknown-index access between eligible and ineligible
resets. Verification and quantum linearity are checked before and afterward.
Failed origin proofs retain their existing cost: this is not a general
linear-time claim for every reset program or canonicalization pipeline.

Pinned alternating probes on allocated registers performing reset/H/measure per
constant slot reduced canonicalization from 114.66 to 18.43 ms at 1,000 slots
and 430.51 to 38.15 ms at 2,000 slots. The original diagnostic variant that kept
resets was only hotspot isolation; these implementation probes remove the same
number of resets as the baseline. These are synthetic workload results, not
universal compiler speedups.

## Validation

Validation passed: 1,976 tests across 12 compiler, conversion, QTensor, QCO, and
OpenQASM suites; `uvx nox -s lint`; full-file
`uvx nox -s cpp-lint -- ce608b0822c25dcfc5554dd367711e1fdd053f80` on all four
changed C++ files with zero findings; and
`cmake --build --preset release --target mlir-doc`.

Local ignored probes under `build/dialect-export-audit` retain the workload
generator, timing driver, isolated source variants, alternating measurements,
and logs. The benchmark uses CPU 19, Clang/MLIR 23.1.0, Release with ThinLTO,
two warmups per block and five alternating blocks of three measurements per
version. Final production sources were used in the implementation probes; the
comparison executable uses the recorded audit baseline.

Open PR #2477 also touches QTensor canonicalization. Its index-helper changes
are separate from the accepted traversal optimization. The compact GHZ input
from the audit failed existing OpenQASM constant-index support checks in both
paths, so no compact-loop speedup is claimed. No hosted CI result is inferred
from local validation.

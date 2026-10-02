# QCO DD classical evaluation

Status: complete.

The DD interpreter evaluates scalar integer, index, and f64 operations for
runtime arguments, measurements, and structured control flow. It preserves input
IR, integer widths, numeric semantics, and existing runtime diagnostics.
Ordinary values use LLVM APInt/APFloat and standard math functions. MLIR folding
on a clone retains the behavior of overflow, exact, and non-negative flags,
floating-point flags, explicit rounding, non-finite inputs, and math domains.
The shared integer evaluator owns shift-range validation.

`QCOProgram::cleanup()` already runs canonicalization without an iteration
limit. It remains an optional compiler optimization. MLIR defines
canonicalization as best effort, not a stable normal form. Dynamic operands
survive the pass; requiring it would not remove their execution or validation
code. The low-level APIs accept verified functions and caller-owned argument
bindings.

The runtime-input tests cover every added operation, all comparison predicates,
wide integers, signed zero, non-finite values, flags, rounding, and invalid
shifts and math domains. After rebasing onto the terminology changes in #2519,
3,513 native tests passed and one existing sample-device job-ID test was
skipped. Repository lint and full changed-file C++ lint passed. The rebase
preserves OpenQASM import names and MQT compiler terminology.

A matched Clang 23/LLVM-MLIR 23.1.0 ThinLTO experiment on a DGX Spark used five
serial runs on CPU 0 per variant. Both variants share the rebased libraries; the
control retains only the earlier direct-addition optimization. At 9,000
iterations and 64 dynamic shots, mixed integer evaluation fell from 1,824 ms to
860 ms and mixed floating-point evaluation from 1,970 ms to 1,411 ms. Observed
ranges were 1,819–1,836 ms versus 857–919 ms for integers, and 1,927–2,033 ms
versus 1,378–1,442 ms for floats. Addition and gate-sampling controls stayed
within the observed spread. All 120 runs preserve ordered outcomes and input IR.

Canonicalization leaves the dynamic loops unchanged. For constant math, it
reduces the control from 2,825 ms to 1,212 ms; direct evaluation then reduces
the canonicalized program to 873 ms. Canonicalization costs about 0.5 ms here.
Unlimited canonicalization reaches the same fixed point on these fixtures. These
measurements apply to classical interpretation, not general quantum-gate
simulation. Hosted CI is separate from local validation.

# MLIR dialects and passes

This reference describes the MQT Compiler Collection's MLIR infrastructure. For
compilation, simulation, and device execution, start with
{doc}`compilation and execution <../compilation/index>`. Compiler contributions
follow the [development policy](../development.md#mlir).

We define multiple dialects, each with its dedicated purpose:

- The {doc}`MQT dialect <MQT>` stores frontend-neutral program metadata that
  remains meaningful across dialect conversions.
- The {doc}`QC dialect <QC>` uses reference semantics and is designed as a
  compatibility dialect that simplifies translations from and to existing
  representations such as Qiskit circuits, OpenQASM, or QIR.
- The {doc}`QCO dialect <QCO>` uses value semantics and is mainly designed for
  running optimizations.
- The {doc}`QTensor dialect <QTensor>` adds support for one-dimensional tensors
  of qubits with linear semantics and is used in the QCO dialect to represent
  collections of qubits such as registers.
- The {doc}`CBit dialect <CBit>` represents initialized classical-bit registers
  shared by QC and QCO.

These dialects define various canonicalization patterns and transformation
passes that enable the compilation of quantum programs to native quantum
hardware. Passes that are not tied to a single dialect are documented on the
{doc}`passes <Transforms>` page. For interoperability, we provide
{doc}`conversions <Conversions>` between dialects.

## Quantum resource contract

QC, QCO, and QTensor share these requirements for semantic round trips:

- Quantum operands of a gate or helper call denote disjoint resources. A
  register and one of its elements cannot be separate operands of the same call.
  Multiple references to a qubit are allowed when used sequentially.
- Each register slot retains the same qubit identity throughout its lifetime.
  `qtensor.extract` borrows that qubit; `qtensor.insert` restores it to the same
  slot. Gates change quantum states, including `swap`, which exchanges states
  without changing the identities stored in the register. QC register stores may
  initialize a physical reference buffer or write back the same reference; they
  cannot move, replace, or copy qubits between slots.
- A register passed to a helper, carried through a structured region boundary,
  returned, or dynamically deallocated must be complete: every extracted qubit
  has been restored. A scalar helper may use an extracted qubit while the
  remaining tensor stays in its caller.
- Quantum results of helpers and structured control flow correspond to quantum
  inputs in the same order. Updated quantum arguments form a suffix after the
  ordinary function results. Quantum values in QCO have exactly one use.

Indices must be in bounds. Distinct runtime indices used together must select
non-overlapping resources; extraction and reinsertion indices must select the
same slot. These are program preconditions. Conversions diagnose known
violations and unsupported IR shapes, but do not prove arbitrary index
relationships or insert runtime checks. Type checking and linearity alone do not
establish this contract.

Slot identity refers to the underlying register and index. A view or slice may
renumber those indices; it does not change the qubits they denote. Overlapping
views may be used sequentially, but their overlapping resources cannot be
separate quantum operands of one operation. This contract does not imply that
every conversion already accepts every view representation.

For example, this helper borrows one qubit. The caller restores it before
passing the complete register to the next loop iteration:

```mlir
func.func private @flip(%q: !qco.qubit) -> !qco.qubit {
  %out = qco.x %q : !qco.qubit -> !qco.qubit
  return %out : !qco.qubit
}
func.func @main() attributes {mqt.entry_point} {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %two = arith.constant 2 : index
  %reg = qtensor.alloc(%two) : tensor<2x!qco.qubit>
  %out = scf.for %i = %zero to %two step %one
      iter_args(%current = %reg) -> (tensor<2x!qco.qubit>) {
    %rest, %q = qtensor.extract %current[%i] : tensor<2x!qco.qubit>
    %flipped = func.call @flip(%q) : (!qco.qubit) -> !qco.qubit
    %full = qtensor.insert %flipped into %rest[%i] : tensor<2x!qco.qubit>
    scf.yield %full : tensor<2x!qco.qubit>
  }
  qtensor.dealloc %out : tensor<2x!qco.qubit>
  return
}
```

Calling a whole-register helper with `%rest` before reinsertion is invalid: that
argument still has an extracted slot. Yielding `%rest` and `%flipped` as
separate loop-carried values is also outside the contract. Keep `%rest` in the
caller while a scalar helper processes `%q`, then restore `%full` before a
whole-register call or loop yield.

QC represents the same loop with `memref.load` and an in-place scalar helper; no
write-back is needed. A loaded QC reference remains valid across later gates,
helper calls, and identity stores because its register slot is stable. Round
trips preserve program behavior, not the number or placement of loads.

```{toctree}
:maxdepth: 2

MQT
QC
QCO
QTensor
CBit
Transforms
Conversions
```

# Quantum allocation scope and function results

Status: implemented; validated locally except the Python test session.

## Goal and scope

A dynamic quantum allocation (`qc.alloc`, `qco.alloc`, qubit `memref.alloc`, or
`qtensor.alloc`) directly in the entry block of the `mqt.entry_point` function
may live until the program ends. Any other dynamic allocation, in a helper
function or in a nested block, must be released in the block that allocates it.
A module without an entry point accepts no dynamic allocation. Classical
allocations and static qubit references are unchanged.

Every function returns one trailing value for each QCO qubit or register
argument, and those values continue the arguments in argument order. Unitary
functions already had this rule; it now covers every function.

## Decisions

The MQT entry-point attribute verifier owns both whole-program rules
(`mqt::verifyQuantumAllocations` and `mqt::verifyQuantumArgumentReturns` in
`mlir/lib/Dialect/MQT/IR/MQTDialect.cpp`). QC/QCO program construction loads the
MQT verifier even for caller-supplied contexts and checks modules without an
entry marker. Raw unmarked MLIR fragments can be verified independently;
operation verification alone does not establish these program-wide invariants.

"Released in the block" means: a QC reference has a `qc.dealloc` or
`memref.dealloc` in the same block; a QCO value, followed forward through the
operations that continue it, reaches `qco.sink` or `qtensor.dealloc` in the same
block. Handing the value to a register, a terminator, or another block lets it
escape. The forward walk crosses region operations and calls by position, so the
verifier confirms the release by tracing the released value back to the
allocation. This rejects a branch that exchanges a scratch qubit with a borrowed
one. The rule is per block rather than per function so that inlining a valid
helper into a loop body keeps the program valid.

Both rules trace values backward with `qco::traceQuantumOrigin`
(`mlir/lib/Dialect/QCO/Utils/FunctionUtils.cpp`). It crosses a region operation
only after proving that every region yields the value at its own position, and
it fails instead of aborting on IR it cannot follow, because verifiers see
unverified IR. The wire and tensor iterators assume linear, well-formed chains
and abort otherwise, so the verifier does not use them.

QCO-to-QC conversion reuses the same checks instead of its own forward walk. It
calls `mqt::verifyQuantumArgumentReturns`, since modules converted outside a
program have not met the entry-point verifier, and uses the shared trace for its
QC-only requirements: every region yields its quantum arguments in order, and
every extracted qubit returns to its own register slot.

Mapping, QIR conversion, and target compilation need every qubit allocated up
front. They call `mqt::verifyEntryBlockQuantumAllocations`, the former strict
rule, and diagnose other placements as unsupported. Hoisting callee allocations
to their callers is the intended way to make such programs lowerable. The
OpenQASM 3 and Qiskit exporters already reject allocations in control flow, and
accept only unitary gate functions as callees, which cannot allocate.

The QC and QCO builders accept dynamic allocation in the entry block of any
non-unitary function. `createFunction` releases what the body leaves live, so
helper allocations satisfy the release rule. Allocation in nested blocks stays
rejected by the builders, which is narrower than the IR rule.

## Validation

With LLVM/MLIR 23.1.0, all configured MLIR CTest entries pass
(`ctest -L mqt-mlir-unittests`). clang-tidy reports nothing new in the changed
C++ files; the remaining compiler diagnostics there predate this change.
Repository hooks pass on the changed files.

`test/python/test_mlir_loops.py::test_loop_resource_allocation_is_rejected_at_export`
has not run locally; it needs the MLIR Python extension built from this change.

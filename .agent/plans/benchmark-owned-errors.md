# Owned benchmark errors

Status: complete; local prototype.

## Goal and scope

Complete the benchmark portion of the exception-free API migration on the LLVM
JSON implementation. Native construction, probability, evaluation, and Shor
factoring return owned LLVM errors. Python retains its constructors and error
categories. Valid numerical results and JSON contracts remain unchanged.

The scope includes all 16 benchmark families, Phase, JSON, native generation
callers, bindings, and their tests. The broader DD, QDMI, and QIR migration is
separate. Teleportation construction and validated getters remain infallible.

## Decisions

Use llvm::Expected and llvm::Error, which already own errors in CoreBench. Keep
CoreBench static so LLVM error identities remain in the consuming image. Use
standard error codes to preserve Python ValueError and OverflowError. Shor
callbacks return Expected counts; Python callback exceptions are captured and
rethrown entirely within the binding. Native frames must not unwind.

Adapt the factory design of PR #2545 rather than importing its old benchmark
implementation. Preserve current subnormal fidelity arithmetic, current family
limits, FanOut, QAOA, and program-sequence support. No shared diagnostic runtime
is needed for this bounded migration.

## Validation

The release suite passes 4,089 native tests and 2,036 Python tests; GCC debug
passes all 167 benchmark tests. C++ lint, repository lint, stub generation, and
executable documentation pass. Public Python stubs are unchanged. Deliberately
changing the overflow category or replacing a Python callback exception makes
the corresponding regression tests fail.

Public GHZ and QFT pipelines produce identical manifests, evaluation results,
and verified LLVM QIR. Paired phase timings show generation, lowering, and
emission dominate. Neither timings nor allocation measurements justify further
optimization. Native benchmark IR contains no exception invokes or throw calls.

Independent review checked error ownership, callback cleanup, numerical
behavior, and binding defaults. No broader diagnostic framework is needed.

## Limits

The native C++ API changes; the Python API stays compatible. Native Shor
callbacks must return errors rather than throw. Allocation failure remains
outside the recoverable-error contract. MSVC retains its normal STL exception
ABI; this prototype has only been built locally on Linux. It has not been pushed
or tested in hosted CI.

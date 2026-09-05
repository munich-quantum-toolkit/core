# Make MQT QIR portable, complete, and safe to execute

Status: historical implementation record.

## Goal and scope

MQT Core currently has a QC-to-QIR compiler, a QIR runtime and JIT runner, and a
DDSIM QDMI device that executes QIR. These components do not share one complete
calling convention. In particular, parameterized gates use an MQT-specific
argument order, controlled gates use names not recognized by other QIR tools,
and the runtime does not implement every call emitted by the compiler. The JIT
and DDSIM device also share process-global simulator state, which makes
concurrent QIR jobs unsafe.

After this work, QIR produced by MQT uses a complete, explicitly documented
quantum instruction set with the common parameter-first convention, uses the
current QIR 2.1 runtime functions where the profile specifies them, and is
rejected with a precise diagnostic if its declarations do not match the
supported ABI. Calls with one or two controls use dedicated QIS functions; calls
with three or more controls use generic `__ctl` and `__ctladj` specializations
with a control Array and the original gate arguments. These functions are an MQT
QIS extension to the Base and Adaptive profiles and do not change the entry
point's `qir_profiles` attribute. The runtime does not emulate qir-runner's
legacy resource-management and output APIs. The standalone runner can select an
entry point and execute multiple deterministic shots. Each DDSIM QIR job owns
its execution state, so concurrent jobs cannot affect one another. A user can
observe the result by compiling the gate-matrix tests, running Base and Adaptive
QIR through the MQT JIT, and submitting concurrent QIR jobs through the DDSIM
QDMI device.

The implementation is delivered as a native GitHub stack with three branches.
The first layer establishes the compiler/runtime contract, the second completes
and isolates the general runner, and the third migrates DDSIM and adds
end-to-end coverage. Every layer must build and pass the tests relevant to its
own diff before the next layer is created.

## Constraints

- The QIR specification defines profile structure and runtime functions, but
  deliberately leaves QIS names backend-defined. Evidence: the Base and Adaptive
  profile documents describe calls to backend-defined QIS functions, while
  qir-runner supplies the widely used body, adjoint, and controlled function
  conventions. The compiler therefore needs both a strict profile verifier and a
  separately documented portable QIS policy.

- `JitSession` locates an `entry_point` function to read metadata, but later
  resolves the literal symbol `main` and invokes it through `int(int, char**)`.
  Evidence: `src/qir/jit/Session.cpp` contains both `getEntryPointFunction` and
  `jit_->lookup("main")`; QIR profile entry points are parameterless and return
  `i64`.

- the QIR v2.1 and legacy qir-runner allocation APIs reuse
  `__quantum__rt__qubit_allocate` with incompatible LLVM function types:
  `ptr(ptr)` and `ptr()`, respectively. Supporting both therefore adds typed
  adapter and registration complexity without improving current-spec compliance.
  The revised implementation supports the QIR 2.1 form only.

- `src/qdmi/devices/dd/Device.cpp` obtains `Runtime::getInstance()` in
  asynchronous job execution. The singleton owns quantum state, result storage,
  RNG state, output state, and measurements, so simultaneous QIR jobs race even
  though non-QIR QDMI concurrency tests pass.

- a conversion-only gate table did not prevent drift because the runtime
  declarations, definitions, and JIT symbols remained separate lists. Evidence:
  after replacing all four lists with `mlir/Conversion/GateTable.def`, the
  release build compiled every gate family and all 58 runtime, 9 JIT, 112
  builder, 267 QC-to-QIR, and 304 QC/QCO conversion tests passed.

- state extraction cannot be implemented by erasing a fixed list of measurement
  and result symbols. Evidence: that rewrite left arbitrary gates after the
  first measurement executable and depended on spellings rather than the Base
  Profile's required `irreversible` attribute. Truncating the selected entry
  point at the semantic boundary removes the entire terminal measurement/output
  region and supports backend-defined measurement names.

## Decisions

- Break the unreleased MQT-specific QIR ABI instead of retaining source or
  binary compatibility. Rationale: compatibility shims would preserve ambiguous
  symbols, incorrect parameter order, and divergent compiler/runtime tables. The
  user explicitly prioritized compliance and broad ecosystem compatibility over
  compatibility with unreleased code.

- Treat qir-spec as the authority for QIR 2.1 profiles and runtime functions,
  and use qir-runner only as a reference for generic controlled specializations.
  Rationale: copying qir-runner's older allocation API into emitted Adaptive QIR
  would make the compiler less standards-compliant.

- Keep MQT's complete extension QIS for compiler/runtime parity while using the
  conventional parameter-first ABI. Retain only qir-runner's Array/Tuple
  representation for generic controlled specializations because it provides a
  practical ABI for arbitrary controls; remove its legacy resource, output, and
  Pauli-rotation compatibility shims. Rationale: arbitrary controls materially
  improve interoperability, while the other shims complicate a runtime whose QIR
  code is not yet released.

- Emit dedicated functions for exactly one and two controls, and use `__ctl` or
  `__ctladj` for three or more. Keep the entry point's declared Base or Adaptive
  profile instead of introducing a `custom` mode. Rationale: the common cases
  remain easy to call, arbitrary controls remain available, and backend-defined
  QIS extensions are independent of the QIR profile.

- Place the cross-cutting gate registry at
  `mlir/include/mlir/Conversion/GateTable.def` and use `prx` as the QIR symbol
  stem for MQT's two-angle R gate. Reserve `r` for qir-runner's incompatible
  Pauli rotation instead of overloading one name with two signatures.

- Make runtime state session-owned and use a scoped active-runtime binding only
  as the private C ABI dispatch mechanism; do not expose a public `Activation`
  helper. Rationale: this preserves plain exported C entry points while
  isolating parallel JIT sessions and DDSIM jobs without making dispatch
  mechanics part of the public runtime API.

- Derive state extraction from the selected Base Profile entry point's
  `irreversible` boundary and reject every other profile. Rationale: this
  follows the profile's semantic contract, remains compatible with
  backend-defined measurement names, and fails safely rather than changing
  adaptive behavior.

## Outcome and validation

The implementation is complete as a three-layer stack. The bottom PR owns the
shared MLIR gate registry and compiler/runtime QIS contract. The middle PR owns
the current QIR 2.1 resource APIs, strict JIT validation, session-local runtime,
and runner usability. The top PR owns DDSIM integration, Base-profile semantic
state extraction, and concurrent-job isolation. The cumulative release build,
all 4,622 CTest cases, the focused QIR and DDSIM suites, repository lint, and
diff checks pass. Two job-ID queries remain fixture-skipped because no external
job service is configured; no implementation was weakened for them.

## Code and ownership

The compiler path starts in `mlir/lib/Conversion/QCToQIR/`. Common preparation
and gate conversion live in `QIRCommon/QIRCommon.cpp`; profile-specific resource
and control-flow lowering live in `QIRBase/QCToQIRBase.cpp` and
`QIRAdaptive/QCToQIRAdaptive.cpp`. Function names and output helpers are in
`mlir/include/mlir/Dialect/QIR/Utils/QIRUtils.h` and
`mlir/lib/Dialect/QIR/Utils/QIRUtils.cpp`. The public builder in
`mlir/include/mlir/Dialect/QIR/Builder/QIRProgramBuilder.h` must use the same
argument order and names as automatic lowering.

The host runtime C ABI is declared by `include/mqt-core/qir/runtime/QIR.h` and
implemented in `src/qir/runtime/QIR.cpp`. Simulator state and DD operations live
in `include/mqt-core/qir/runtime/Runtime.hpp` and `src/qir/runtime/Runtime.cpp`.
`src/qir/jit/Session.cpp` parses LLVM text or bitcode, binds host functions, and
invokes the entry point. The CLI wrapper is `src/qir/runner/Runner.cpp`.

The DDSIM QDMI device accepts QIR text and bitcode in
`src/qdmi/devices/dd/Device.cpp`. Sampling repeatedly executes the JIT and
collects `Runtime::getMeasurements()`. Statevector retrieval rewrites a Base
Profile module so execution stops before measurement and then takes the DD state
from the runtime. Tests mirror these components under `test/qir/`,
`mlir/unittests/Conversion/QCToQIR/`, and `test/qdmi/devices/dd/`.

A QIR profile constrains LLVM control flow, resource representation, runtime
calls, attributes, and entry-point signature. A QIS is the backend-defined set
of quantum gate functions. This work uses QIR 2.1 for the former and MQT's
explicitly documented QIS for the latter. Its generic controlled specializations
use the same Array/Tuple argument representation as qir-runner.

## Acceptance

The bottom layer is accepted when every operation in the shared gate table has
matching compiler lowering, C declaration, runtime implementation, and JIT
registration; no generated declaration uses the old qubits-first parameter
order; duplicate aliases are removed; and the runtime, JIT, builder, QC/QCO,
Base, and Adaptive compiler unit tests pass.

The middle layer is accepted when deliberately malformed entry-point and
declaration types fail with diagnostics that name the symbol and expected type,
two JIT sessions can execute independently, an entry point whose name is not
`main` returns its `i64` status correctly, a fixed seed reproduces a multi-shot
result sequence, output contains complete metadata and the actual shot exit
code, modern result arrays are recorded as one bit string in memory order, and
representative body, adjoint, and generic controlled modules execute through the
supported ABI without relying on legacy qir-runner resource or output adapters.

The top layer is accepted when simultaneous DDSIM QIR jobs return correct,
independent histograms without writing to global stdout, Base state extraction
returns the pre-measurement state, Adaptive sampling succeeds, and the existing
Python FoMaC QIR execution test still passes.

The entire stack is accepted when the release build, affected C++/MLIR/Python
tests, repository lint session, `git diff --check`, and clean-status audit pass
for the final cumulative head. Any environment-only limitation must be recorded
with its exact command and diagnostic instead of weakening the implementation.

## Interfaces

The canonical compiler-facing QIS must use parameter-first signatures such as
`void __quantum__qis__rx__body(double, Qubit*)` and fixed controlled shorthands
such as `void __quantum__qis__cx__body(Qubit*, Qubit*)`. MQT's two-angle `R`
gate uses `void __quantum__qis__prx__body(double, double, Qubit*)`; the
incompatible qir-runner Pauli-rotation spelling `__quantum__qis__r__body` is not
part of MQT's supported QIS.

The current Adaptive runtime interface must include the QIR 2.1 allocation
forms, including `Qubit* __quantum__rt__qubit_allocate(bool* outErr)` and
buffer-based array allocation/release. The exact exported source spelling may
use the repository's opaque QIR types, but the translated LLVM declaration must
match the profile's `ptr(ptr)` and related signatures.

`JitSession::run` must invoke an exact `int64_t (*)()` entry-point function.
`JitSession` must expose its owned `Runtime&` so DDSIM can retrieve measurements
or move out the state without consulting global state. Runtime reset must clear
per-shot quantum/results/output state without silently reseeding a configured
random-number stream.

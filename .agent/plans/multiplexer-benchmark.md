# Add a scalable, validated quantum multiplexer benchmark

Status: historical implementation record.

## Goal and scope

MQT Core users can generate a typed quantum multiplexer benchmark whose output
is nontrivial and independently checkable. The program prepares every control
state with equal probability and rotates one target qubit by an angle selected
from the control value. The analytic reference predicts the complete sampled
distribution, so an execution test detects incorrect controls, angles, target
placement, and result-bit order.

The fixed angle schedule has an exact linear implementation. A benchmark with
`k` controls executes `k` Hadamard gates and `k` singly controlled Y rotations
instead of iterating over all `2^k` control states. The generated QC program
keeps those operations in structured loops and remains compact at the supported
maximum of 1024 total qubits.

## Constraints

- The fixed schedule is not a general arbitrary-angle multiplexer. For control
  bits `b_i`, its angle is `pi * s / 2^k = sum_i b_i * pi / 2^(k-i)`. Controlled
  Y rotations around the same axis add, so one singly controlled rotation per
  bit implements the exact same unitary.

- Uniform control preparation turns the former all-zero output into a
  distribution that tests every control state in one execution.

- The 1024-qubit program lowers to Jeff, serializes to bytes, and deserializes
  as valid Jeff in the focused MLIR test.

- The three-qubit program passes the public Python generation, QCO lowering, DD
  sampling, and analytic evaluation path with total variation distance below
  0.03.

- Running `stubs` before `docs` left the shared MinSizeRel Python build tree
  configured with `BUILD_MQT_CORE_DOCUMENTATION=OFF`. The first docs attempt
  therefore reported 25 missing generated MLIR reference files. An explicit
  `mqt-core` reinstall with the docs session's CMake arguments regenerated those
  files, after which the unmodified docs command passed.

## Decisions

- Keep the fixed schedule `theta(s) = s*pi/2^k`, where `k` is the control count.
  Rationale: it retains the existing one-parameter family and permits an exact
  linear circuit without an exponential angle payload.

- Apply Hadamard gates to all controls and leave the target in `|0>`. Rationale:
  the resulting distribution exercises the multiplexer semantics instead of
  selecting only state zero.

- Traverse controls from most to least significant, start the controlled Y angle
  at `pi/2`, and halve it after each iteration. Rationale: this directly
  implements the binary weights without an integer state count or a `2^k` loop.

- Support 2 through 1024 total qubits. Rationale: 1024 is a clear power-of-two
  catalogue ceiling; at 1023 controls both the uniform control weight and the
  smallest scheduled angle remain nonzero as binary64 values, and the maximum
  program round-trips through Jeff.

- Store the target measurement at result index zero and control `i` at index
  `i + 1`. Rationale: the displayed big-endian result is `c[k-1]...c[0]t`, which
  permits direct binary interpretation of its control prefix.

- Keep family ID `multiplexer`, definition version 1, and the sole `qubits`
  parameter. Rationale: the family is unreleased and needs no compatibility
  method or arbitrary-angle variant.

## Outcome and validation

The implementation now has a nontrivial analytic reference and an exact linear
generator. The maximum is 1024 qubits, where the compact structured program
lowers to Jeff and survives a stable byte round-trip. The public Python path
executes the generated circuit through QCO and the DD sampler and agrees with
the analytic distribution. All focused and repository-wide validation passes.

## Code and ownership

The benchmark interface lives in `include/mqt-core/bench/Multiplexer.hpp` and
`src/bench/Multiplexer.cpp`. It validates the qubit count, describes one logical
`result` output, returns ideal probabilities, and evaluates sampled counts.
`src/bench/JSON.cpp` supplies the strict parameter schema, canonical instance
specification, manifest, and generic evaluation integration.

The generator lives in `mlir/bench/programs/Multiplexer.cpp`. It uses
`qc::QCProgramBuilder` and standard `scf.for` operations to construct compact QC
dialect IR. The normal compiler pipeline lowers that program to QCO and Jeff.
The Python binding in `bindings/bench/register_multiplexer.cpp` exposes the same
typed family and calls the shared generator.

For `q` total qubits, let `k = q - 1`. A displayed outcome is `c[k-1]...c[0]t`,
where `t` is the target bit. Interpret the control prefix as the binary fraction
`x = 0.c[k-1]...c[0]`, so the selected rotation is `theta = pi*x`. Its ideal
probability is `2^-k * cos(theta/2)^2` for `t = 0` and `2^-k * sin(theta/2)^2`
for `t = 1`.

## Acceptance

The C++ tests must accept qubit counts 2 and 1024, reject 1 and 1025, validate
outcome widths and characters, and match all eight three-qubit probabilities.
The JSON tests must publish 1024 as the maximum and retain strict parsing,
canonical manifests, and stable case identity.

The MLIR tests must inspect semantic operations rather than a textual snapshot.
They must prove uniform control preparation, one single-control rotation body,
the correct binary angle order, no state-selection X gates, bounded generated
IR, and a successful maximum-size Jeff byte round-trip.

The Python test must exercise the public family, generator, QC-to-QCO lowering,
DD sampler, output order, and analytic evaluator in one path. Compilation alone
does not satisfy this acceptance criterion.

All focused tests, the full CTest suite, stub check, documentation build, C++
lint, full lint, and final diff checks must pass. Record an environment or
infrastructure failure with its command and output instead of presenting it as a
product failure.

## Interfaces

The installed C++ and Python interfaces remain `MultiplexerOptions`/`Options`
and `Multiplexer`, with `qubits` as their only parameter. The supported maximum
changes from 31 to 1024 and the ideal probability changes from an all-zero
placeholder to the documented uniform-control distribution. Family ID
`multiplexer` and definition version 1 remain unchanged.

Use only MQT Core, LLVM, MLIR, nanobind, nlohmann JSON, GoogleTest, and the
existing DD runtime. Add no dependency.

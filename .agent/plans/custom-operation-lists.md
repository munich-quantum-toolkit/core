# Expose custom operation lists through FoMaC

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

After this change, a QDMI provider can return an implementation-defined device
property whose value is an array of `QDMI_Operation` handles, and FoMaC users
can consume that list without decoding raw bytes. C++ users call
`Device::queryCustomOperations(CustomProperty)`. Python users call
`Device.query_custom_operations(CustomProperty)`. Both interfaces return no
value when the provider does not support the selected custom property. Returned
operations use the normal FoMaC `Operation` API and retain the device session
that owns their QDMI handles.

The visible proof is a test provider that exposes valid, empty, malformed, and
unsupported custom operation properties. Its tests query operation names from
valid handles after the original device wrapper is gone, and the normal Core
providers demonstrate that unsupported properties return `std::nullopt` and
`None`.

## Progress

- [x] (2026-08-10 16:20Z) Created an isolated worktree from current
      `origin/main` and reviewed the FoMaC query, ownership, binding, and
      test-provider paths.
- [x] (2026-08-10 16:25Z) Added the generic handle-array decoder, shared
      operation-wrapper path, and public C++ API.
- [x] (2026-08-10 16:29Z) Bound the API to Python and regenerated the checked-in
      stub.
- [x] (2026-08-10 16:34Z) Extended the native test provider and added raw QDMI,
      C++, Python unsupported-result, operation-property, and ownership tests.
- [ ] Add the changelog entry after the draft pull request provides its number.
- [x] (2026-08-10 16:52Z) Ran focused and full C++ tests, focused Python tests,
      generated-file checks, documentation, changed-line clang-tidy 22, and full
      lint. The final lint check will be repeated after the PR-numbered
      changelog entry.
- [ ] Publish the exact reviewed head and inspect replacement
      continuous-integration checks.

## Surprises & Discoveries

- Observation: Existing FoMaC operations already store the device as a shared
  owner, so a custom list does not need a new wrapper type. Evidence:
  `Operation` in `include/mqt-core/fomac/FoMaC.hpp` stores
  `std::shared_ptr<QDMI_Device_impl_d> device_`, and `Device::getOperations` in
  `src/fomac/FoMaC.cpp` constructs each operation with that owner.
- Observation: The Python test package contains only production providers, all
  of which correctly report custom operation slots as unsupported. The
  successful custom-list and post-device-lifetime path must therefore use the
  native test provider, while Python can verify the public unsupported result
  and the shared `Operation` lifetime contract that both standard and custom
  lists use.

## Decision Log

- Decision: Add a handle-array decoder in `fomac::detail` and use it from the
  new API. Rationale: Size validation and the two-step QDMI query protocol are
  independent of the Amazon Braket provider and can be reused if QDMI later
  standardizes another handle-array property. Date/Author: 2026-08-10 / Codex.
- Decision: Return `std::optional<std::vector<Operation>>` in C++ and
  `list[Device.Operation] | None` in Python. Rationale: An unsupported property
  and a supported empty operation list are different provider results and must
  remain distinguishable. Date/Author: 2026-08-10 / Codex.
- Decision: Keep raw `queryCustomProperty<std::vector<std::byte>>` unchanged.
  Rationale: The new typed method is additive and does not remove the lossless
  raw custom-property path. Date/Author: 2026-08-10 / Codex.

## Outcomes & Retrospective

The C++, Python, and raw QDMI paths are implemented and locally validated. The
successful provider path distinguishes unsupported, empty, malformed, and valid
arrays; normal operation-property queries and ownership after device-wrapper
destruction both pass. The documentation build completed without a content
warning. Publication, the PR-numbered changelog entry, and live continuous
integration remain.

## Context and Orientation

QDMI is the C interface that provider libraries implement. A device property is
queried in two calls: the first call uses a null output pointer and returns the
required byte count; the second call provides a buffer of that size. An
operation handle has the C type `QDMI_Operation`. It remains valid only while
its owning device session remains valid.

FoMaC is MQT Core's C++ and Python wrapper around QDMI. The public C++ classes
are declared in `include/mqt-core/fomac/FoMaC.hpp`, and their non-template
methods are implemented in `src/fomac/FoMaC.cpp`. Python bindings are defined
with nanobind in `bindings/fomac/FoMaC.cpp`. The generated public type stub is
`python/mqt/core/fomac.pyi`; it must be regenerated with the repository's Nox
session and must not be edited by hand.

`CustomProperty` selects QDMI custom slots one through five. The existing
`Device::queryCustomProperty<T>` method decodes scalar, string, or raw-byte
values. It cannot safely turn arbitrary raw bytes into owned FoMaC operations.
The new method supplies that missing typed path while retaining the existing raw
path.

The native test provider in `test/qdmi/driver/session_device.cpp` is loaded by
`test/qdmi/driver/test_driver.cpp`. It can expose deterministic custom property
responses and operation-property responses without changing a production
provider. The driver test already links FoMaC and already receives the provider
library path through `test/qdmi/driver/CMakeLists.txt`.

This task changes only this worktree. It must preserve unrelated changes and
must not edit another task's worktree. It must follow `AGENTS.md` and
`docs/ai_usage.md`. This plan records implementation work but does not by itself
authorize a GitHub action.

## Plan of Work

First, add a small template in `fomac::detail` in
`include/mqt-core/fomac/FoMaC.hpp`. The template accepts a QDMI-style query
callable, returns `std::nullopt` on `QDMI_ERROR_NOTSUPPORTED`, rejects a byte
count that is not a multiple of the handle size, and otherwise performs the
second query into a correctly sized vector. It must preserve a supported empty
list as an engaged optional containing an empty vector.

Next, declare `Device::queryCustomOperations(CustomProperty)` next to the
existing custom device-property API and implement it in `src/fomac/FoMaC.cpp`.
Convert the selector with `detail::toDeviceProperty`, use the generic decoder
for `QDMI_Operation`, and wrap each handle in the same `Operation` class used by
`Device::getOperations`. Every wrapper receives `device_`, so it retains the
owning session.

Then, bind the method as `Device.query_custom_operations` in
`bindings/fomac/FoMaC.cpp`. The docstring must state the difference between an
unsupported property and a supported empty list. Regenerate
`python/mqt/core/fomac.pyi` with the Nox stub session.

Extend `test/qdmi/driver/session_device.cpp` with stable test operation objects.
One custom device-property slot returns their handle array, one reports a
supported empty list, one reports a malformed byte count, and another remains
unsupported. Implement the operation name and fixed arity/parameter queries for
the returned handles. In `test/qdmi/driver/test_driver.cpp`, verify raw QDMI
size/value queries, FoMaC unsupported and empty results, malformed-size
rejection, valid operation wrappers, operation-property queries, and ownership
after the temporary `Device` wrapper is destroyed. Add Python tests in
`test/python/fomac/test_fomac.py` for the unsupported result and the established
operation lifetime contract shared by both list-producing methods.

Finally, add a concise changelog entry with the actual pull request number,
validate the complete diff, commit with a signed commit and the required
`Assisted-by: GPT-5.6 via Codex` trailer, publish the branch, and inspect all
checks against the exact published head.

## Concrete Steps

Run all commands from the repository root. Configure and build the focused
native tests with:

    MLIR_DIR=/path/to/llvm/lib/cmake/mlir ./.agent/run.sh cmake --preset release
    ./.agent/run.sh cmake --build build/release --target mqt-core-qdmi-driver-test mqt-core-fomac-test
    ./.agent/run.sh build/release/test/qdmi/driver/mqt-core-qdmi-driver-test --gtest_filter='DeviceRegistrationTest.CustomOperation*'
    ./.agent/run.sh build/release/test/fomac/mqt-core-fomac-test

Build the bindings, run focused Python tests, and regenerate stubs with:

    MLIR_DIR=/path/to/llvm/lib/cmake/mlir ./.agent/run.sh uvx nox --non-interactive -s tests-3.13 -- test/python/fomac/test_fomac.py -k 'custom_operations or operation_keeps'
    MLIR_DIR=/path/to/llvm/lib/cmake/mlir ./.agent/run.sh uvx nox --non-interactive -s stubs

Run the repository gate with:

    MLIR_DIR=/path/to/llvm/lib/cmake/mlir ./.agent/run.sh uvx nox --non-interactive -s lint
    git diff --check
    git status --short

The focused native test must report every custom-operation test as passed. The
focused Python test must report no failure. The stub session must leave only the
expected `fomac.pyi` API addition. The final lint session and diff check must
exit successfully.

## Validation and Acceptance

The C++ API is accepted when an unsupported custom property produces
`std::nullopt`, a supported zero-byte property produces an engaged empty vector,
and a malformed byte count throws `std::invalid_argument`. A valid property must
return the expected number of `Operation` objects. Their names, arities, and
parameter counts must come from normal QDMI operation-property queries. At least
one returned operation must remain usable after the `Device` variable that
produced it is destroyed.

The raw C path is accepted when the same test provider returns the documented
byte count and the exact handles through `QDMI_device_query_device_property`.
This proves that the FoMaC method does not require provider-specific code.

The Python API is accepted when every production device returns `None` for its
unsupported custom slot, the generated annotation is
`list[Device.Operation] | None`, and an operation created from a freshly opened
device remains usable without a separate device variable. Downstream provider
tests can then exercise a supported custom list without changing this generic
Core API.

## Idempotence and Recovery

All build, test, stub-generation, and lint commands are repeatable. Generated
build files stay below `build/`, `.nox/`, and `.cache/` and are ignored by Git.
If CMake cached an obsolete toolchain, move only the affected generated build
directory aside and rerun the configure preset. Do not reset or clean source
files to recover. If publication finds a changed remote branch, stop and fetch
the new head instead of using an unguarded force push.

## Artifacts and Notes

Expected key result shapes are:

    unsupported.has_value() == false
    empty.has_value() == true && empty->empty()
    valid->front().getName() == "custom-rx"
    malformed query throws std::invalid_argument

Add final exact test counts and the published commit SHA here after validation.

Current focused evidence:

    121/121 mqt-core-qdmi-driver-test cases passed
    268/268 mqt-core-fomac-test cases passed
    6/6 focused Python cases passed
    changed-line clang-tidy 22 produced no diagnostics
    Nox lint and documentation sessions passed

## Interfaces and Dependencies

The completed public C++ interface must contain:

    [[nodiscard]] std::optional<std::vector<Operation>>
    Device::queryCustomOperations(CustomProperty property) const;

The completed Python interface must contain:

    def query_custom_operations(
        self, custom_property: CustomProperty
    ) -> list[Device.Operation] | None: ...

The implementation depends only on the existing QDMI client API, the existing
FoMaC `CustomProperty` selector, and the existing `Operation` ownership model.
It adds no provider SDK and no new external dependency.

Revision note (2026-08-10): Created the initial self-contained execution plan
after inspecting the current FoMaC, nanobind, generated-stub, and native test
provider implementations.

Revision note (2026-08-10): Recorded the completed API, provider fixture,
generated binding, and local native, Python, ownership, and clang-tidy evidence.

Revision note (2026-08-10): Recorded the successful full documentation build and
completed local validation gate before publication.

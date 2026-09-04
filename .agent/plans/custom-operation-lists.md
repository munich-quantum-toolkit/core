# Expose custom operation lists through FoMaC

Status: historical implementation record.

## Goal and scope

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

## Constraints

- Existing FoMaC operations already store the device as a shared owner, so a
  custom list does not need a new wrapper type. Evidence: `Operation` in
  `include/mqt-core/fomac/FoMaC.hpp` stores
  `std::shared_ptr<QDMI_Device_impl_d> device_`, and `Device::getOperations` in
  `src/fomac/FoMaC.cpp` constructs each operation with that owner.

- The Python test package contains only production providers, all of which
  correctly report custom operation slots as unsupported. The successful
  custom-list and post-device-lifetime path must therefore use the native test
  provider, while Python can verify the public unsupported result and the shared
  `Operation` lifetime contract that both standard and custom lists use.

## Decisions

- Add a handle-array decoder in `fomac::detail` and use it from the new API.
  Rationale: Size validation and the two-step QDMI query protocol are
  independent of the Amazon Braket provider and can be reused if QDMI later
  standardizes another handle-array property.

- Return `std::optional<std::vector<Operation>>` in C++ and
  `list[Device.Operation] | None` in Python. Rationale: An unsupported property
  and a supported empty operation list are different provider results and must
  remain distinguishable.

- Keep raw `queryCustomProperty<std::vector<std::byte>>` unchanged. Rationale:
  The new typed method is additive and does not remove the lossless raw
  custom-property path.

## Outcome and validation

C++, Python, and raw QDMI paths distinguish unsupported, empty, malformed, and
valid custom operation arrays. Tests cover ordinary property queries and
ownership after device-wrapper destruction. Local native/Python validation and
documentation passed; final publication and hosted CI were not recorded.

## Code and ownership

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

## Acceptance

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

## Interfaces

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

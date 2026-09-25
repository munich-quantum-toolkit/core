# Replaceable QDMI driver

Status: implemented in Core #2229. Native tests, Python tests and executable
documentation pass. C++ lint and installed-consumer validation accompany
publication.

The C++ and Python wrappers consume the standard QDMI Client Interface. Each
session selects its driver; devices and jobs retain their session. Validated
libraries remain loaded through global destruction. The optional
`builtin_driver` API discovers installed manifests without loading devices and
opens independent device sessions with configured stable IDs and per-call
overrides.

Use QDMI's example driver to verify runtime replacement and failed
initialization. Keep only a small invalid-driver fixture for ABI and symbol
rejection. Existing registry, driver and SDK tests cover configuration,
lifetimes and execution. Validate installed CMake consumers locally on the
current wheel, alongside the full native suite, relevant Python tests, generated
stubs, lint and documentation.

Publish the combined implementation in #2229; close #2230 and #2231. Provider
source pins and temporary LLVM setup remain until suitable releases exist.

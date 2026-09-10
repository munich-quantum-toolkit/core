# Device-directed compilation

Status: complete.

`compile_program` selects a device-supported payload before mapping and returns
`CompiledProgram`. `Device.submit` checks compatibility and submits source or
compiled programs; `submit_program` opens the device and delegates. The C++ QDMI
adapter owns format selection and compatibility checks.

Selection prefers Adaptive QIR, OpenQASM 3.1, then Base QIR, with binary before
text. Capabilities default to maximal compiler support. Explicit payload
specifications can impose restrictions. Compatibility ignores calibration
changes but preserves ordered site and operation semantics. QIR submission uses
an `i64 ()` entry point.

Validated with 452 native tests, 723 Python tests, six documentation cells,
regenerated stubs, repository lint, and whole-file C++ lint.

Source submission uses one device snapshot. Artifact submission refreshes
legality without querying calibration. Routing distances are computed once on
demand; connectivity validation remains eager. Reordered operation tuples use
sorted views for compatibility checks. Measurements and reproduction commands
are in
[the API benchmark](../benchmarks/device-compiler-api/README.md).

Examples display counts and demonstrate DDSIM state extraction after sampling.
OpenQASM payload capabilities match the exporter's 3.1 output, including switch.

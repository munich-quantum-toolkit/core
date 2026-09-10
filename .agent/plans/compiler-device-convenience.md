# Device-directed compilation

Status: complete.

`compile_program` selects a device-supported payload before mapping and returns
`CompiledProgram`. `Device.submit` checks compatibility and submits source or
compiled programs; `submit_program` opens the device and delegates. The C++ QDMI
adapter owns format selection and compatibility checks.

Selection prefers Adaptive QIR, OpenQASM 3, then Base QIR, with binary before
text. Capabilities default to maximal compiler support; DDSIM confirms this with
a versioned marker. Explicit payload specifications can impose restrictions.
Compatibility ignores calibration changes but preserves ordered site and
operation semantics. QIR submission uses an `i64 ()` entry point.

Validated with 522 native tests, 334 Python tests, four documentation cells,
generated stubs, and repository lint checks.

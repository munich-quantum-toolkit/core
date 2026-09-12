# Reproducible target mapping controls

Status: complete.

## Goal and scope

Expose the existing native mapper seed and initial trial count through C++,
Python, and `mqt-cc`. Target compilation owns these options; Qiskit performs no
mapping or synthesis. Preserve current defaults and existing calls.

## Decisions

Use one `MappingOptions` value across the target pipeline and QDMI compilation.
An omitted trial count retains the mapper's CPU-dependent default. An explicit
positive count and seed make trial selection independent of CPU count for a
fixed Core build, input, and target. All-to-all placement needs no randomized
routing and ignores valid mapping options. Reject zero trials before rewriting.
These controls do not prescribe the selected layout or guarantee the same result
across compiler releases. Layout provenance is a separate change.

## Validation

The seven mapping cases in `test/python/test_mlir.py` pass, as do 37 native
`*Target*:*Mapping*` tests and all four `mqt-cc` CTests. The native build, stub
generation, repository lint, and full changed-file C++ lint pass.

## Outcome

`MappingOptions` reaches the existing mapper from native, Python, QIR/device,
and CLI entry points. Existing call defaults remain unchanged. To submit source
with explicit controls, compile first and submit the compiled payload. Layout
selection and layout provenance remain separate contracts.

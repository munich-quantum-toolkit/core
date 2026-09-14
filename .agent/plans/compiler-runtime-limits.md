# Preserve structured compiler and runtime execution

Status: complete.

## Scope and decisions

- QC/QCO helper conversion preserves borrowed-register shapes and returned wire
  order through the shared `FunctionUtils` analysis. Entry points keep
  ownership.
- `TranslateQCToOpenQASM3.cpp` emits logical indices directly, immutable
  physical reference arrays as site switches, and constant rank-one f64 tables
  as grouped switches. This uses the existing frontend without new array or
  function syntax. Site dispatch retains its 65,536-case and 256-level nesting
  limits; explicit table data is exempt because its expansion is linear in table
  size and reads.
- Indexed loops survive placement only for unrestricted all-to-all targets and
  payloads with multiway branching. Indexed-loop routing needs scalar-wire
  selection and layout handling; raising the unrolling budget does not fix it.
- DD execution limits conditional loops to 100,000 iterations and rejects
  recursive calls. Counted loops use widened APInt trip counts. Runtime
  assertions are omitted by the frontend; verified IR retains its bounds
  preconditions.
- Fixed-gate powers share constant/runtime lowering with rotations and global
  phase. Direct DD interpretation reuses matrix powering for constant bodies;
  arbitrary runtime body matrices remain unsupported.
- Adaptive QIR reuses Boolean storage for computed CBit outputs. Boolean records
  join DDSIM shot/count strings; measurement-only registers retain Result arrays
  and batch sampling. Base-profile restrictions remain unchanged.

## Validation

Use the native build and checks in [AGENTS.md](../../AGENTS.md), plus
`test/python/qdmi/test_compilation.py` against the rebuilt package and provider.
Owning tests cover borrowed-wire order, physical site identity, emission limits,
constant/runtime full matrices at `5e-13`, and mixed computed/measured outputs.
The optimized native suite passes all 3,557 entries apart from one expected
Slurm skip. All 50 Python compilation tests, executable docs, generated-link
checks, repository lint, and whole-file C++ lint for 28 changed sources pass.

Earlier probes executed Shor 65 and 143 OpenQASM with seeded QIR agreement. The
31-bit composite 2147483645 compiled and reimported as 27.8 MB of OpenQASM;
simulation was not attempted. Shared table functions need frontend support.

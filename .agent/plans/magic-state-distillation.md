# Concatenated magic-state distillation benchmark

Status: complete; implementation, required execution paths, tests, generated
stubs, and executable documentation are validated.

## Outcome and scope

`MagicStateDistillation` and Python `bench.magic_state_distillation` implement
15-to-1 Reed–Muller distillation of ideal `|T> = T|+>` inputs. The
`magic-state-distillation` family has definition version 1 and supports
`levels=1` through `levels=4`, with one level as the default. It allocates
exactly 15, 225, 3,375, or 50,625 qubits, respectively.

C++, Python, strict JSON, CLI discovery/generation/evaluation, manifests, and
semantic case IDs use the existing benchmark infrastructure. The two-bit
`result` has a sticky rejection flag in bit 1 and a root-state check in bit 0;
ideal big-endian output is `00`. Noise options, physical error correction,
independent factories, retries, and the separate Shor benchmark are outside this
change. Ideal inputs check execution, without claiming an improvement in
noisy-state fidelity.

The implementation lives in `include/mqt-core/bench/MagicStateDistillation.hpp`,
`src/bench/MagicStateDistillation.cpp`, and
`mlir/bench/programs/MagicStateDistillation.cpp`. Executable examples are in
`docs/benchmarks.md`; tests mirror the native, MLIR, and Python entry points.

## Decisions

- Use the direct input-state protocol in
  [Bravyi–Haah, Appendix A](https://arxiv.org/pdf/1209.2426) and the 15-qubit
  code of [Bravyi–Kitaev](https://arxiv.org/abs/quant-ph/0403025). Columns are
  the nonzero four-bit vectors. The logical row is all ones; the four even rows
  are the coordinate bits.
- A fixed 52-CNOT decoder exposes ten Z syndromes. Undoing that decoder,
  applying conditional `SX` corrections and transversal S-dagger, and decoding
  again leaves four X checks and a retained T state on the first wire. The
  correction locations and circuit are fixed data; there is no code-synthesis
  framework. An independent binary-matrix and amplitude calculation checked the
  output phase for all 1,024 Z-syndrome branches.
- Allocate qubits in the entry function. A private function borrows 15 scalar
  qubits and returns a rejection bit. Structured loops feed actual retained
  child outputs into parent blocks. Cleanup removes the root's single-iteration
  loop; the other level loops and private block remain in QC/QCO.
- Keep rejection in a local CBit register. After the root's T-dagger/H check,
  reset and reuse that measured qubit to return rejection as a measurement. This
  keeps the exact qubit count and works with QIR's current lack of writes from
  computed booleans into result slots
  ([qir-spec #65](https://github.com/qir-alliance/qir-spec/issues/65)).
- Compiled OpenQASM exposed a DD interpreter gap: parameterized `qco.call`
  operations reached the constant-matrix fallback. They now reuse the existing
  `func.call` interpreter, including parameter bindings, return-wire mapping,
  recursion protection, and the execution budget. Controlled composite modifiers
  retain their existing constant-matrix requirement.

## Validation

Validated on 2026-09-12 with LLVM/MLIR 23.1.0. Native checks used the configured
Clang 23 Release build with ThinLTO. The focused native and Python suites,
stubs, C++ lint, and executable documentation also passed after rebasing onto
`bb5ec85e5` from current main.

- Native benchmark references: 65 tests passed, including invalid options and
  counts, both rejection-bit values, strict JSON, manifests, and case IDs.
- Native benchmark generation: 34 tests passed. The tests sample 16 ideal shots,
  reject all 15 single and 105 double input Z errors, and report a wrong root
  state for all 35 accepted triple-error patterns. Faults are injected only into
  leaf preparation in tests. Expected syndromes come independently from XORing
  the erroneous column labels.
- Levels 2–4 generate and convert to QCO with the exact qubit counts and compact
  operation growth. Tests check parent indices, leaf-only preparation, and
  accumulated rejection. Simulation of levels 3–4 was not run.
- DD functionality: all 89 tests passed, including a new parameterized nested
  unitary-call regression covering functionality, statevectors, sampling, and
  global phase.
- Python: 108 tests cover `test/python/bench/` and
  `test/python/qdmi/test_compilation.py`. CI simulates level 1 only, using 16
  shots per execution path. Higher levels retain inexpensive generation and
  metadata checks. Level 2 simulation was validated manually and is excluded
  from CI. The CLI and JSON CTest selection also passed.
- `uvx nox -s stubs` and changed-file C++ lint passed. Repository lint passed
  with unrelated untracked audit scripts excluded from type checking through a
  temporary configuration; tracked lint settings are unchanged.
- Generated MLIR references and `uvx nox --non-interactive -s docs` passed,
  including all executable examples and generated local-link checks.

The examples in `docs/benchmarks.md` reproduce the 15-qubit execution through
`sample`, then `compile_program` and `submit_program` for both DDSIM payload
paths. Each uses 16 shots and seed 17 (`custom1=17` for QDMI), waits for the
job, and asserts `{"00": 16}`. Intermediate measurements control subsequent
gates, so the simulator executes the circuit once per shot. These examples check
a deterministic ideal output; they do not need a large statistical sample.

Shot-count measurements on 2026-09-12 used the same generated level 1 program,
reused the compiled QDMI payloads, and took the median of three runs per case.
Every histogram matched its ideal output. Generation took 0.014 s; target
compilation took 0.070 s for Adaptive QIR and 0.038 s for OpenQASM 3.

| Path                        | 256 shots | 16 shots |
| --------------------------- | --------- | -------- |
| Direct DD sampling          | 4.163 s   | 0.190 s  |
| DDSIM, Adaptive QIR bitcode | 2.164 s   | 0.181 s  |
| DDSIM, OpenQASM 3           | 2.335 s   | 0.155 s  |

The QDMI timings include submission and execution but exclude target
compilation. These are shared-host observations, not performance guarantees.
Adaptive jobs are validated through counts and do not expose an uncollapsed
statevector; see `docs/qdmi/ddsim_device.md`.

Manual level 2 validation used 225 qubits and 256 shots per path with seed 17.
Direct sampling returned `{"00": 256}` in 225.325 s, DDSIM Adaptive QIR returned
`{"00": 256}` in 175.874 s, and DDSIM OpenQASM 3 returned `{"00": 256}` in
262.640 s. These individual runs used the implementation based on `383827e64`,
before the rebase, and are not CI tests.

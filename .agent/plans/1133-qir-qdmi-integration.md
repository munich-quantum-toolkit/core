# Complete the target-aware QIR-to-QDMI path

Status: historical implementation record.

## Goal and scope

MQT Core already compiles OpenQASM and MLIR programs to textual LLVM IR or LLVM
bitcode and executes either form through the DDSIM QDMI device. After this
change, the same end-to-end workflow also works when compilation uses the
`CompilerTarget` snapshot of that QDMI device. All-to-all targets need neither
heuristic placement nor routing, so target compilation deterministically uses
the target's first sites instead of randomly assigning the program to arbitrary
simulator site IDs.

The behavior is visible from Python by opening `mqt.ddsim.default`, snapshotting
it as a compiler target, compiling a Bell program to QIR Base Profile bitcode,
submitting that bitcode to the same device, and observing a correct Bell
histogram over 1,024 completed shots.

## Constraints

- Most of issue #1133 is already implemented on `main`. The repository contains
  the DD-based QIR runtime and runner, QIR Base and Adaptive compiler lowering,
  textual and bitcode QDMI formats, FoMaC byte submission, and Python tests that
  compile and execute both payload forms.

- the existing Python integration tests compile without a target. A target-aware
  probe snapshots DDSIM's 65,535 sites, maps two program qubits to arbitrary
  high IDs, and traps during QIR execution. Evidence: the failing LLVM IR used
  static qubit IDs 18,449 and 2,472 instead of compact IDs zero and one.

- the `CompilerTarget` contract already states that an absent topology means
  all-to-all connectivity. This case requires no routing, but the mapping pass
  still has to replace logical allocations with physical `qco.static` sites for
  final target-conformance verification.

- removing the mapping pass for all-to-all targets was therefore not viable. The
  focused compiler test immediately exposed the missing static placement,
  allowing the implementation to remain in Mapping rather than adding a pipeline
  special case.

- `docs/qdmi/ddsim_device.md` still claims that DDSIM accepts only OpenQASM 2
  and 3 even though its device properties and tests include QIR Base and
  Adaptive Profile strings and modules.

## Decisions

- do not add another compiler, FoMaC, QDMI, or command-line convenience API.
  Rationale: the existing `CompilerTarget.from_device`, `compile_program`,
  `QIRProgram`, and `Device.submit_job` values already form a direct typed
  workflow; another facade would duplicate ownership and format boundaries.

- retain physical placement for every target, but use the identity dense layout
  when `CompilerTarget::hasExplicitTopology()` is false. Rationale: final
  conformance requires physical `qco.static` sites, while an all-to-all device
  needs no SABRE refinement or routing. Mapping program indices to the first
  target sites is deterministic, compact, and preserves explicit site
  identifiers. Explicit-topology targets retain the existing heuristic behavior.

- strengthen the existing Python QIR execution test instead of adding another
  executable or subprocess harness. Rationale: it already crosses the compiler,
  binding, registry, FoMaC, QDMI, JIT, runtime, and result boundaries; supplying
  the device snapshot as its compiler target closes the missing seam without
  test bloat.

## Outcome and validation

Topology-free targets use deterministic physical placement without layout
refinement or routing. The existing target-aware compiler/QIR/DDSIM workflow
exercised the full integration; no new facade was needed. Focused and full
relevant suites, strict documentation, and repository lint passed. Implemented
in PR `#2007`.

## Code and ownership

`mlir/include/mlir/Compiler/Target.h` defines `mlir::CompilerTarget`. An absent
coupling topology represents all-to-all connectivity, while an absent operation
set means every operation is native.

`mlir/lib/Compiler/TargetCompilation.cpp` composes the canonical target
pipeline. It currently adds multi-qubit decomposition, generic optimization,
two-qubit fusion, mapping, cleanup, native synthesis, and final conformance in
that order. The mapping pass remains necessary for physical placement even when
its routing work is unnecessary for an all-to-all target.

`mlir/lib/Dialect/QCO/Transforms/Mapping/Mapping.cpp` performs both initial
placement and routing. Its random layout spans every target site. This is
appropriate for explicit hardware topology, but an all-to-all simulator with
65,535 sites can therefore receive arbitrary high static site IDs even for a
two-qubit program.

`test/python/fomac/test_fomac.py` already compiles QIR and executes it through
the DDSIM QDMI device in textual and binary form. The textual test will snapshot
the same device as a compiler target before compilation, providing the missing
end-to-end regression.

`docs/qdmi/ddsim_device.md` documents the simulator provider. It must list its
actual QIR formats and show the target-aware compile-and-submit workflow without
duplicating the lower-level payload contract in `docs/qir/index.md`.

## Acceptance

The focused C++ regression must prove that target compilation assigns an
all-to-all program to the target's first physical sites without SWAPs. Existing
explicit-topology tests must continue to exercise heuristic mapping.

The focused Python regression must open `mqt.ddsim.default`, snapshot that same
device as a `CompilerTarget`, compile a Bell program to QIR Base Profile, submit
the textual LLVM IR through QDMI, wait successfully, and return exactly the
requested number of shots. The existing binary test must continue to pass.

Strict documentation must describe QASM and all four QIR formats accurately,
including the Base-only restriction for QIR state extraction. Repository lint
and `git diff --check` must pass. A fresh independent review must find no new
target semantic, QIR, packaging, or test-cohesion issue before publication.

# Implement exact payload capabilities across QDMI and MQT Core

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

After this work, a QDMI device can identify an accepted payload by its format
family, version, profile, and encoding and can report the runtime features that
it accepts for that exact payload. MQT Core selects one such payload, stores the
complete target environment in MLIR, applies ordinary compiler transformations,
and rejects only the unsupported features that remain. The same metadata can
then be projected into Qiskit and PennyLane without claiming features that their
selected serializers cannot emit.

The user can observe this through three end-to-end behaviors. A QIR 2.1 Base and
Adaptive payload from one device have distinct profiles. A static loop can
compile for a target without runtime iteration after MLIR unrolls it, while a
residual dynamic loop fails. Qiskit and PennyLane expose dynamic behavior only
when the exact payload selected at device construction supports it.

## Progress

- [x] (2026-08-22 17:20Z) Refreshed both pull-request heads, preserved immutable
      backup refs, and merged current Core `main` into PR #2162.
- [x] (2026-08-22 18:15Z) Rewrote QDMI PR #508 around exact payload descriptors,
      closed per-descriptor feature queries, and payload-declared results.
- [x] (2026-08-22 18:35Z) Added Core's exact payload descriptor and adapted QDMI
      1.4 metadata without format-based capability inference.
- [x] (2026-08-22 19:05Z) Materialized the complete target snapshot in
      `mqt.target_env`; the standard mapping, synthesis, and conformance
      pipeline reads it from IR.
- [x] (2026-08-22 19:20Z) Replaced the custom target analyzer with MLIR
      normalization, bounded SCF unrolling, call-graph reachability, and
      dialect-conversion legality.
- [x] (2026-08-22 19:45Z) Moved QIR 2.1 feature derivation and validation before
      LLVM translation; translation only serializes the derived string tuples
      and repairs scalar widths.
- [x] (2026-08-22 18:40Z) Projected the selected profile into Qiskit and
      PennyLane.
- [x] (2026-08-22 20:20Z) Ran focused and complete C++ and Python tests,
      generated stubs, built executable documentation, and passed lint. The link
      checker reaches one pre-existing unreleased documentation URL that returns
      404.
- [x] (2026-08-22 20:40Z) Published signed commits to both existing pull
      requests, updated their descriptions and metadata, and fixed QDMI's hosted
      Doxygen and Clang-Tidy findings. QDMI's final required jobs pass; Core's
      platform matrix is running on the final dependency revision.

## Surprises & Discoveries

- Observation: Core PR #2162 had diverged from current `main`, which now
  contains overlapping OpenQASM and Qiskit work. Evidence: merging `origin/main`
  added three completed ExecPlans and the corresponding compiler changes without
  a conflict.
- Observation: the current Core implementation does not consume QDMI PR #508. It
  remains pinned to QDMI 1.3.3 and infers a QIR Adaptive profile from legacy
  format presence.
- Observation: MLIR 22 already provides
  `LoopLikeOpInterface::getStaticTripCount`, `scf::loopUnrollFull`,
  region-branch canonicalization patterns, SCCP, symbol DCE, and
  `getUsedValuesDefinedAbove`. The custom trip-count, reachability, and
  region-depth engines are not required.
- Observation: MLIR 22's `LLVM::ModuleFlagAttr` accepts only integer and string
  values for unknown keys, while QIR requires string tuples for integer and
  floating-point computation flags. The MLIR pass therefore derives and
  validates the type lists, stores them as QIR module attributes, and the
  translation boundary serializes those lists without rescanning LLVM IR.

## Decision Log

- Decision: Preserve both existing pull requests and Simon Hofmann's authorship.
  Rationale: Simon implemented the originally agreed contract and supplied
  useful operations and tests; the contract changed after ecosystem study.
  Date/Author: 2026-08-22, Codex.
- Decision: QDMI 1.4 uses one fixed exact descriptor and string feature records
  instead of extending numeric enums. Rationale: exact identity avoids
  version/profile inference and string features can grow without changing the C
  ABI. Date/Author: 2026-08-22, Codex.
- Decision: A descriptor-specific feature query has only two metadata states:
  unsupported means unknown, and success returns the complete optional set,
  which may be empty. Rationale: this removes the `NONE` sentinel and
  contradictory per-row completeness fields. Date/Author: 2026-08-22, Codex.
- Decision: Multi-program jobs are independent follow-up work and do not block
  capability negotiation. Rationale: Core, Qiskit, and PennyLane can use the
  existing one-program job path, so batching would enlarge the critical change
  without enabling the target contract. Date/Author: 2026-08-22, Codex.
- Decision: Core uses MLIR dialect conversion for residual legality but ordinary
  canonicalization and SCCP for normalization. Rationale: program requirements
  change after rewrites; a one-time preflight set is not a proof of
  legalizability. Date/Author: 2026-08-22, Codex.
- Decision: Keep explicit-target pass factory overloads for focused pass tests
  and source compatibility, but make the standard and textual pass path read the
  complete target snapshot from `mqt.target_env`. Rationale: this achieves
  reproducible pipelines without forcing unrelated test rewrites. Date/Author:
  2026-08-22, Codex.

## Outcomes & Retrospective

The implementation now follows the three-layer contract: QDMI reports accepted
semantics for an exact payload; MLIR derives residual program requirements and
legalizes supported alternatives; SDK adapters project only the selected
producer path. The old recursive preflight analyzer and its
implementation-specific tests were deleted. Static control is normalized,
bounded loops are unrolled, multiway QCO branches lower to nested forward
branches, and failed compilation remains transactional. The QDMI branch has been
pushed with signed commits. Core passed 4,326 CTest cases, 148 compiler tests,
344 focused Python tests, 42 PennyLane tests, generated-stub checks, executable
documentation, and all lint hooks. The link checker found only the existing
unreleased Qiskit-backend documentation URL. Both existing pull requests now
contain the implementation and describe the final contract. QDMI's hosted matrix
passes. Core's hosted matrix is the only remaining external validation.

## Context and Orientation

QDMI is a C interface in a separate repository. Its `include/qdmi/constants.h`
currently represents program formats as numeric enum values that combine a
format, profile, and encoding. PR #508 adds another numeric enum for runtime
features and returns relation records through a generic device property. That
shape cannot identify QIR 2.1 independently from the existing QIR 1 contract,
cannot report integer or floating-point widths, and can represent contradictory
completeness states.

MQT Core wraps QDMI in `include/mqt-core/qdmi/Client.hpp` and
`src/qdmi/Client.cpp`. The MLIR compiler target is declared in
`mlir/include/mlir/Compiler/Target.h`. PR #2162 adds
`mlir/include/mlir/Compiler/ProgramFormat.h`, a typed target attribute in
`mlir/include/mlir/Dialect/MQT/IR/MQTDialect.td`, and target compilation in
`mlir/lib/Compiler/TargetCompilation.cpp`. The target compilation file is more
than two thousand lines because it reimplements constant-loop evaluation,
reachability, classical-bit provenance, feedback provenance, type discovery, and
quantum capture checks. The final dialect conversion has no rewrite patterns and
therefore only verifies the custom analysis result.

A payload descriptor is the exact tuple accepted at job submission: format ID,
version, optional profile, and text or binary encoding. A capability is a string
ID and a numeric value. Boolean capabilities use value zero. Integer and
floating-point capabilities repeat the ID for each supported width. A normative
baseline is the behavior guaranteed by one standardized descriptor. The
effective profile is that baseline plus the complete optional list reported by
the device. When a device cannot report the optional list, Core retains that
fact for diagnostics and conservatively assumes only the baseline.

`mqt.target_env` is a typed MLIR attribute on `module`. It must contain the
selected exact payload and the immutable target snapshot so that passes can be
constructed from a textual pipeline without hidden C++ objects. It implements
`DLTIQueryInterface`, the generic MLIR query interface for target information;
it is not a memory data-layout specification.

## Plan of Work

First, revise QDMI PR #508. Replace the legacy format enum at the public job and
device boundaries with a C11-compatible fixed descriptor containing a packed
version, encoding, fixed format ID, and fixed profile ID. Add a fixed string
feature record and a descriptor-specific two-call query to both client and
device interfaces. `QDMI_ERROR_NOTSUPPORTED` means optional metadata is unknown;
success returns a complete list. Define QIR 2.1 Base and Adaptive descriptors
and the initial feature IDs. Clarify that SHOTS and histogram keys contain
payload-declared flat bit outputs and add a format-native program-output result.
Remove calibration and batch pseudo-formats from the payload enum and remove the
calibration-need property. Update the example, device template, tests,
changelog, and upgrade guide.

Second, update Core to consume the QDMI contract. Introduce one context-free
payload descriptor and capability record. Remove `QIRProfile` and profile
variants from `ProgramFormat`. Replace the target factory overload matrix with
one validated description. Change the QDMI adapter to map descriptors and query
their optional features; remove every feature inference based only on format
presence.

Third, complete `mqt.target_env`. Store the selected descriptor, effective
capabilities, metadata completeness, sites, couplings, operations, and timing
unit. Materialize it on a compilation copy before running passes. Refactor
mapping, native synthesis, and conformance passes to read that attribute, then
remove target parameters from pass constructors and pipeline population
functions.

Fourth, replace `TargetCompilation.cpp`. Run symbol DCE, SCCP, QCO cleanup, and
region-branch canonicalization first. Use `LoopLikeOpInterface` and
`scf::loopUnrollFull` to erase constant counted loops when the payload lacks
runtime iteration. Bound only the estimated expanded operation count. Add a
small operation interface that returns instance-dependent capability records;
provide external models for SCF and arithmetic operations. Use dynamic legality
and real rewrite patterns to lower multiway branching to forward branches when
possible and to reject unsupported residual operations. Reuse QCO's
`WireIterator` and SSA use lists for measurement and qubit requirements. Use a
standard MLIR dataflow analysis only if a remaining non-local case proves that
one is necessary.

Fifth, move QIR feature derivation into the MLIR LLVM-dialect pipeline. Derive
integer and floating-point widths, functions, branch modes, multiple returns,
arrays, and dynamic allocation before translation. Compare them with the
selected QIR 2.1 profile. LLVM translation then serializes the validated module
flags without discovering or deleting capabilities.

Finally, update the SDK plugins. Qiskit selects one exact descriptor and
serializer at backend construction and adds its native control-flow instruction
classes from that profile. PennyLane selects one OpenQASM 3 descriptor and
advertises one-shot mid-circuit measurement only when both the profile and its
converter support the path. Its transform order is split-non-commuting first,
then dynamic-one-shot for every split tape.

## Concrete Steps

From the QDMI repository root, iterate with:

    cmake --preset release
    cmake --build --preset release -j2
    ./build/release/test/qdmi_test --gtest_filter='*ProgramFormat*:*ProgramFeature*:*Result*'
    uvx prek run -a

From the Core repository root, iterate with:

    cmake --preset release
    cmake --build --preset release --target mqt-core-mlir-unittests-compiler -j2
    ./build/release/mlir/unittests/Compiler/mqt-core-mlir-unittests-compiler --gtest_filter='CompilerTargetTest.*:CompilerPipelineTest.TargetCompilation*'
    uv run --no-sync pytest test/python/test_mlir.py test/python/qdmi test/python/test_mlir_qiskit_translation.py

After bindings change, run:

    uvx nox -s stubs

Before handoff, run the complete QDMI and Core suites, build both documentation
sets, run `git diff --check`, and finish with `uvx nox -s lint` in Core and
`uvx prek run -a` in QDMI. Record every failure that is unrelated or cannot run
locally.

## Validation and Acceptance

QDMI acceptance requires exact descriptors to round-trip through discovery and
job submission. QIR Base text, QIR Base binary, QIR Adaptive text, and QIR
Adaptive binary are four independent descriptors. An unsupported feature query
is observable as unknown; a successful empty query is observable as baseline
only. Width records and format-native output survive the two-call APIs.

Core acceptance requires `mqt.target_env` to round-trip through textual MLIR and
all target passes to run without captured target objects. Static branches and
bounded loops compile for a target without the corresponding runtime feature. A
residual dynamic loop fails with a capability diagnostic. A multiway branch
lowers to forward branches when supported. Failed compilation leaves the source
program unchanged. QIR 2.1 flags are present before LLVM translation and match
the selected descriptor.

SDK acceptance requires Qiskit to expose control-flow classes only for its
selected serializer and PennyLane to advertise one-shot only for a selected
OpenQASM 3 path that can emit and reconstruct all required bits. Metadata from
another accepted descriptor must never leak into either SDK target.

## Idempotence and Recovery

The reviewed heads are preserved under local backup refs before modification.
Builds and tests write only to ignored build directories. Ordinary source edits
are repeatable. Do not discard unrelated worktree changes. Before any remote
update, fetch the exact remote head again, verify every new commit signature,
and push with a normal fast-forward. If branch history must be rewritten, create
another backup and use an exact `--force-with-lease` value; never use an
unqualified force push.

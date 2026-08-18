# Backport the remaining compatible main changes to v3.x

This ExecPlan is a living document. The sections `Progress`,
`Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must
be kept up to date as work proceeds.

This ExecPlan must be maintained in accordance with `.agent/PLANS.md` from the
repository root.

## Purpose / Big Picture

MQT Core 3.9.0 should receive the compatible fixes, QDMI functionality,
PennyLane cleanup, and maintenance merged into `main` after the previous
combined v3 backport. After this work, a QDMI client can start calibration jobs,
the OpenQASM parser handles failed assignment expressions safely and preserves
the difference between scalar qubits and one-element registers, and the
PennyLane plugin converts programs through one private converter per opened
device session. The maintained branch also receives the applicable dependency,
continuous-integration, audit, and writing-guidance updates.

The observable result is a draft pull request against `v3.x` with signed,
provenance-preserving commits. Its native QDMI and OpenQASM tests, Python QDMI
and PennyLane tests, supported Python sessions, documentation, and lint checks
must pass.

## Progress

- [x] (2026-08-18 22:24Z) Refreshed `main`, `v3.x`, pull requests #2153 and
  #2157, GitHub authentication, the worktree, and commit signing.
- [x] (2026-08-18 22:24Z) Selected compatible changes and recorded all v3
  exclusions.
- [ ] Port compatible maintenance and the SpecAudit method from #2121, #2122,
  #2123, #2124, #2126, #2128, #2129, and #2130.
- [ ] Port all non-merge commits from #2147 and adapt the final design to the
  v3 PennyLane plugin without weakening its behavior.
- [ ] Port QDMI calibration jobs from #2148 and OpenQASM fixes from #2156 and
  #2157.
- [ ] Regenerate stubs and the lockfile, then complete focused and aggregate
  validation.
- [ ] Publish a draft pull request against `v3.x`, apply repository metadata,
  and inspect checks for the exact head.

## Surprises & Discoveries

- Observation: `v3.x` already contains the automatic backports of #2141,
  #2142, and #2146. Evidence: its current first-parent history ends in pull
  requests #2144, #2143, and #2151.
- Observation: The automatic #2145 backport remains a separate open pull
  request. Evidence: pull request #2153 targets `v3.x` and is not merged.
- Observation: Pull request #2157 is open but its two substantive commits are
  available independently of its merge-from-main commit. Evidence: commits
  `6eaea2ad` and `0009e392` contain the implementation and changelog entry.
- Observation: The product and test changes from #2147 apply to the v3 plugin,
  but its reconciled audit ledger requires #2124 first. Evidence: #2147 modifies
  `.agent/audits/pennylane-plugin.md`, which #2124 creates.

## Decision Log

- Decision: Use one combined pull request with separate signed commits.
  Rationale: This limits stable-branch CI use while preserving upstream
  authorship, review boundaries, and revert boundaries. Date/Author: 2026-08-18
  / Codex.
- Decision: Port #2147 in full, including the removal of public
  `ConvertedProgram` and `convert_program`. Rationale: The maintainer explicitly
  requested the complete SpecAudit result for v3. Date/Author: 2026-08-18 /
  maintainer and Codex.
- Decision: Port #2124 before #2147. Rationale: The full #2147 result includes
  the audit reconciliation and therefore needs the audit method, probe, and
  original ledger. Date/Author: 2026-08-18 / Codex.
- Decision: Port #2157 even if it has not merged when implementation begins.
  Rationale: The maintainer explicitly included it; its substantive commits are
  stable and exclude only a branch-sync merge. Date/Author: 2026-08-18 /
  maintainer and Codex.
- Decision: Do not duplicate #2145. Rationale: Pull request #2153 already owns
  that backport. Date/Author: 2026-08-18 / maintainer and Codex.
- Decision: Regenerate `uv.lock` from v3 declarations. Rationale: The main
  lockfile represents a different compiler and dependency graph. Date/Author:
  2026-08-18 / Codex.

## Outcomes & Retrospective

Implementation, validation, and publication are in progress. Update this
section after each major milestone with exact test results, remaining gaps, and
the draft pull-request URL.

## Context and Orientation

`v3.x` is the maintained MQT Core 3 release line. It retains optional LLVM and
MLIR 21 support and the native FoMaC C++ implementation. Python exposes that
implementation through the `mqt.core.qdmi` namespace. `main` contains the v4
compiler architecture and uses the native QDMI C++ namespace, so a backport must
adapt QDMI code to `include/mqt-core/fomac/FoMaC.hpp` and
`src/fomac/FoMaC.cpp` without importing v4 compiler layers.

The PennyLane plugin lives under `python/mqt/core/plugins/pennylane/`, with tests
under `test/python/plugins/qdmi_pennylane/`. The full #2147 change makes the
converter private, binds it to one opened QDMI session, reads advertised gates
once, and tests conversion through `QDMIDevice` instead of a public helper.

The OpenQASM parser lives under `include/mqt-core/qasm3/` and `src/qasm3/`, with
its regression tests in `test/ir/test_qasm3_parser.cpp`. Pull request #2156
guards assignment type checking after a failed right-hand expression. Pull
request #2157 distinguishes scalar `qubit q;` declarations from registers such
as `qubit[1] q;`.

## Plan of Work

Apply the maintenance pull requests as provenance-preserving commits, omitting
only #2121's Qiskit C API wheel-source paths because those files do not exist on
v3. Apply #2124 next so its audit ledger exists before #2147. Apply the 17
non-merge #2147 commits in their original order, resolve differences against
the v3 plugin, and add a v3 upgrade note for callers of the removed public
converter API.

Adapt #2148 to `fomac::Device`. Add optional binary and text calibration-job
entry points, bind them as `submit_calibration_job`, keep custom job parameters,
and do not set a shot count. Keep generic calibration and batch submission
unavailable with explicit diagnostics. Regenerate the Python stub rather than
editing it manually.

Apply #2156 before #2157 because both update OpenQASM type checking. If #2157
has merged, use its resulting main commit. Otherwise use its two substantive
commits and exclude the merge-from-main commit. Resolve all changelog entries
semantically, preserve every human contributor, and regenerate the final v3
lockfile.

Do not port the v4-only compiler, QCO, QIR, or removal changes from #1973,
#2054, #2111, #2112, #2114, #2115, #2118, #2119, #2125, #2127, #2133, #2136,
#2137, #2138, #2140, or #2154. Do not port #2120 because checked-in v3 ExecPlans
still depend on its worktree-local command wrapper.

## Concrete Steps

Run commands from the repository root. Apply each upstream commit with signing
and provenance, inspect its full commit message, and verify the result:

    git cherry-pick -S -x <upstream-commit>
    git show --format=fuller --stat HEAD
    git verify-commit HEAD

Regenerate bindings and dependency metadata after source changes:

    ./.agent/run.sh uvx nox -s stubs
    ./.agent/run.sh uv lock
    ./.agent/run.sh uv lock --check

Place LLVM 21.1.8's executable directory on `PATH`, then configure, build, and
test the non-MLIR release preset:

    ./.agent/run.sh cmake --preset release
    ./.agent/run.sh cmake --build --preset release
    ./.agent/run.sh ctest --preset release

Run focused and supported Python validation:

    ./.agent/run.sh uv run --no-sync pytest test/python/qdmi \
      test/python/plugins/qdmi_pennylane -q
    ./.agent/run.sh uvx nox -s tests-3.10
    ./.agent/run.sh uvx nox -s tests-3.14
    ./.agent/run.sh uvx nox -s minimums-3.12 -- \
      test/python/plugins/qdmi_pennylane

Run repository-wide validation:

    ./.agent/run.sh uvx nox --non-interactive -s docs
    ./.agent/run.sh uvx nox -s lint
    git diff --check origin/v3.x...HEAD

## Validation and Acceptance

The QDMI tests must prove that calibration jobs reach the device with absent,
text, empty, and binary payloads without a shot count. Generic submission must
direct calibration callers to the dedicated entry point and must state that
batch jobs are unsupported.

The PennyLane tests must prove exact parameter values, unordered shot
multiplicity, exact accumulated execution time, QASM3-first format selection,
QASM2 fallback, advertised site and coupling-map validation, and one operation
table read per opened session. The package must no longer export
`ConvertedProgram` or `convert_program`.

The OpenQASM tests must preserve the original diagnostic after a failed
assignment expression. Scalar qubits must work as gate and measurement operands
and reject indexing, while sized registers retain their current behavior.

Acceptance also requires a complete non-MLIR release build and CTest run, the
supported Python 3.10 and 3.14 sessions, the focused minimum-dependency
PennyLane session, generated stubs, documentation, the lock check, lint, commit
signature verification, and `git diff --check`. Record exact passes, failures,
and environmental limitations.

## Idempotence and Recovery

All build, test, stub, lock, documentation, and lint commands are repeatable.
If an upstream commit conflicts, resolve only files in that commit and continue
the signed cherry-pick. Do not reset or discard unrelated work. Regenerate
derived files from their declarations instead of splicing generated output.

Before rewriting any published history, create a backup ref, record the remote
head, and use an exact `--force-with-lease` guard. Never merge the resulting
pull request; a human reviews and merges it.

## Artifacts and Notes

The scan starts after combined backport #2117. Automatic backports #2141,
#2142, and #2146 are already present. Pull request #2153 remains responsible for
#2145. Pull requests #2124, #2147, #2148, #2156, and #2157 are the main
functional sources for this follow-up.

## Interfaces and Dependencies

The final C++ API adds `fomac::Device::submitCalibrationJob` overloads for an
optional byte span and a text payload. The final Python API adds
`mqt.core.qdmi.Device.submit_calibration_job`. Both forms accept the existing
five optional custom job parameters and omit a shot count.

The PennyLane package retains `QDMIDevice` as the public execution surface and
removes `ConvertedProgram` and `convert_program`. Conversion remains private and
bound to one opened QDMI session. No new dependency is added. The existing
PennyLane optional-dependency and Python 3.10 boundaries remain unchanged.

Revision note (2026-08-18): Created after refreshing the live branches and pull
requests. The scope includes the maintainer's explicit decisions to port #2147
in full and to include #2157 before its merge.

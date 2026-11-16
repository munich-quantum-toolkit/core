# Called-function measurements in QCO sampling

Status: proposed follow-up; no implementation authorized by this plan alone.

## Evidence and scope

The [DD performance investigation](dd-simulation-performance.md) measures the
supported level-one magic-state-distillation generator at 10,000 shots. QCO
takes the called-function per-shot fallback: median 1.43 s, no garbage
collections, and 99.92% matrix-vector cache hits. The generated helpers contain
measurements, so repeated interpreter work survives effective DD caching.

The objective is to avoid that repeated work while preserving quantum state,
classical outputs, seeded repeatability within a sampling strategy, helper
argument/result correspondence, tensor ownership, and caller-program
immutability. Do not add parallel structured branching or change QIR execution.

## Investigation

First compare the same generated QCO module before and after explicit use of
MLIR's existing inliner, followed by existing cleanup. Include preprocessing and
module-copy time in the comparison. Confirm whether the resulting program uses
`sampleBranches`, then compare five runs at small and large shot counts. Use
distillation, a helper whose measured result controls a caller gate, and nested
helpers. Keep original and transformed outputs as correctness evidence.

Only if existing inlining cannot provide a small, complete solution should a
separate design consider resumable call frames in branch sampling. That design
must account for each group's classical environment, qubit/tensor maps, return
continuations, recursion protection, and DD root ownership. Do not duplicate
these contracts across frontends.

## Acceptance

Keep a change only with a material end-to-end gain on the supported generator,
no meaningful small-shot regression, and numerical/output regressions that
exercise measurements across call boundaries. Preserve the current fallback for
unsupported forms. Record baseline revisions, timing spread, RSS, group counts,
and cache/collection statistics outside the repository.

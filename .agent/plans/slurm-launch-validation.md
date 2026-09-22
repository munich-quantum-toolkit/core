# Optional Slurm launch validation

Status: implemented and validated on real Slurm; final provider image checks
remain in progress.

## Goal and scope

Add opt-in readiness validation to the shared SPANK component. Execute the MIT
QDMI checker as the job user before task launch. Keep provider code outside
Slurm daemons and preserve static injection when validation is disabled.

## Decisions

- Validate in remote `task_init`, after Slurm restores allocated licenses and
  permanently drops privileges. Return a task failure, never a node failure.
- Copy the final `S_JOB_ENV` for `execve`; do not inherit daemon credentials.
- Check once per node and step through an anonymous shared gate created before
  task forks. Compare later tasks' effective configuration with the first task's
  snapshot. Reject a mismatch without claiming that earlier tasks did not start.
- Provider configuration must be step-wide. Slurm rank metadata is excluded from
  snapshot comparison and is unsupported as provider configuration input.
- Bound environment storage, checker execution, and waiting for another task's
  result. Do not add controller RPCs, a persistent cache, or an all-task
  barrier.
- Reset and unblock the checker's parent-death signal before exec. Slurm's
  caught signal handlers remain installed in a fork child until exec resets
  them.

## Work remaining

- Complete both provider workloads with direct configuration, shared injection,
  and optional validation against the smaller native and wheel images.

## Validation

The existing runner unit tests pass (22 tests), as do full repository lint,
standalone Linux compilation, and full-file clang-tidy 23 checks. The installed
Linux checker passed its forced-parent-death case. Native configuration
correctly rejects the macOS host before compiling this Linux-only component. The
full strict documentation build passed with MLIR 23.1.0.

The full real Slurm fixture passed on Linux aarch64 with cgroup v2 and Slurm
25.11.2. It verifies hook ordering, task launch, license release, node state,
task environment, and process cleanup. Cases cover disabled validation, multiple
nodes/tasks, batch and nested steps, failures, timeouts, cancellation, differing
task inputs, and bounded storage. Static injection and provider migrations do
not depend on this optional PR.

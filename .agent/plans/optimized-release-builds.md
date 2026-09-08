# Optimized release builds

Status: implementation and local validation complete; dependency publication and
hosted validation pending.

## Goal and scope

Optimize Linux and macOS release artifacts while retaining Windows support and
portable CPU baselines. Coordinate the portable MLIR toolchain, setup action,
shared build workflows, and Core CI. Benchmark LTO and optimization of binding
translation units before selecting production defaults.

## Decisions

- Preserve the existing assertion-enabled SDK selection. Add an explicit
  assertion-free release variant with matching generated LLVM configuration; do
  not override an installed SDK's ABI settings in Core.
- Keep compiler-specific LTO intermediates out of redistributable static SDK
  libraries. Core-owned final binaries are the first LTO benchmark target.
- Keep PGO and post-link optimization exploratory until representative training
  and independent evaluation workloads demonstrate a worthwhile improvement.
- Preserve QDMI device and replaceable-driver shared-library boundaries when
  assessing wheel packaging. The v4.1 driver stack starts at Core #2229.
- Preserve the small C++ SDK. Enable ELF section garbage collection on the DDSIM
  device and benchmark executable: the local wheel shrinks by 29.3%.
- Enable full LTO for Linux and macOS Core release wheels and keep binding
  optimization defaults. Preserve the earlier runtime measurements as a reason
  to re-evaluate with the optimized SDK. Native SDK archives need a coordinated
  compiler policy before LTO.

## Work remaining

- [x] Add the portable SDK variant and validate setup selection locally.
- [ ] Publish the new SDK archives and finish hosted Core matrix validation.
- [x] Benchmark LTO and computational bindings; remove obsolete `/Zm10`.
- [x] Enable persistent wheel compiler caching and select mold on Linux.
- [x] Trace installed wheel consumers and quantify packaging opportunities.
- [x] Review Astral's PGO/BOLT pipelines and record an MQT training strategy.
- [x] Publish Core #2476 and companion PRs: toolchain #94, setup #255, workflows
      #464.

## Validation

Use each repository's existing lint and tests. Compare benchmark variants from
identical source revisions and representative held-out workloads. Validate
installed wheels, including bundled QDMI devices and driver loading, rather than
relying solely on build-tree tests. Distinguish local evidence from hosted
cross-platform validation and unavailable release artifacts.

Detailed measurements, distribution contracts, and the PGO/BOLT follow-up
strategy are in [the audit](../audits/optimized-release-builds.md).

Regular C++ and Python CI uses assertion-enabled SDKs. The CD wheel jobs,
including their pull-request checks, use the assertion-free SDK. Windows wheel
LTO is disabled. Linux wheel and SDK image digests match cibuildwheel 4.2.0 and
must advance together with an SDK release.

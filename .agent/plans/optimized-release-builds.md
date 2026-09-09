# Optimized release builds

Status: native assertion-free SDK and Core-only LTO/BOLT validated locally. SDK
publication and hosted matrix validation remain external gates.

## Goal and scope

Optimize Linux and macOS release artifacts while retaining Windows support and
portable CPU baselines. Coordinate the portable MLIR toolchain, setup action,
shared build workflows, and Core CI. Measure full LTO and Linux BOLT with the
release SDK; retain the earlier binding optimization comparison as a separate
experiment.

## Decisions

- Preserve the existing assertion-enabled SDK selection. Add an explicit
  assertion-free release variant with matching generated LLVM configuration; do
  not override an installed SDK's ABI settings in Core.
- SDK variants contain native libraries. Core release wheels enable LTO for
  their own code and BOLT after final linking on Linux.
- Use the cibuildwheel 4.2.0 manylinux image tag in the SDK; let Core follow
  cibuildwheel defaults and macOS use the runner's default Xcode.
- Use mold for native SDK links. Keep BFD for Core BOLT links: the released mold
  has a reproduced relocation bug; its upstream fix plus `--no-relax` works
  locally but would add maintenance.
- BOLT runs after final linking, with fresh instrumentation profiles and
  `-lite`, before LLVM stripping, wheel repair, and metadata generation. The
  repaired wheel passes training again. Local held-out gains are 1.4-3.4%, with
  a 9.5% smaller wheel than the matching LTO baseline.
- Keep compiler PGO exploratory. Validate Linux BOLT with bounded training and
  independent evaluation workloads before claiming a performance gain.
- Preserve QDMI device and replaceable-driver shared-library boundaries when
  assessing wheel packaging. The v4.1 driver stack starts at Core #2229.
- Preserve the small C++ SDK. Enable ELF section garbage collection on the DDSIM
  device and benchmark executable: the local wheel shrinks by 29.3%.
- Enable full LTO for Linux and macOS Core release wheels and keep binding
  optimization defaults. Preserve the earlier runtime measurements as a reason
  to re-evaluate with the optimized SDK. SDK archives carry native code and do
  not require an exact consumer compiler match.

## Work remaining

- [x] Validate the native assertion-free SDK with Core LTO/BOLT and compare
      held-out runtime against the fat SDK.
- [x] Reproduce mold's symbol-index bug and test its upstream fix locally.
- [x] Validate fat-LTO archives with native and LTO consumers and Core CD.
- [x] Measure selected distribution components locally without narrowing the
      published SDK tool inventory.
- [x] Add the portable SDK variant and validate setup selection locally.
- [x] Validate matched SDK LTO and BOLT end to end with a manylinux Core wheel.
- [x] Compare BOLT runtime and artifact size on held-out workloads.
- [x] Compare the assertion-enabled policy against assertions-off plus LTO/BOLT
      with matching GCC 14 builds: 9.1-19.2% less time on held-out workloads.
- [ ] External gate: publish the new SDK archives and finish hosted Core matrix
      validation after the companion changes land.
- [x] Benchmark LTO and computational bindings; remove obsolete `/Zm10`.
- [x] Enable persistent wheel compiler caching; use GNU ld for BOLT releases.
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
LTO is disabled. Linux SDK image pins and Core cibuildwheel updates are
independent.

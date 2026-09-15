# Shared static Slurm deployment

Status: in progress; implementation and both provider integrations need
validation.

## Goal and boundaries

Implement Core #2366 and the static deployment guide in #2312, under #2360. Core
owns the Slurm fixture, configuration injection, and scheduler documentation.
Providers retain catalogues, authentication, mock services, and SDK adapters.
The optional checker and launch validation have separate PRs.

## Decisions

- Target `v4.1`, ordinary QDMI 1.3, and Slurm 25.11 or newer.
- Build `spank/` independently using Slurm headers and a C++20 compiler.
  Preserve its GPL notices and exclude it from MIT Python distributions.
- Configure concrete license IDs and administrator-declared reference mappings.
  Explicit options override submitted environment, which overrides site
  defaults. Read only job environment values; do not resolve credentials in
  Slurm.
- Reuse `test/slurm` for both provider integrations. Provider-specific services
  consume the shared cluster rather than copying scheduler setup or tests.
- Preserve the Slurm adapter's existing selection and IDLE/BUSY semantics.
  Selection does not attest an allocation or authorize provider access.

## Work remaining

- [ ] Standalone injection build, installed license, wheel/sdist separation.
- [ ] Shared fixture and transport regression tests.
- [ ] Braket and IQM native/wheel catalogue and SDK integration tests.
- [ ] Canonical guide, focused lint, and final base-to-head diff review.

## Validation

Use the existing Core Slurm runner and both provider suites against this
checkout. Run the isolated SPANK build without Core or LLVM configuration.
Inspect wheel/sdist inventories and preserve the existing fixture test
scenarios. Record actual command results when completed; real Slurm validation
is required.

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

- [ ] Standalone Linux injection build and installed license.
- [ ] Real Slurm transport, daemon isolation, and node-state checks.
- [ ] Braket and IQM native/wheel catalogue and SDK integration tests.
- [ ] Required C++ lint with clang-tidy 23.

## Validation

The shared runner's 22 tests and full `uvx nox -s lint` passed. The source-built
wheel and source archive exclude SPANK sources and binaries. The canonical guide
and full base diff were reviewed.

The full strict documentation build passed with MLIR 23.1.0. Link checking
failed only on a timeout for the unchanged VS Code Marketplace link in
`contributing.md`.

Run `test/slurm/run_integration.py` and both provider workloads for the
remaining Linux checks. The standalone SPANK build must not configure Core or
LLVM. The transport test keeps a batch task alive while inspecting both Slurm
daemon environments, preserving the former provider test's isolation proof. Real
Slurm validation remains required.

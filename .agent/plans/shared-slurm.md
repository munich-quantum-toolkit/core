# Shared static Slurm deployment

Status: complete.

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
- Keep the QDMI reference size limit separate from Slurm's allocation metadata.
  A long license list must not reject an unrelated job or hide a matching ID.
- Discard temporary provider build output and package caches before saving the
  image. Providers supply required CMake options through the shared build.

## Validation

The shared runner's 22 tests, full `uvx nox -s lint`, standalone Linux build,
installed license check, and full-file clang-tidy 23 checks passed. The
source-built wheel and source archive exclude SPANK sources and binaries. The
canonical guide and full base diff were reviewed.

The full strict documentation build passed with MLIR 23.1.0. Link checking
failed only on a timeout for the unchanged VS Code Marketplace link in
`contributing.md`.

The full real Slurm suite passed on Linux aarch64 with cgroup v2 and Slurm
25.11.2. It covers admission, license release, environment precedence, daemon
isolation, node state, and license lists larger than 4 KiB. The standalone SPANK
build does not configure Core or LLVM. The transport test keeps a batch task
alive while inspecting both Slurm daemon environments.

Braket and IQM passed direct configuration and shared injection with both native
and wheel installations, using released Core 4.0.0 and local mock services. The
smaller images passed with Docker capped at 4 GiB RAM. Braket covers Qiskit and
PennyLane results; IQM covers Qiskit results and its existing PennyLane
unsupported-format error.

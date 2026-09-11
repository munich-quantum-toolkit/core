# README and documentation for the next release

Status: implementation and local validation complete. Compiler PR #2506 is at
`1f7defd48` on main `fce58f02d` after PR #2519.

## Goal and decisions

Refresh the README and documentation entry points for the next release and
resolve #2419. Describe six capability groups, put a runnable iterative-QPE
example near the beginning, and list MQSC before CDA/TUM without changing
funding attribution. Preserve Sphinx, MyST notebooks, and generated-link checks.

The README owns the first example. The getting-started notebook must execute
that exact source rather than maintaining a second copy. It compares standard
and iterative QPE and evaluates a non-exact phase. The benchmark catalog
includes an executed repeat-until-success demonstration and its phase-sensitive
readout. Both examples run locally on DDSIM and consume QDMI counts directly.

The landing page routes application users, DD users, device integrators,
benchmark users, and C++/compiler developers to examples, guides, and APIs.
Audit current authored docs against declarations, implementation, and execution;
keep historical changelogs and generated API prose outside individual review.

## Result

The README now describes six capability groups and leads with iterative QPE. The
landing page supplies five routes to examples, guides, and APIs. The QPE and RUS
examples execute locally; the C++ page builds its displayed example against the
installed library. Corrected claims and scope limits are recorded in
`../audits/readme-rtd-next-release.md`.

Preserve PR #2519's OpenQASM import and operation-capability names, static
gate-count scope, and jeff conversion/serialization distinction. Displayed RUS
equations use MyST `math` directives within the benchmark catalog. README and
documentation prose use `[MQSC](https://mq.sc)`; copyright notices retain the
full legal name. The glossary expands MQSC and CDA and records their joint
development of MQT Core. The renamed `mlir/mqt_compiler_collection.md` guide
introduces Python, C++, and `mqt-cc`. Navigation and incoming links use the new
path. Remove the redundant RUS README link, the device note below the
landing-page grid, and the Shor implementation aside.

The adaptive examples required a separate compiler fix, published as PR #2506.
The documentation PR is stacked on that branch and closes #2419 after merge to
main. The docs dependency group installs Ninja for the C++ example. The existing
documentation framework remains in use.

## Validation

- A fresh wheel installed in an isolated CPython 3.14.7 environment executes the
  exact README source with no stderr: `{'01100000': 1024}`, phase `3/8`.
- A clean strict HTML build with Doxygen 1.9.8 passes, including generated file
  and fragment checks. All 11 authored notebooks execute without error or stderr
  records. Standard and iterative QPE return the same counts; the RUS parity
  distribution has total variation distance 0.005 from its reference.
- The renamed compiler page and benchmark RUS anchor resolve. Old page paths are
  absent. The three moved RUS equations retain their MyST math directives and
  contents. MQSC links and the compiler page's three interface routes are
  present; rendered landing and compiler pages were inspected.
- The generated QPE notebook contains the exact README source. The isolated
  wheel smoke test uses the same source.
- The exact C++ notebook cell passes with only the docs environment and system
  executables on `PATH`. Without Ninja, configuration fails as in RtD build
  `34501147`; the notebook now includes CMake diagnostics in its output.
- External linkcheck, the two generated-link regression tests, general lint, and
  `git diff --check` pass. The final ponytail review found no further
  unnecessary code or dependencies in the documentation changes.

Compiler validation is recorded in its separate plan and audit: 922 native
tests, 315 Python tests, full-file C++ lint, general lint, and generated MLIR
docs pass.

## Publication

PR #2509 is stacked on PR #2506. Both use signed, verified commits, the
repository template and AI disclosure, the required assignee, and appropriate
labels. Hosted CI is not monitored as part of this task.

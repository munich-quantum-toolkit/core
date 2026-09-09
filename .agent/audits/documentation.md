# Documentation audit

Status: findings implemented and locally validated. Date: 2026-09-08. The audit
examined `33dbc843e589d9e9166308084e825c9f8b2ff89d`; the fixes include the
subsequent main-branch changes. Reassessed against upstream main `0c3fac2cb`
after seventeen additional PRs merged.

## Scope and evidence

Reviewed all 34 authored Markdown pages, documentation generation, contributor
recipes, notebook execution, generated navigation, and the implementation behind
disputed DD, compiler, and QDMI claims. Historical changelogs and every
generated API description were not individually audited. Baseline builds passed
despite three incorrect equations, stale notebook caching, and 77 broken
generated links.

## Dispositions

1. Corrected matrix dyads, matrix-vector row indices, and inner-product
   conjugation terms. Regenerated the matrix-vector SVG from its corrected TeX
   source. Executed complex NumPy/DD comparisons check the algebra.
2. Removed unsupported unconditional bounds for DD addition, multiplication, and
   inner products. Documented phase normalization and tolerance. An executed
   product-state example demonstrates that compact inputs can have a large sum.
3. Scoped dead-gate removal caveats to the relevant passes and corrected Qiskit
   version ranges and calibration payload/API statements. Added one canonical
   Qiskit interoperability reference.
4. Corrected LLVM/MLIR setup order and documentation build commands in the
   canonical MQT templates. Core receives the rendered pages. Native Doxygen
   replaces stale Breathe guidance only for Core.
5. Executed target compilation, compiler-to-QIR-to-DDSIM submission, parameter
   binding, batching, and primitive workflows. Inline fixtures replace missing
   source files. Benchmark path setup remains visible behind a disclosure.
6. Forced notebook execution and removed the incomplete Read the Docs path
   filter. Software changes now trigger the same documentation checks.
7. Replaced IPython shell escapes with checked subprocess calls and readable
   output. Command failures propagate to notebook execution.
8. Isolated documentation discovery with the existing QDMI file and inline
   configuration controls. Examples use packaged devices and seed DDSIM where
   repeatable sampling matters.
9. Enabled Doxygen failure on warnings, fixed calibration parameter comments,
   and derived the downloaded inventory and native links from one QDMI URL.
10. Disabled viewcode's invalid binding-source navigation, selected static
    Doxygen menus, and added a tested generated file/fragment checker. Static
    menus avoid malformed alphabetical shortcuts without rewriting HTML. A
    normal C++ landing page makes the native reference discoverable in both the
    Sphinx toctree and the generated agent index.
11. Reduced Doxyfile to intentional settings, removed unused assets and CSS, and
    removed duplicate pass/reference and backend examples. Retained the custom
    C++ domain and XML fallback because they serve real references.
12. Moved exhaustive Qiskit contracts out of the tutorial, repaired figure and
    equation references, added descriptive image alternatives, formatted counts
    and CLI results, and enlarged the QAOA plots. Pass examples show results;
    the reuse example checks that two allocations become one.

## Reassessment on current main

- All twelve finding groups still require the retained changes. The new DD
  allocation and execution optimizations do not establish universal operation
  complexity bounds or change the corrected algebra.
- Target compilation now takes `TargetEnvironment` and derives its output from
  `PayloadSpecification` (#2219). Both executable target examples and the DDSIM
  submission use that contract, including the explicit-topology example.
- The Qiskit reference retains #2461's reusable custom gates, scalar folding,
  snapshot handling, and validation limits. The tutorial links to that updated
  reference. The no-output description distinguishes void frontend programs from
  the Qiskit exporter's retained constant-zero compatibility case.
- The OpenQASM contracts from #2465, QIR output/allocation contracts from #2446,
  DDSIM execution contracts from #2466, and new benchmark families remain
  intact. Fresh notebook execution also checks the merged QFT-adder example.
- The generated-header-only C++ lint workflow from #2470 remains intact. The
  documentation execution, warning, and navigation gates remain necessary.

## Validation and limits

Local checks and evidence are recorded in the companion execution plan. Build
artifacts stay under `build/docs-audit/` and are not committed. The local HTML
checker covers generated Sphinx and Doxygen navigation; Sphinx linkcheck covers
external document links with the repository's existing exclusions.

No remote hardware, authentication, calibration, or Slurm deployment was
exercised. Provider setup remains an explicit recipe. No universal DD complexity
bound is claimed or proved. The changes retain Sphinx, MyST notebooks, the
existing theme, and native Doxygen rather than introducing another framework.

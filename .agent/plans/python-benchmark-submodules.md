# Organize Python benchmarks into family submodules

Status: historical implementation record.

## Goal and scope

The Python benchmark API currently puts every family type into `mqt.core.bench`.
That flat module is already crowded with five families and will become hard to
browse as the catalog grows. After this change, users can import one family
submodule, such as `mqt.core.bench.qft`, while shared result types remain in
`mqt.core.bench`.

The public example is `qft.QFT(qft.Options(...))`. The primary benchmark class
keeps its algorithm name, while module-local helper types use unprefixed names
such as `Options`, `Method`, `Topology`, and `Basis`. The extra hierarchy
`mqt.core.bench.benchmarks.qft` is deliberately not added because `bench`
already identifies the domain.

## Constraints

- MQT Core already exposes native submodules from extension modules. Evidence:
  `mqt.core.ir.operations` and `mqt.core.qdmi.driver` use nanobind
  `def_submodule`, and recursive stub generation creates one `.pyi` file per
  submodule.

## Decisions

- Use `mqt.core.bench.bv`, `.ghz`, `.grover`, `.qft`, and `.qpe`. Keep `Output`
  and `Evaluation` in the root module and put `Phase` in `qpe`. Rationale:
  direct submodules scale without repeating the word “benchmark.”

- Keep primary classes such as `qft.QFT`, but expose helper types as
  `qft.Options`, `qft.Method`, `ghz.Topology`, and similar unprefixed names. Do
  not provide flat or prefixed compatibility aliases. Rationale: the submodule
  supplies the family context, while the primary class still names the algorithm
  represented by an instance.

## Outcome and validation

The Python API now groups five benchmark families into direct native submodules.
The primary types retain their algorithm names, helper types use concise local
names, and shared results stay at the root. The generated stubs mirror that
runtime hierarchy. The extension build, official stub generation, focused
behavior tests, repository lint, and diff checks passed. The user stopped the
full C++ lint session during its build, before clang-tidy analyzed the changed
files; no C++ lint result is claimed.

## Code and ownership

`bindings/bench/register_bench.cpp` defines the native `mqt.core.bench`
extension and currently registers every shared and family-specific Python type
in one module. `python/mqt/core/bench.pyi` is generated from that extension by
the `stubs` Nox session. `bindings/bench/CMakeLists.txt` builds the extension
and installs generated stubs for editable builds. `test/python/test_bench.py`
and `docs/benchmarks.md` exercise and document the public Python API.

A nanobind submodule is a Python module created by the native extension with
`def_submodule`. Recursive stub generation represents this structure as
`python/mqt/core/bench/__init__.pyi` and one sibling `.pyi` file for each
family. Generated stub files must never be edited by hand.

## Acceptance

Direct imports of all five family modules must succeed. Each family must create
its existing typed benchmark, serialize and parse its existing JSON, evaluate
counts, and generate a QC program. `Output` and `Evaluation` must remain
available only from `mqt.core.bench`; `Phase` must be available from
`mqt.core.bench.qpe`. The old flat family names must be absent. The focused
Python suite and relevant build targets must pass. Generated stubs must describe
the same hierarchy and contain no stale flat `bench.pyi` file.

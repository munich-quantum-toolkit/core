# Compiler getting-started workshop

Status: complete.

## Outcome and scope

Three executable MyST-NB notebooks explain compiler representations,
optimizations, structured quantum-classical programs, and target compilation to
readers who know Python and basic quantum computing. The workshop preserves the
existing QPE quickstart and links to interface guides for detailed contracts.
C++ pass implementation and public API changes remain out of scope.

## Decisions

- Scalar qubits expose QC/QCO differences before register bookkeeping. The
  optimization experiment compares full unitary matrices before measurement.
- A GHZ loop and measurement-driven correction explain QTensor, CBit, and
  quantum ownership through control flow. Draw the indexed GHZ program only
  after unrolling; Qiskit export requires static quantum indices.
- All-to-all and line models share RZ/RY/CX support. Compare native gates,
  connectivity, and logical outputs without fixing a routing layout or requiring
  identical histograms. Inspect OpenQASM assignments because Qiskit's drawer
  omits classical stores.
- MyST-NB owns execution and notebook output; Sphinx supplies download links.
  Existing Qiskit/Matplotlib dependencies generate figures. Assertions in the
  notebooks cover the experiments without a separate testing framework.
- Two conceptual SVGs replace the original tutorial's static figure collection.

## Validation

The complete documentation build executes all 38 workshop cells and validates
local generated links. Downloaded notebooks also execute in fresh kernels
outside the repository, including GHZ widths 2, 3, and 4. The only expected
stderr is the explicitly demonstrated insufficient-capacity diagnostic.

Validation commands are `uvx nox --non-interactive -s docs`,
`uvx nox --non-interactive -s docs -- -b linkcheck`, and `uvx nox -s lint`.
Generated figures and HTML navigation/downloads were inspected. Live browser
preview was unavailable; no browser-interaction validation is claimed.

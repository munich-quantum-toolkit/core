# Compiler tutorials

Status: complete.

## Outcome and scope

Three independent MyST-NB notebooks in `docs/tutorials/` teach compiler
representations, structured programs, and hardware compilation to readers who
know Python and basic quantum computing. A top-level landing page provides
notebook downloads and one `uv` setup command. The QPE quickstart and interface
guides retain their separate roles. The QDMI overview explains its interface and
links to the Amazon Braket and IQM case studies.

## Decisions

- Use scalar qubits before registers. Check the optimization against full
  unitary matrices; use sampled logical outputs for measurement and routing.
- Teach QTensor ownership with a GHZ loop and classical feedback with a
  measurement-driven correction. Unroll before Qiskit export when register
  indices depend on the loop.
- Compare all-to-all and line targets with the same native gates. Assert native
  operations, connectivity, register preservation, and logical results without
  fixing the compiler's chosen layout or exact finite-shot histograms.
- Share the measurement-store analysis between OpenQASM and Qiskit export.
  Preserve quantum operation order and fuse only stores whose classical accesses
  permit the move. OpenQASM retains temporaries for other cases and does not
  modify the input IR. Qiskit retains its narrower supported subset.
- MyST-NB owns execution; Sphinx provides downloads. Fold supporting imports,
  plotting, and validation inputs. The shared requirements file includes the
  compiler, JupyterLab, Qiskit visualization, and the two syntax lexers.

## Validation

The clean documentation build executes 38 tutorial cells and checks generated
links. The downloaded notebooks also pass in fresh kernels outside the
repository, using the documented `uv` requirements and a wheel built from this
revision. The exercises include GHZ widths 2, 3, and 4 and the expected
insufficient-capacity diagnostic.

Native regression tests cover grouped measurement order, register bit order,
input immutability, and conflicting classical accesses. All 375 Qiskit
translation tests pass. The optimized native test suite passes with the existing
job-ID test skipped. Stub regeneration produces no changes. Full repository lint
and C++ lint pass.

Validation uses the local `release-clang-ipo` configure, build, and CTest
presets; `uvx nox -s stubs`; `uvx nox -s lint`; `uvx nox -s cpp-lint`;
`uvx nox --non-interactive -s docs`; and the docs session with `-b linkcheck`.
Generated figures, folded cells, navigation, and downloads were inspected. No
live browser was available.

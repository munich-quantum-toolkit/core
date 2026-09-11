[![PyPI](https://img.shields.io/pypi/v/mqt.core?logo=pypi&style=flat-square)](https://pypi.org/project/mqt.core/)
![OS](https://img.shields.io/badge/os-linux%20%7C%20macos%20%7C%20windows-blue?style=flat-square)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/JOSS-10.21105/joss.07478-blue.svg?style=flat-square)](https://doi.org/10.21105/joss.07478)
[![CI](https://img.shields.io/github/actions/workflow/status/munich-quantum-toolkit/core/ci.yml?branch=main&style=flat-square&logo=github&label=ci)](https://github.com/munich-quantum-toolkit/core/actions/workflows/ci.yml)
[![CD](https://img.shields.io/github/actions/workflow/status/munich-quantum-toolkit/core/cd.yml?style=flat-square&logo=github&label=cd)](https://github.com/munich-quantum-toolkit/core/actions/workflows/cd.yml)
[![Documentation](https://img.shields.io/readthedocs/mqt-core?logo=readthedocs&style=flat-square)](https://mqt.readthedocs.io/projects/core)
[![codecov](https://img.shields.io/codecov/c/github/munich-quantum-toolkit/core?style=flat-square&logo=codecov)](https://codecov.io/gh/munich-quantum-toolkit/core)

<p align="center">
  <a href="https://mqt.readthedocs.io">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/logo-mqt-dark.svg" width="60%">
      <img src="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/logo-mqt-light.svg" width="60%" alt="MQT Logo">
    </picture>
  </a>
</p>

# MQT Core - The Backbone of the Munich Quantum Toolkit (MQT)

MQT Core is a collection of open-source C++20 and Python libraries for quantum
computing. Its MQT Compiler Collection is built on **MLIR and LLVM** and
connects structured quantum-classical programs to optimization, hardware
mapping, and execution. Its libraries form the backbone of the
[_Munich Quantum Toolkit (MQT)_](https://mqt.readthedocs.io).

**MQT Core 4 is a major architectural release:** the compiler's program
representations replace the classic circuit APIs. Start with the
[v4 release overview](CHANGELOG.md#unreleased) and
[v3-to-v4 upgrade guide](UPGRADING.md#unreleased) when migrating an existing
application. The low-level DD and QDMI libraries remain available.

<p align="center">
  <a href="https://mqt.readthedocs.io/projects/core">
  <img width=30% src="https://img.shields.io/badge/documentation-blue?style=for-the-badge&logo=read%20the%20docs" alt="Documentation" />
  </a>
</p>

## Key Features

- **[MQT Compiler Collection](https://mqt.readthedocs.io/projects/core/en/stable/mlir/mqt_compiler_collection.html):**
  Generate and optimize structured quantum/classical programs, map them to
  devices, synthesize native gates, and exchange programs through OpenQASM,
  Qiskit, QIR, and jeff.
- **[Decision diagrams](https://mqt.readthedocs.io/projects/core/en/stable/dd_package.html):**
  Represent quantum states and operations, simulate programs, and analyze their
  behavior through C++ and Python.
- **[QIR execution](https://mqt.readthedocs.io/projects/core/en/stable/qir/index.html):**
  Execute supported QIR Base and Adaptive Profile programs with the DD runtime.
- **[QDMI](https://mqt.readthedocs.io/projects/core/en/stable/qdmi/index.html):**
  Discover devices, query capabilities, compile programs, and submit jobs. Use
  bundled DDSIM for execution and superconducting hardware models for
  compilation.
- **SDK and HPC integration:** connect devices through
  [Qiskit](https://mqt.readthedocs.io/projects/core/en/stable/qdmi/qdmi_backend.html),
  [PennyLane](https://mqt.readthedocs.io/projects/core/en/stable/qdmi/pennylane_device.html),
  and
  [Slurm](https://mqt.readthedocs.io/projects/core/en/stable/qdmi/slurm.html).
- **[Structured benchmarks](https://mqt.readthedocs.io/projects/core/en/stable/benchmarks.html):**
  Generate configurable quantum programs, query analytic references, and
  evaluate sampled results.

## Getting Started

Install [mqt.core](https://pypi.org/project/mqt.core/) in a Python 3.11 or newer
virtual environment:

```console
uv pip install mqt.core
```

Estimate the phase `3/8` with eight bits of precision using
**iterative quantum phase estimation (QPE)**. This uses two qubits and
measurement feedback. Compile for the bundled DDSIM device, then submit the
compiled program:

```python
from fractions import Fraction

from mqt.core.bench import qpe
from mqt.core.mlir import compile_program, submit_program
from mqt.core.qdmi.driver import open_device

benchmark = qpe.QPE(qpe.Options(precision=8, phase=Fraction(3, 8), method=qpe.Method.ITERATIVE))
program = benchmark.generate()
device = open_device("mqt.ddsim.default")
compiled = compile_program(program, target=device)
job = submit_program(compiled, target=device, num_shots=1024)
job.wait()

counts = job.get_counts()
outcome = max(counts, key=lambda bits: counts[bits])
phase = Fraction(int(outcome, 2), 2**benchmark.output.width)
assert phase == benchmark.options.phase
assert benchmark.evaluate(counts).total_variation_distance < 1e-12
print(f"Counts: {counts}")
print(f"Estimated phase: {phase}")
```

```text
Counts: {'01100000': 1024}
Estimated phase: 3/8
```

The
[QPE walkthrough](https://mqt.readthedocs.io/projects/core/en/stable/getting_started.html)
compares standard and iterative QPE and evaluates a phase that cannot be
represented exactly with eight bits. This phase-gate benchmark illustrates the
phase-estimation step used in algorithms such as Shor's.

## Further Documentation

- [Install and build MQT Core](https://mqt.readthedocs.io/projects/core/en/stable/installation.html).
- [Compile and execute programs](https://mqt.readthedocs.io/projects/core/en/stable/getting_started.html).
- Browse the
  [Python API](https://mqt.readthedocs.io/projects/core/en/stable/api/mqt/core/index.html)
  and
  [C++ entry point](https://mqt.readthedocs.io/projects/core/en/stable/cpp_api.html).

## Development

Source builds require a C++20 compiler and CMake 3.28 or newer. Building the
compiler collection also requires
[LLVM/MLIR 23.1 or newer](https://mqt.readthedocs.io/projects/core/en/stable/installation.html#setting-up-mlir).
Prebuilt Python wheels include the compiler and simulator. Graphviz is optional
for exporting DD visualizations.

See the
[contribution guide](https://mqt.readthedocs.io/projects/core/en/stable/contributing.html)
for development setup and checks. For questions and suggestions, open a
[discussion](https://github.com/munich-quantum-toolkit/core/discussions) or an
[issue](https://github.com/munich-quantum-toolkit/core/issues).

## Contributors and Supporters

MQT Core is developed by [MQSC](https://mq.sc) and the
[Chair for Design Automation](https://www.cda.cit.tum.de/) at the
[Technical University of Munich](https://www.tum.de/).
Among others, it is part of the
[Munich Quantum Software Stack (MQSS)](https://www.munich-quantum-valley.de/research/research-areas/mqss)
ecosystem, which is being developed as part of the
[Munich Quantum Valley (MQV)](https://www.munich-quantum-valley.de) initiative.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-logo-banner-dark.svg" width="90%">
    <img src="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-logo-banner-light.svg" width="90%" alt="MQT Partner Logos">
  </picture>
</p>

Thank you to all the contributors who have helped make MQT Core a reality!

<p align="center">
  <a href="https://github.com/munich-quantum-toolkit/core/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=munich-quantum-toolkit/core" alt="Contributors to munich-quantum-toolkit/core" />
  </a>
</p>

The MQT will remain free, open-source, and permissively licensed — now and in
the future. We are firmly committed to keeping it open and actively maintained
for the quantum computing community.

To support this endeavor, please consider:

- Starring and sharing our repositories:
  <https://github.com/munich-quantum-toolkit>
- Contributing code, documentation, tests, or examples via issues and pull
  requests
- Citing the MQT in your publications (see [Cite This](#cite-this))
- Citing our research in your publications (see
  [References](https://mqt.readthedocs.io/projects/core/en/stable/references.html))
- Using the MQT in research and teaching, and sharing feedback and use cases
- Sponsoring us on GitHub: <https://github.com/sponsors/munich-quantum-toolkit>

<p align="center">
  <a href="https://github.com/sponsors/munich-quantum-toolkit">
  <img width=20% src="https://img.shields.io/badge/Sponsor-white?style=for-the-badge&logo=githubsponsors&labelColor=black&color=blue" alt="Sponsor the MQT" />
  </a>
</p>

## Cite This

Please cite the work that best fits your use case.

### MQT Core (the tool)

When citing the software itself or results produced with it, cite the MQT Core
paper:

```bibtex
@article{burgholzer2025MQTCore,
  title        = {{{MQT Core}}: {{The}} Backbone of the {{Munich Quantum Toolkit (MQT)}}},
  author       = {Burgholzer, Lukas and Stade, Yannick and Peham, Tom and Wille, Robert},
  year         = 2025,
  journal      = {Journal of Open Source Software},
  publisher    = {The Open Journal},
  volume       = 10,
  number       = 108,
  pages        = 7478,
  doi          = {10.21105/joss.07478},
  url          = {https://doi.org/10.21105/joss.07478}
}
```

### The MQT Compiler Collection

When citing the compilation framework built on MLIR, cite the MQT Compiler
Collection paper:

```bibtex
@article{MQTCompilerCollection2026,
  title        = {The {{MQT Compiler Collection}}: {{A}} Blueprint for a Future-Proof Quantum-Classical Compilation Framework},
  author       = {Burgholzer, Lukas and Haag, Daniel and Stade, Yannick and Rovara, Damian and Hopf, Patrick and Wille, Robert},
  year         = {2026},
  booktitle    = {Design, Automation and Test in Europe},
  doi          = {10.23919/DATE69613.2026.11539504},
  eprint       = {2604.08674},
  eprinttype   = {arxiv},
}
```

### The Munich Quantum Toolkit (the project)

When discussing the overall MQT project or its ecosystem, cite the MQT Handbook:

```bibtex
@inproceedings{mqt,
  title        = {The {{MQT}} Handbook: {{A}} Summary of Design Automation Tools and Software for Quantum Computing},
  shorttitle   = {{The MQT Handbook}},
  author       = {Wille, Robert and Berent, Lucas and Forster, Tobias and Kunasaikaran, Jagatheesan and Mato, Kevin and Peham, Tom and Quetschlich, Nils and Rovara, Damian and Sander, Aaron and Schmid, Ludwig and Schoenberger, Daniel and Stade, Yannick and Burgholzer, Lukas},
  year         = 2024,
  booktitle    = {IEEE International Conference on Quantum Software (QSW)},
  doi          = {10.1109/QSW62656.2024.00013},
  eprint       = {2405.17543},
  eprinttype   = {arxiv},
  addendum     = {A live version of this document is available at \url{https://mqt.readthedocs.io}}
}
```

---

## Acknowledgements

The Munich Quantum Toolkit has been supported by the European Research Council
(ERC) under the European Union's Horizon 2020 research and innovation program
(grant agreement No. 101001318), the Bavarian State Ministry for Science and
Arts through the Distinguished Professorship Program, as well as the Munich
Quantum Valley, which is supported by the Bavarian state government with funds
from the Hightech Agenda Bayern Plus.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-funding-footer-dark.svg" width="90%">
    <img src="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-funding-footer-light.svg" width="90%" alt="MQT Funding Footer">
  </picture>
</p>

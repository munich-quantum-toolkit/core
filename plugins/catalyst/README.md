[//]: # "TODO: Update the banners"

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

# MQT Core Catalyst MLIR Plugin

This sub-package of MQT Core provides a [Catalyst](https://github.com/PennyLaneAI/catalyst) plugin for the [MLIR](https://mlir.llvm.org/) framework.
It allows you to use MQT Core's MLIR dialects and transformations within the Catalyst framework, enabling advanced quantum circuit optimizations and transformations.

TODO: extend this section with more details about the Catalyst plugin, its features, and how to use it.

If you have any questions, feel free to create a [discussion](https://github.com/munich-quantum-toolkit/core/discussions) or an [issue](https://github.com/munich-quantum-toolkit/core/issues) on [GitHub](https://github.com/munich-quantum-toolkit/core).

## Contributors and Supporters

The _[Munich Quantum Toolkit (MQT)](https://mqt.readthedocs.io)_ is developed by the [Chair for Design Automation](https://www.cda.cit.tum.de/) at the [Technical University of Munich](https://www.tum.de/) and supported by the [Munich Quantum Software Company (MQSC)](https://munichquantum.software).
Among others, it is part of the [Munich Quantum Software Stack (MQSS)](https://www.munich-quantum-valley.de/research/research-areas/mqss) ecosystem, which is being developed as part of the [Munich Quantum Valley (MQV)](https://www.munich-quantum-valley.de) initiative.

<p align="center">
  <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-logo-banner-dark.svg" width="90%">
   <img src="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-logo-banner-light.svg" width="90%" alt="MQT Partner Logos">
  </picture>
</p>

Thank you to all the contributors who have helped make MQT Core a reality!

<p align="center">
<a href="https://github.com/munich-quantum-toolkit/core/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=munich-quantum-toolkit/core" />
</a>
</p>

## Getting Started

TODO
`mqt.core.catalyst` is available via [PyPI](https://pypi.org/project/mqt.core/) for all major operating systems and supports Python 3.10 to 3.13.

```console
(.venv) $ pip install TODO
```

The following code gives an example on how to use an MQT pass with PennyLane's Catalyst

```python3
import catalyst
import pennylane as qml
from catalyst.passes import apply_pass

@apply_pass("mqt.mqtopt-to-catalystquantum")
@apply_pass("mqt.mqt-core-round-trip")
@apply_pass("mqt.catalystquantum-to-mqtopt")
@qml.qnode(qml.device("lightning.qubit", wires=2))
def circuit() -> None:
    qml.Hadamard(wires=[0])
    qml.CNOT(wires=[0, 1])
    catalyst.measure(0)
    catalyst.measure(1)

@qml.qjit(target="mlir", autograph=True)
def module() -> None:
    return circuit()
```

## System Requirements

TODO

## Cite This

If you want to cite MQT Core, please use the following BibTeX entry:

```bibtex
TODO
```

---

## Acknowledgements

The Munich Quantum Toolkit has been supported by the European
Research Council (ERC) under the European Union's Horizon 2020 research and innovation program (grant agreement
No. 101001318), the Bavarian State Ministry for Science and Arts through the Distinguished Professorship Program, as well as the
Munich Quantum Valley, which is supported by the Bavarian state government with funds from the Hightech Agenda Bayern Plus.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-funding-footer-dark.svg" width="90%">
    <img src="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-funding-footer-light.svg" width="90%" alt="MQT Funding Footer">
  </picture>
</p>

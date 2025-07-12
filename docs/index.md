# MQT Core - The Backbone of the Munich Quantum Toolkit (MQT)

```{raw} latex
\begin{abstract}
```

MQT Core is an open-source C++17 and Python library for quantum computing that forms the backbone of the quantum software tools developed as part of the _{doc}`Munich Quantum Toolkit (MQT) <mqt:index>`_.
To this end, MQT Core consists of multiple components that are used throughout the MQT, including a fully fledged intermediate representation (IR) for quantum computations, a state-of-the-art decision diagram (DD) package for quantum computing, and a state-of-the-art ZX-diagram package for working with the ZX-calculus.

This documentation provides a comprehensive guide to the MQT Core library, including {doc}`installation instructions <installation>`, a {doc}`quickstart guide for the MQT Core IR <mqt_core_ir>`, its {doc}`decision diagram (DD) package <dd_package>`, and its {doc}`ZX-calculus package <zx_package>`, as well as detailed {doc}`API documentation <api/mqt/core/index>`.
The source code of MQT Core is publicly available on GitHub at [munich-quantum-toolkit/core](https://github.com/munich-quantum-toolkit/core), while pre-built binaries are available via [PyPI](https://pypi.org/project/mqt.core/) for all major operating systems and all modern Python versions.
MQT Core is fully compatible with Qiskit 1.0 and above.

````{only} latex
```{note}
A live version of this document is available at [mqt.readthedocs.io/projects/core](https://mqt.readthedocs.io/projects/core).
```
````

```{raw} latex
\end{abstract}

\sphinxtableofcontents
```

```{toctree}
:hidden:

self
```

```{toctree}
:maxdepth: 1
:caption: User Guide
:hidden:

installation
mqt_core_ir
dd_package
zx_package
mlir
NA QDMI Device <na_qdmi_device>
references
CHANGELOG
UPGRADING
```

````{only} not latex
```{toctree}
:maxdepth: 1
:caption: DD Package Evaluation
:hidden:

dd_package_evaluation
```

```{toctree}
:maxdepth: 1
:titlesonly:
:caption: Developers
:glob:
:hidden:

contributing
support
DevelopmentGuide
```
````

```{toctree}
:caption: Python API Reference
:maxdepth: 1
:hidden:

api/mqt/core/index
```

```{toctree}
:glob:
:caption: C++ API Reference
:maxdepth: 1
:hidden:

api/cpp/namespacelist
```

````{only} html
## Contributors and Supporters

The _[Munich Quantum Toolkit (MQT)](https://mqt.readthedocs.io)_ is developed by the [Chair for Design Automation](https://www.cda.cit.tum.de/) at the [Technical University of Munich](https://www.tum.de/)
and supported by the [Munich Quantum Software Company (MQSC)](https://munichquantum.software).
Among others, it is part of the [Munich Quantum Software Stack (MQSS)](https://www.munich-quantum-valley.de/research/research-areas/mqss) ecosystem,
which is being developed as part of the [Munich Quantum Valley (MQV)](https://www.munich-quantum-valley.de) initiative.

<div style="margin-top: 0.5em">
<div class="only-light" align="center">
  <img src="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-logo-banner-light.svg" width="90%" alt="MQT Banner">
</div>
<div class="only-dark" align="center">
  <img src="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-logo-banner-dark.svg" width="90%" alt="MQT Banner">
</div>
</div>

Thank you to all the contributors who have helped make MQT Core a reality!

<p align="center">
<a href="https://github.com/munich-quantum-toolkit/core/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=munich-quantum-toolkit/core" />
</a>
</p>

## Cite This

If you want to cite MQT Core, please use the following BibTeX entry:

```bibtex
@article{burgholzer2025MQTCore,
  title     = {{{MQT Core}}: {{The}} Backbone of the {{Munich Quantum Toolkit (MQT)}}},
  author    = {Lukas Burgholzer and Yannick Stade and Tom Peham and Robert Wille},
  year      = {2025},
  journal   = {Journal of Open Source Software},
  publisher = {The Open Journal},
  volume    = {10},
  number    = {108},
  pages     = {7478},
  doi       = {10.21105/joss.07478},
  url       = {https://doi.org/10.21105/joss.07478},
}
```
````

# MQT Core

MQT Core provides reusable C++20 and Python libraries for quantum computing: the
**MQT Compiler Collection**, built on **MLIR and LLVM**, together with decision
diagrams, QIR execution, QDMI device access, SDK and HPC integrations, and
structured benchmarks. It forms the backbone of the
{doc}`Munich Quantum Toolkit (MQT) <mqt:index>`.

**MQT Core 4 is a major architectural release.** Structured compiler
representations replace the classic circuit APIs and connect quantum-classical
programs to optimization, hardware mapping, and execution. For the release
highlights, see the {doc}`v4 overview <CHANGELOG>`. Existing users should start
with the {doc}`v3-to-v4 migration guide <UPGRADING>`.

## Start with your task

Install the [Python package](https://pypi.org/project/mqt.core/) or follow the
{doc}`source-build instructions <installation>`. Then choose a starting point:

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Compile and execute a program
:img-top: _static/tasks/compile.webp
:img-alt:

Estimate a phase with two qubits and measurement feedback.

- **First example:** {doc}`QPE walkthrough <getting_started>`
- **Tutorials:** {doc}`Understand quantum compilation <tutorials/index>`
- **Guide:** {doc}`Compilation and execution <compilation/index>`
- **API:** {py:mod}`mqt.core.mlir`
:::

:::{grid-item-card} Connect or implement a device
:img-top: _static/tasks/devices.webp
:img-alt:

Discover QDMI devices, integrate SDKs, and implement device interfaces.

- **First example:** [Discover and use a device](qdmi/driver.md#python-bindings)
- **Guide:** {doc}`QDMI devices and integrations <qdmi/index>`
- **API:** {py:mod}`mqt.core.qdmi` and {doc}`C++ <cpp_api>`
:::

:::{grid-item-card} Use decision diagrams
:img-top: _static/tasks/decision-diagrams.webp
:img-alt:

Represent and manipulate quantum states and operations in C++ or Python.

- **First example:** [DD quickstart](dd_package.md#quickstart)
- **Guide:** {doc}`Decision diagrams <dd_package>`
- **API:** {py:mod}`mqt.core.dd` and {doc}`C++ <cpp_api>`
:::

:::{grid-item-card} Generate and evaluate benchmarks
:img-top: _static/tasks/benchmarks.webp
:img-alt:

Configure structured programs and compare results with analytic references.

- **First example:**
  [Configure a benchmark](benchmarks.md#configure-a-typed-instance)
- **Guide:** {doc}`Structured benchmarks <benchmarks>`
- **API:** {py:mod}`mqt.core.bench`
:::

:::{grid-item-card} Exchange quantum programs
:img-top: _static/tasks/exchange.webp
:img-alt:

Move between Qiskit, OpenQASM, jeff, and QIR text or bitcode.

- **First example:** [Qiskit to QIR](qir/index.md#from-a-qiskit-circuit-to-qir)
- **Guides:** {doc}`QIR <qir/index>`, {doc}`OpenQASM <mlir/OpenQASM>`, and
  {doc}`Qiskit <mlir/qiskit>`, plus {doc}`jeff exchange <jeff>`
- **API:** {py:func}`~mqt.core.mlir.compile_program`
:::

:::{grid-item-card} Embed or extend MQT Core
:img-top: _static/tasks/extend.webp
:img-alt:

Use the C++ libraries or work on the MLIR compiler infrastructure.

- **First example:** [C++ library quickstart](cpp_api.md#use-the-dd-library)
- **Guide:** [C++ compilation](mlir/target_compilation.md#c-source-tree-api) and
  [compiler development](development.md#mlir)
- **API:** {doc}`C++ reference <cpp_api>` and
  {doc}`MLIR dialects and passes <mlir/index>`
:::

::::

Source code is available on
[GitHub](https://github.com/munich-quantum-toolkit/core). For installation
options, platform requirements, and development setup, see {doc}`installation`.

```{toctree}
:hidden:

self
```

```{toctree}
:maxdepth: 1
:caption: User Guide
:hidden:

installation
compilation/index
qdmi/index
dd_package
benchmarks
references
CHANGELOG
UPGRADING
```

```{toctree}
:maxdepth: 1
:titlesonly:
:caption: Developers
:glob:
:hidden:

contributing
ai_usage
development
glossary
tooling
support
```

```{toctree}
:caption: API Reference
:maxdepth: 1
:hidden:

api/mqt/core/index
cpp_api
mlir/index
```

## Contributors and Supporters

MQT Core is developed by [MQSC](https://mq.sc) and the
[Chair for Design Automation](https://www.cda.cit.tum.de/) at the
[Technical University of Munich](https://www.tum.de/).
Among others, it is part of the
[Munich Quantum Software Stack (MQSS)](https://www.munich-quantum-valley.de/research/research-areas/mqss)
ecosystem, which is being developed as part of the
[Munich Quantum Valley (MQV)](https://www.munich-quantum-valley.de) initiative.

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
  <img src="https://contrib.rocks/image?repo=munich-quantum-toolkit/core" alt="MQT Core contributors" />
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
- Citing the MQT in your publications (see {doc}`References <references>`)
- Using the MQT in research and teaching, and sharing feedback and use cases
- Sponsoring us on GitHub:
  <https://github.com/sponsors/munich-quantum-toolkit>

<p align="center">
<iframe src="https://github.com/sponsors/munich-quantum-toolkit/button" title="Sponsor munich-quantum-toolkit" height="32" width="114" style="border: 0; border-radius: 6px;"></iframe>
</p>

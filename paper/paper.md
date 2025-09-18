---
title: "MQT Core: The Backbone of the Munich Quantum Toolkit (MQT)"
tags:
  - Python
  - C++
  - MQT
  - Quantum Computing
  - Design Automation
  - Intermediate Representation
  - Data Structures
  - Decision Diagrams
  - ZX-Calculus
authors:
  - name: Lukas Burgholzer
    corresponding: true
    orcid: 0000-0003-4699-1316
    affiliation: "1, 2"
  - name: Yannick Stade
    orcid: 0000-0001-5785-2528
    affiliation: 1
  - name: Tom Peham
    orcid: 0000-0003-3434-7881
    affiliation: 1
  - name: Robert Wille
    orcid: 0000-0002-4993-7860
    affiliation: "1, 2, 3"
affiliations:
  - name: Chair for Design Automation, Technical University of Munich, Germany
    index: 1
  - name: Munich Quantum Software Company GmbH, Garching near Munich, Germany
    index: 2
  - name: Software Competence Center Hagenberg GmbH, Hagenberg, Austria
    index: 3
date: 7 November 2024
bibliography: paper.bib
---

# Summary

MQT Core is an open-source C++ and Python library for quantum computing that forms the backbone of
the quantum software tools developed as part of the _Munich Quantum Toolkit (MQT,
[@willeMQTHandbookSummary2024])_ by the [Chair for Design Automation](https://www.cda.cit.tum.de/)
at the [Technical University of Munich](https://www.tum.de/) as well as the [Munich Quantum Software Company (MQSC)](https://munichquantum.software).
To this end, it consists of multiple components that are used throughout the MQT, including a fully fledged intermediate representation
(IR) for quantum computations, a state-of-the-art decision diagram (DD) package for quantum
computing, and a state-of-the-art ZX-diagram package for working with the ZX-calculus. Pre-built
binaries are available via [PyPI](https://pypi.org/project/mqt.core/) for all major operating
systems and all modern Python versions. MQT Core is fully compatible with IBM's Qiskit 1.0 and above
[@qiskit2024], as well as the OpenQASM format [@cross2022openqasm], enabling seamless integration
with the broader quantum computing community.

# Statement of Need

Quantum computing is rapidly transitioning from theoretical research to practice, with potential
applications in fields such as finance, chemistry, machine learning, optimization, cryptography, and
unstructured search. However, the development of scalable quantum applications requires automated,
efficient, and accessible software tools that cater to the diverse needs of end users, engineers,
and physicists across the entire quantum software stack.

The Munich Quantum Toolkit (MQT, [@willeMQTHandbookSummary2024]) addresses this need by leveraging
decades of design automation expertise from the classical computing domain. Developed by the Chair
for Design Automation at the Technical University of Munich, the MQT provides a comprehensive suite
of tools designed to support various design tasks in quantum computing. These tasks include
high-level application development, classical simulation, compilation, verification of quantum
circuits, quantum error correction, and physical design.

MQT Core offers a flexible intermediate representation for quantum computations that forms the basis
for working with quantum circuits throughout the MQT. The library provides interfaces to IBM's
Qiskit [@qiskit2024] and the OpenQASM format [@cross2022openqasm] to make the developed tools
accessible to the broader quantum computing community. Furthermore, MQT Core integrates
state-of-the-art data structures for quantum computing, such as decision diagrams
[@willeDecisionDiagramsQuantum2023] and the ZX-calculus [@vandeweteringZXcalculusWorkingQuantum2020;
@duncanGraphtheoreticSimplificationQuantum2020], that power the MQT's software packages for classical
quantum circuit simulation ([MQT DDSIM](https://github.com/cda-tum/mqt-ddsim)), compilation ([MQT QMAP](https://github.com/cda-tum/mqt-qmap)),
verification ([MQT QCEC](https://github.com/cda-tum/mqt-qcec)), and more. As such, MQT Core has enabled
more than 30 research papers over its first five years of development [@willeDecisionDiagramsQuantum2023;
@hillmichJustRealThing2020;
@hillmichApproximatingDecisionDiagrams2022; @grurlStochasticQuantumCircuit2021;
@grurlConsideringDecoherenceErrors2020; @grurlNoiseawareQuantumCircuit2023;
@grurlAutomaticImplementationEvaluation2023; @burgholzerHybridSchrodingerFeynmanSimulation2021;
@burgholzerExploitingArbitraryPaths2022; @burgholzerSimulationPathsQuantum2022;
@burgholzerEfficientConstructionFunctional2021; @hillmichAccurateNeededEfficient2020;
@hillmichConcurrencyDDbasedQuantum2020; @hillmichExploitingQuantumTeleportation2021;
@burgholzerLimitingSearchSpace2022; @pehamDepthoptimalSynthesisClifford2023; @pehamOptimalSubarchitecturesQuantum2023;
@schmidComputationalCapabilitiesCompiler2024; @schmidHybridCircuitMapping2024;
@burgholzerAdvancedEquivalenceChecking2021; @burgholzerImprovedDDbasedEquivalence2020;
@burgholzerPowerSimulationEquivalence2020; @burgholzerRandomStimuliGeneration2021;
@burgholzerVerifyingResultsIBM2020; @pehamEquivalenceCheckingParadigms2022;
@pehamEquivalenceCheckingParameterized2023; @pehamEquivalenceCheckingQuantum2022;
@willeVerificationQuantumCircuits2022; @sanderHamiltonianSimulationDecision2023;
@willeToolsQuantumComputing2022; @willeVisualizingDecisionDiagrams2021; @willeEfficientCorrectCompilation2020].

To ensure performance, MQT Core is primarily implemented in C++. Since the quantum computing
community predominantly uses Python, MQT Core provides Python bindings that allow seamless
integration with existing Python-based quantum computing tools. In addition, pre-built Python wheels
are available for all major platforms and Python versions, making it easy to install and use MQT
Core in various environments without the need for manual compilation.

# Related Work

MQT Core builds on a rich history of research in quantum computing, design automation, and data
structures. The design of its IR is heavily inspired by IBM's Qiskit [@qiskit2024], with which it
has stayed compatible since `qiskit-terra` version 0.16.1 (released at the end of 2020).
MQT Core remains one of the few libraries providing drop-in replacements for large parts of Qiskit's
core data structures in C++.
Alternative IRs, that come as part of larger quantum computing frameworks, include
Quantinuum's C++-based t$|$ket$\rangle$ [@sivarajah_tket_2021],
LBNL's Python-based `bqskit` [@younis_berkeley_2021],
Xanadu's MLIR-based `catalyst` [@Ittah2024], and
NVIDIA's MLIR-based `CUDA-Q` [@cuda_q].

The origin of the decision diagram package in MQT Core dates back to the seminal work on
Quantum Multiple-Valued Decision Diagrams (QMDDs) [@niemann_qmdds_2016].
It provides a state-of-the-art implementation of QMDDs that natively integrates with the MQT Core IR.
Alternative types of quantum decision diagrams and related software packages include
TDD's [@hong_tensor_2022], Bit-Slicing Decision Diagrams [@tsaiBitSlicingHilbertSpace2021], and
LIMDDs [@vinkhuijzen_limdd_2023].
In comparison to MQT Core, most of these libraries have remained academic prototypes and have not
seen widespread adoption in the quantum computing community.

The ZX-diagram package in MQT Core is inspired by the `PyZX` library [@kissingerPyZXLargeScale2020]
and the t$|$ket$\rangle$ [@sivarajah_tket_2021] compiler.
It provides an efficient C++ implementation of core ZX-calculus concepts and is tightly integrated
with the MQT Core IR.
Compared to other implementations, the ZX package in MQT Core is fine-tuned for verification use
cases and provides dedicated support for handling qubit permutations as well as numerical
inaccuracies that arise in practice.

# Acknowledgements

We would like to thank all past and present contributors to the MQT Core project, including (in alphabetical order):
Hartwig Bauer, Martin Fink, Ioan-Albert Florea, Elias Foramitti, Rebecca Ghidini, Thomas Grurl,
Stefan Hillmich, Fabian Hingerl, Daniel Lummerstorfer, Joachim Marin, Katrin Muck, Christoph Pichler,
Tobias Prienberger, Parham Rahimi, Janez Rotman, Damian Rovara, Roope Salmi, Aaron Sander,
Ludwig Schmid, Sarah Schneider, Tianyi Wang, Theresa Wasserer, and Alwin Zulehner.
Their contributions have been instrumental in making MQT Core a robust and feature-rich library.

The Munich Quantum Toolkit has been supported by the European Research Council (ERC) under the
European Union's Horizon 2020 research and innovation program (grant agreement No. 101001318), the
Bavarian State Ministry for Science and Arts through the Distinguished Professorship Program, as
well as the Munich Quantum Valley, which is supported by the Bavarian state government with funds
from the Hightech Agenda Bayern Plus.

# References

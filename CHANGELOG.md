<!-- v4 entries follow user workflows; PR references are in ascending number order. Earlier releases retain their existing order. -->

# Changelog

All notable changes to this project will be documented in this file.

The format is based on a mixture of [Keep a Changelog] and [Common Changelog].
This project adheres to [Semantic Versioning], with the exception that minor
releases may include breaking changes.

## [Unreleased]

- ✨ Optimize constant two-qubit blocks before placement and during native
  synthesis. Remove cancelled interactions before routing and preserve cheaper
  native operations in both target compilation and synthesis. Avoid redundant
  cleanup in both target pipelines. ([#2537]) ([**@burgholzer**])

## [4.0.0] - 2026-09-11

### MQT Core 4: a compiler foundation built on MLIR and LLVM

MQT Core 4 is a major architectural release centered on the
**MQT Compiler Collection**, a quantum-classical compilation framework built on
**MLIR and LLVM**. It replaces the classic circuit representation with a
compiler infrastructure that connects program import, optimization, hardware
mapping, program exchange, and execution through C++, Python, and `mqt-cc`.

The new foundation makes quantum operations and classical computation part of
the same structured program. Quantum dialects work with MLIR's classical
arithmetic, functions, and control flow, so the compiler can represent loops,
reusable functions, and measurement feedback throughout supported pipelines.

The central changes are:

- **A new program model.** QC provides reference-based quantum operations for
  frontend interoperability; QCO uses linear quantum values for transformations.
  QTensor and CBit represent quantum and classical registers. Typed programs,
  builders, and verifiers expose this structure to compiler users and
  developers.
- **A shared, extensible compilation pipeline.** Import OpenQASM or Qiskit,
  inspect intermediate representations, compose optimization passes, synthesize
  gates, and place and route programs for a device's operations and topology.
  Python, C++, and the command-line driver use the same compiler infrastructure.
- **Compilation connected to execution.** Exchange structured programs through
  jeff or emit OpenQASM and QIR. QIR lowering produces LLVM text or bitcode for
  Base and Adaptive profiles; DDSIM uses LLVM JIT compilation and the DD runtime
  to execute supported QIR through QDMI. QCO also supports direct DD simulation,
  sampling, and unitary construction.
- **Workflows built around the compiler.** Typed benchmarks, SDK integrations,
  device-directed compilation, and six executable tutorials connect program
  construction to results. Python wheels include the compiler and local
  simulator; source builds enable the LLVM/MLIR compiler by default.

**Upgrading from v3 requires an explicit migration.** Classic circuit APIs,
including `QuantumComputation`, are removed. The low-level DD and QDMI libraries
remain available. Start with the [v3-to-v4 upgrade guide](UPGRADING.md#400) for
the new program model, migration paths, API replacements, and build changes
relative to v3.10.0. The entries below retain the contributing PRs and authors.

### Added

#### Compiler APIs and representations

- ✨ Introduce the MLIR/LLVM-based MQT Compiler Collection with typed QC, QCO,
  OpenQASM, jeff, and QIR programs; shared C++ and Python compilation APIs;
  inspection methods; and the `mqt-cc` driver. ([#1264], [#1470], [#1471],
  [#1815], [#1914], [#2149], [#2343], [#2519]) ([**@burgholzer**],
  [**@denialhaag**], [**@simon1hofmann**], [**@taminob**])

- ✨ Add QC/QCO program builders with implicit locations, tracked qubits and
  tensors, register names, gate matrices, and checked linear quantum values.
  Preserve caller-owned inputs and reject invalid programs at checked
  boundaries. ([#1402], [#1428], [#1430], [#1443], [#1446], [#1465], [#1472],
  [#1474], [#1510], [#1542], [#1580], [#1602], [#1620], [#1623], [#1626],
  [#1627], [#1728], [#1730], [#1781], [#1869], [#1872], [#2014], [#2017],
  [#2136], [#2213], [#2220], [#2266], [#2295], [#2296], [#2300], [#2514])
  ([**@burgholzer**], [**@denialhaag**], [**@li-mingbao**],
  [**@MatthiasReumann**], [**@simon1hofmann**], [**@taminob**])

- ✨ Represent classical registers and structured `if`, `for`, `while`, and
  switch control flow across QC and QCO, including `break`, `continue`, wide
  register arithmetic, and reusable quantum functions with tracked wire
  correspondence. ([#1506], [#1638], [#1700], [#1717], [#1749], [#1806],
  [#1808], [#1824], [#1925], [#1927], [#1935], [#1936], [#1938], [#1975],
  [#1976], [#2026], [#2158], [#2181], [#2193], [#2194], [#2297], [#2307],
  [#2308], [#2320], [#2322], [#2336], [#2344], [#2357], [#2369], [#2443],
  [#2448], [#2464]) ([**@burgholzer**], [**@denialhaag**], [**@DRovara**],
  [**@li-mingbao**], [**@MatthiasReumann**], [**@simon1hofmann**])

- ✨ Canonicalize quantum gates, modifiers, and static qubits while preserving
  phase, wire correspondence, and side effects. Bound angle arithmetic and
  reduce repeated traversal and validation work. ([#1330], [#1426], [#1436],
  [#1464], [#1475], [#1550], [#1554], [#1567], [#1569], [#1603], [#1751],
  [#1762], [#1782], [#1807], [#1985], [#2041], [#2210], [#2281], [#2282],
  [#2290], [#2291], [#2293], [#2435], [#2477], [#2484], [#2485], [#2486],
  [#2492], [#2502], [#2505], [#2516], [#2524]) ([**@burgholzer**],
  [**@denialhaag**], [**@DRovara**], [**@Ectras**], [**@J4MMlE**],
  [**@simon1hofmann**], [**@taminob**])

#### Compilation for devices

- ✨ Compile for immutable device targets through C++, Python, and `mqt-cc`.
  Match ordered operation availability, native gates, and payload capabilities;
  legalize supported control flow; and reuse compiled payloads for explicit
  device submission. ([#1687], [#1993], [#1999], [#2049], [#2162], [#2211],
  [#2215], [#2218], [#2219], [#2285], [#2323], [#2495], [#2497], [#2506],
  [#2510]) ([**@burgholzer**], [**@denialhaag**], [**@MatthiasReumann**],
  [**@simon1hofmann**])

- ✨ Place and route QCO programs across sparse and directional device
  topologies, preserving classical feed-forward and quantum wire mappings
  through branches, loops, and switches. ([#1537], [#1547], [#1581], [#1583],
  [#1588], [#1600], [#1664], [#1709], [#1716], [#1748], [#1805], [#1870],
  [#1904], [#1911], [#1951], [#1956], [#1997], [#2004], [#2007], [#2016],
  [#2060], [#2119], [#2179], [#2184], [#2185], [#2205], [#2240], [#2301],
  [#2316], [#2351], [#2436], [#2489]) ([**@burgholzer**], [**@denialhaag**],
  [**@MatthiasReumann**], [**@rturrado**], [**@simon1hofmann**])

- ✨ Evaluate multiple initial layouts, including identity and greedy
  candidates, with CPU-based default trial counts. Reuse routing search storage
  and avoid repeated program-graph and liveness analysis. ([#1568], [#1574],
  [#1930], [#2180], [#2488], [#2499], [#2500], [#2517], [#2535])
  ([**@burgholzer**], [**@MatthiasReumann**], [**@simon1hofmann**])

#### Optimization and synthesis

- ✨ Optimize single-qubit runs with Hadamard lifting, quaternion-based rotation
  merging, Euler synthesis, and fusion of constant or parameterized unitaries.
  ([#1407], [#1605], [#1672], [#1674], [#2002], [#2038], [#2228], [#2478])
  ([**@burgholzer**], [**@denialhaag**], [**@J4MMlE**], [**@lirem101**],
  [**@MatthiasReumann**], [**@simon1hofmann**], [**@taminob**])

- ✨ Decompose multi-controlled X, Z, phase, Pauli rotations, and SWAP
  operations. Fuse two-qubit unitaries and synthesize target-native entanglers,
  including RXX, RYY, RZX, RZZ, iSWAP, ECR, and optimal square-root iSWAP
  circuits. ([#1774], [#1802], [#1803], [#1809], [#1810], [#1814], [#1832],
  [#1850], [#1865], [#1961], [#1996], [#1998], [#2001], [#2444], [#2467],
  [#2468], [#2478], [#2507], [#2531]) ([**@burgholzer**], [**@denialhaag**],
  [**@simon1hofmann**])

- ✨ Normalize global phases and expand multi-operation quantum modifiers while
  preserving full-unitary semantics. ([#1986], [#1995], [#2006], [#2015],
  [#2189]) ([**@burgholzer**], [**@denialhaag**], [**@simon1hofmann**])

- ✨ Add measurement lifting, classical-control replacement, explicit dead-gate
  elimination, quantum-loop unrolling, qubit reuse, and Pauli twirling.
  ([#1705], [#1718], [#1755], [#1756], [#1923], [#1924], [#2039], [#2118],
  [#2216], [#2224]) ([**@burgholzer**], [**@denialhaag**], [**@DRovara**],
  [**@MatthiasReumann**], [**@simon1hofmann**])

#### Program import and exchange

- ✨ Import OpenQASM 2 and 3 into QC and emit OpenQASM 3 with reusable gates,
  structured control flow, fixed-width angles, affine qubit indices, and
  dynamically bounded loops. Diagnose unsupported inputs and bound frontend and
  export resource use. ([#1780], [#1910], [#1987], [#1994], [#2003], [#2050],
  [#2169], [#2203], [#2304], [#2338], [#2456], [#2465], [#2482], [#2483],
  [#2515]) ([**@burgholzer**], [**@denialhaag**], [**@J4MMlE**],
  [**@simon1hofmann**])

- ✨ Import and export Qiskit circuits, reusable custom gates, captured
  classical expressions, and structured control flow. Preserve measurements,
  wide register comparisons, and target-aware gate definitions. ([#2031],
  [#2133], [#2140], [#2175], [#2176], [#2223], [#2342], [#2348], [#2398],
  [#2439], [#2452], [#2461], [#2508]) ([**@burgholzer**], [**@denialhaag**],
  [**@MatthiasReumann**], [**@simon1hofmann**])

- ✨ Preserve symbolic Qiskit parameters and parameter-vector provenance through
  supported compiler conversions. ([#2150], [#2178]) ([**@burgholzer**],
  [**@simon1hofmann**])

- ✨ Exchange jeff programs as bytes or files, preserving supported gates,
  phase, scalar control-flow state, arrays, and reusable functions. Use native
  integer arithmetic and report recoverable deserialization failures. ([#1479],
  [#1548], [#1565], [#1637], [#1676], [#1706], [#1776], [#1836], [#1934],
  [#1939], [#2000], [#2018], [#2105], [#2212], [#2339], [#2451], [#2457],
  [#2525]) ([**@burgholzer**], [**@denialhaag**], [**@simon1hofmann**])

- ✨ Generate QIR 2.1 Base and Adaptive profiles as LLVM text or bitcode. Derive
  capability and resource metadata, lower reusable functions and classical
  computation, preserve output order, and reject unsupported Base control flow
  during conversion. ([#1446], [#1513], [#1521], [#1548], [#1567], [#1569],
  [#1570], [#1572], [#1580], [#1620], [#1624], [#1626], [#1648], [#1710],
  [#1751], [#1787], [#1815], [#1823], [#1933], [#1978], [#1979], [#2026],
  [#2030], [#2066], [#2217], [#2294], [#2302], [#2340], [#2446], [#2447],
  [#2449], [#2463], [#2526]) ([**@burgholzer**], [**@denialhaag**],
  [**@li-mingbao**], [**@MatthiasReumann**], [**@simon1hofmann**])

#### Decision diagrams and execution

- ✨ Construct functionality, simulate, and sample QCO programs through C++ and
  Python, with dense-array helpers for supported compiler inputs. Support
  classical arithmetic, structured control flow, dynamic quantum data, and
  statically sized entry-block registers for unitary construction. ([#1915],
  [#1973], [#2077], [#2078], [#2079], [#2334], [#2455], [#2474], [#2518],
  [#2526]) ([**@burgholzer**], [**@denialhaag**], [**@simon1hofmann**])

- ✨ Return ordered DDSIM QIR shots and matching histograms, including
  variable-length recorded outputs. ([#2368]) ([**@burgholzer**])

- ✨ Extract statevectors from eligible Adaptive QIR with classical control
  flow, dynamic allocations, and direct helpers. Retain uncollapsed states after
  eligible OpenQASM and QIR terminal sampling, with lazy dense and sparse result
  queries. ([#2491], [#2494], [#2512]) ([**@burgholzer**])

- ✨ Capture framed, typed QIR output records through DDSIM with `custom2=True`
  and result `CUSTOM1`, alongside ordinary shots and counts. ([#2526])
  ([**@burgholzer**], [**@simon1hofmann**])

#### Structured benchmarks

- ✨ Add typed quantum benchmarks with versioned instance specifications,
  analytic references, deterministic manifests, and C++, Python, and CLI
  interfaces. Families cover BV, GHZ, Grover, QFT, QPE, multiplexers,
  teleportation, QFT adders, modular multiplication, and repeat-until-success
  programs. ([#2135], [#2299], [#2315], [#2324], [#2337], [#2380], [#2402],
  [#2404], [#2409], [#2410], [#2493]) ([**@burgholzer**], [**@denialhaag**])

#### Documentation and development

- 📝 Connect compilation, execution, device access, and program exchange in the
  user guide. Add six executable compiler tutorials and checked Python/C++
  examples with downloadable notebooks. ([#1555], [#1635], [#1773], [#1899],
  [#1959], [#2058], [#2165], [#2462], [#2503], [#2509], [#2526])
  ([**@burgholzer**], [**@denialhaag**], [**@MatthiasReumann**],
  [**@simon1hofmann**], [**@ystade**])

- 📝 Consolidate development policy and agent guidance, including compiler
  contracts, terminology, validation, and reproducible benchmarks. ([#1905],
  [#1907], [#1908], [#1909], [#1960], [#2072], [#2120], [#2128], [#2256],
  [#2272], [#2354], [#2490], [#2496], [#2498], [#2501], [#2523], [#2526])
  ([**@burgholzer**], [**@denialhaag**], [**@simon1hofmann**])

- 🐳 Add a dev container for local development. ([#1786]) ([**@denialhaag**])

### Changed

#### Builds and supported configurations

- 🔧 Build the LLVM/MLIR 23.1.1 compiler infrastructure and DDSIM device by
  default. Support LLVM builds without exceptions or RTTI, and use
  `BUILD_MQT_CORE_MLIR=OFF` for DD/QDMI builds that omit the compiler and DDSIM.
  ([#1356], [#1549], [#1953], [#2125], [#2127], [#2284], [#2298], [#2538])
  ([**@burgholzer**], [**@denialhaag**], [**@simon1hofmann**])

- 💥 Require CMake 3.28 or newer for source builds and embedded projects, with
  native dependency exclusions and system include handling. ([#2421])
  ([**@burgholzer**])

- ⚡ Reduce build, documentation, and test-discovery overhead. Share Ninja
  presets and generate C++ lint prerequisites without a full build while
  honoring disabled interprocedural optimization. ([#1944], [#1954], [#1988],
  [#2047], [#2075], [#2083], [#2426], [#2459], [#2470]) ([**@burgholzer**],
  [**@denialhaag**])

- 📦 Use `vcs-versioning` and standard dynamic metadata for SCM-derived Python
  package versions, with lower bounds for build dependencies. ([#1544], [#2065],
  [#2145], [#2163]) ([**@burgholzer**], [**@denialhaag**])

- ⬆️ Update clang-tidy to version 23 and adapt the C++ sources to its
  diagnostics. Format TableGen files and align LLVM include and namespace
  conventions. ([#1573], [#1673], [#1675], [#1765], [#2028], [#2328])
  ([**@burgholzer**], [**@denialhaag**], [**@simon1hofmann**])

#### Runtime and client behavior

- 💥 Align QIR execution with QIR 2.1 runtime and QIS signatures, isolated
  per-job state, deterministic seeded sampling, and checked state extraction.
  Optimize eligible terminal sampling and output preparation; verify Base output
  records against QIR-Runner. ([#2034], [#2035], [#2036], [#2044], [#2466],
  [#2513]) ([**@burgholzer**])

- ⚡ Reduce QDMI discovery, result-decoding, and Slurm overhead. Release the
  Python GIL during native QDMI calls and avoid repeated capability and result
  queries. ([#2440], [#2460], [#2472], [#2475], [#2481], [#2511])
  ([**@burgholzer**])

- ⚡ Reduce DD cache clearing, element lookup, export, and allocation overhead,
  and avoid repeated QC/QCO conversion work. ([#2441], [#2453], [#2473],
  [#2474], [#2479]) ([**@burgholzer**])

### Fixed

#### Decision diagrams

- 🐛 Correct DD cache invalidation after garbage collection, compressed-matrix
  entries, partial traces, Kronecker products, numeric-table reuse, and logical
  Python vector indexing. Reject malformed paths and ragged dense matrices.
  ([#2441], [#2453], [#2479]) ([**@burgholzer**])

- 🐛 Use deterministic, collision-free DOT node IDs, define signed
  complex-weight hashing, and preserve real-number collection flags during
  relinking. ([#2518]) ([**@burgholzer**])

#### Device access

- 🐛 Preserve QDMI provider lifetimes and target metadata; validate job
  parameters and decoded results; and propagate execution failures through SDK
  integrations. ([#2440], [#2454], [#2458], [#2511], [#2512])
  ([**@burgholzer**])

### Removed

- 💥 Remove `qc::QuantumComputation`, `MQT::CoreIR`, `MQT::CoreQASM`, classic
  Python circuit APIs, and circuit-taking DD helpers. Use compiler-backed QC/QCO
  APIs or remain on the v3 release series. ([#2054], [#2288])
  ([**@burgholzer**], [**@simon1hofmann**])

- 💥 Remove the standalone QIR runner and make the QIR runtime and JIT internal
  DDSIM implementation details. Execute QIR through QDMI. ([#2246])
  ([**@burgholzer**], [**@denialhaag**])

## [3.10.0] - 2026-09-05

_If you are upgrading: please see
[`UPGRADING.md`](UPGRADING.md#3100)._

### Added

- ✨ Expose ordered shots from DDSIM QDMI OpenQASM jobs, with matching
  histograms ([#2368]) ([**@burgholzer**])

### Changed

- ⚡ Run PennyLane QDMI jobs concurrently and release the GIL during waits and
  result retrieval ([#2349]) ([**@burgholzer**])
- 💥 Raise the minimum Qiskit version from 1.1.0 to 2.1.0 ([#2358])
  ([**@burgholzer**])
- 💥 Replace the QDMI-specific primitives with native Qiskit primitives and
  typed backend factories. Sampler and `memory=True` require genuine QDMI
  `SHOTS` ([#2358]) ([**@burgholzer**])
- ⬆️ Update `nanobind` to version 3.0.1 ([#2209], [#2283]) ([**@denialhaag**],
  [**@burgholzer**])
- 💥 Move circuit IR OpenQASM serialization from operation subclasses to
  `qasm3::Serializer` in `qasm3/Serializer.hpp` ([#2249]) ([**@simon1hofmann**])
- 💥 Drop support for x86 macOS and stop publishing the respective wheels
  ([#2259]) ([**@denialhaag**])
- ⬆️ Raise the macOS deployment target to 13.3 to enable `std::format` in libc++
  ([#2259]) ([**@denialhaag**])
- 💥 Require Python 3.11 or newer ([#2209]) ([**@denialhaag**],
  [**@burgholzer**])
- 📦 Publish one split-mode `cp311-abi3` wheel for GIL-enabled CPython 3.11 and
  newer ([#2209]) ([**@denialhaag**], [**@burgholzer**])
- 📦 Publish one `cp315-abi3t` wheel for free-threaded CPython 3.15 and newer
  ([#2209]) ([**@denialhaag**], [**@burgholzer**])
- ⚡ Remove an extra dense copy from `VectorDD.get_vector` ([#2209])
  ([**@burgholzer**])
- 🐛 Protect process-wide DD, IR, and QDMI state for free-threaded Python
  ([#2209]) ([**@burgholzer**])
- 💥 Prune dead and misleading CoreIR APIs, including renaming the non-garbage
  logical output count to `getNoutputQubits()` and `num_output_qubits` ([#2112])
  ([**@simon1hofmann**])

### Removed

- 💥 Remove test-only DD state generators, recursive functionality construction,
  and DD-specific named-gate helpers ([#2257], [#2335]) ([**@simon1hofmann**])
- 💥 Remove the `spdlog` dependency from source builds, installed CMake
  packages, and Python wheels. QDMI diagnostics continue to be written to
  standard error ([#2270]) ([**@denialhaag**])
- 💥 Remove `CircuitOptimizer`. Move circuit flattening and final-measurement
  removal to `QuantumComputation`, equivalence-checking transformations to
  [MQT QCEC], and mapping transformations to [MQT QMAP]. Move single-qubit gate
  fusion to both downstream packages. Remove the public circuit dependency graph
  and transformations without production consumers ([#2262])
  ([**@simon1hofmann**])
- 💥 Remove `MQT::CoreAlgorithms`, its fixed-circuit factories, and the legacy
  DD package evaluation. MQT Core provides no direct replacement ([#2214])
  ([**@burgholzer**])
- 💥 Remove the unused decision-diagram approximation algorithm, including the
  `dd/Approximation.hpp` header, `dd::ApproximationMetadata`, and
  `dd::approximate`. No replacement is provided ([#2154]) ([**@burgholzer**])
- 💥 Remove `nlohmann_json` from the public package contract. MQT Core no longer
  installs or exports the library, no installed header exposes a `nlohmann`
  type, and the decision-diagram statistics report through strings and streams
  ([#2138]) ([**@denialhaag**])
- 💥 Remove the neutral-atom stack, which moves to [MQT QMAP]. This drops the
  neutral-atom computation model, the neutral-atom FoMaC device session, the
  neutral-atom QDMI device and its configuration, the `mqt.core.na` Python
  module, `AodOperation`, and the `Move`, `Bridge`, `AodActivate`,
  `AodDeactivate`, and `AodMove` operation kinds ([#2137]) ([**@denialhaag**])
- 💥 Remove the random-number generator, seed, and `getGenerator()` method from
  `QuantumComputation`; randomized algorithms now own generators initialized
  from their seed arguments ([#2111]) ([**@simon1hofmann**])
- 💥 Remove the FoMaC compatibility name from the C++ and Python QDMI APIs. Use
  the `qdmi` C++ namespace, headers, library, and CMake target; the
  `mqt.core.qdmi` Python module; and module-level functions in
  `mqt.core.qdmi.driver` ([#2115]) ([**@burgholzer**])
- 💥 Remove the ZX-calculus library, including the `mqt-core-zx` target,
  `MQT::CoreZX` alias, `zx` headers and namespace, and its Boost.Multiprecision
  and GMP build support. Equivalence-checking users should use [MQT QCEC]; its
  ZX implementation is internal and does not provide a replacement public API
  ([#2082]) ([**@burgholzer**])
- 🔥 Remove density matrix support from the DD package ([#1466])
  ([**@burgholzer**])
- 🔥 Remove `datastructures` (`ds`) (sub)library ([#1458])
  ([**@burgholzer**])

### Fixed

- 🐛 Initialize Qiskit classical bits before OpenQASM 3 serialization so
  partially measured circuits preserve their zero values ([#2399])
  ([**@burgholzer**])
- 🐛 Handle empty DDSIM results and NUL-terminated QDMI result buffers ([#2288])
  ([**@simon1hofmann**])
- 🐛 Validate output permutations before I/O mapping initialization ([#2278])
  ([**@denialhaag**])

## [3.9.2] - 2026-08-26

_If you are upgrading: please see [`UPGRADING.md`](UPGRADING.md#392)._

### Added

- ✨ Allow C++ and Python QDMI job submissions to omit the shot count, leaving
  repetition semantics to the program and device ([#2258]) ([**@burgholzer**])

## [3.9.1] - 2026-08-25

_If you are upgrading: please see [`UPGRADING.md`](UPGRADING.md#391)._

### Added

- 🚸 Let PennyLane QDMI devices reuse an already-open session, including a
  device selected from a Slurm license ([#2232]) ([**@burgholzer**])
- ✨ Let a package register a program serializer for a program format through
  the `mqt.core.qiskit.program_serializers` entry point group ([#2114])
  ([**@marcelwa**], [**@burgholzer**])
- ✨ Add `mqt.core.qdmi.is_binary_program_format`, which states whether a
  program format requires exact-byte submission ([#2114]) ([**@marcelwa**],
  [**@burgholzer**])

### Removed

- 💥 Remove the IQM JSON converter `qiskit_to_iqm_json` and the `MoveGate` from
  the Qiskit plugin, which [QDMI-on-IQM] now owns ([#2114]) ([**@marcelwa**],
  [**@burgholzer**])

## [3.9.0] - 2026-08-19

_If you are upgrading: please see [`UPGRADING.md`](UPGRADING.md#390)._

### Added

- ✨ Add PennyLane support for gate-based QDMI devices ([#2005], [#2147])
  ([**@burgholzer**], [**@marcelwa**])
- ✨ Add `Device::submitCalibrationJob` and
  `mqt.core.qdmi.Device.submit_calibration_job` for triggering a calibration run
  ([#2148]) ([**@marcelwa**], [**@burgholzer**])
- ✨ Add SpecAudits, a method and probe script for auditing tests that pin
  behavior the project never specified ([#2124]) ([**@marcelwa**])
- 🚸 Add typed stable-ID construction for Qiskit backends, lazy provider
  discovery, and sampler and estimator factories with explicit shot and
  precision defaults ([#2084]) ([**@burgholzer**])
- 🧪 Test static Slurm license admission and QDMI execution with DDSIM and the
  superconducting device, and document the cluster setup ([#2043])
  ([**@burgholzer**])
- ✨ Add C++ FoMaC and Python QDMI adapters that open the device named by one
  local Slurm license environment value ([#2025]) ([**@burgholzer**])
- ✨ Add generic C++ FoMaC and Python QDMI support for custom device properties
  that contain operation handles ([#2042]) ([**@burgholzer**])
- 📝 Generate `llms.txt` documentation indexes with Sphinx-LLM ([#1989],
  [#2046]) ([**@denialhaag**], [**@burgholzer**])
- ✨ Support retrieving existing jobs by ID through the QDMI client API, C++
  FoMaC API, and Python QDMI API, and expose optional device queue length and
  job queue position ([#2008], [#2010]) ([**@burgholzer**])
- 🐍 Build CPython 3.15 wheels. Their post-build tests remain disabled until
  test dependency wheels are available ([#2011]) ([**@denialhaag**])
- ✨ Bundle reusable IQM Garnet and Emerald superconducting device models with
  stable QDMI registry IDs ([#1992]) ([**@burgholzer**])
- ✨ Expose compressed vector and matrix DD serialization through bytes-based
  Python APIs ([#1983]) ([**@burgholzer**])
- ✨ Make the neutral-atom and superconducting QDMI devices runtime configurable
  with session-owned topology, operations, and calibration data ([#1974],
  [#1980]) ([**@burgholzer**])
- ✨ Expose registered QDMI device IDs without loading device libraries
  ([#1972]) ([**@burgholzer**])
- ✨ Add typed runtime configuration transport and relocatable assets for QDMI
  device descriptions ([#1967]) ([**@burgholzer**])

### Changed

- ⬆️ Update QDMI to version 1.3.3 ([#2168]) ([**@denialhaag**])
- ⬆️ Update `nanobind` to version 2.15.0 ([#2141]) ([**@denialhaag**])
- 💥 Replace the MQT-specific QDMI primitive `options` mappings with explicit
  shot and precision defaults ([#2084]) ([**@burgholzer**])
- ♻️ Simplify Python optional-dependency checks while preserving the Qiskit and
  PennyLane availability flags ([#2108]) ([**@simon1hofmann**])
- 💥 Remove the unused `pybind11` CMake helper and rename
  `add_mqt_python_binding_nanobind` to `add_mqt_python_binding` ([#2106])
  ([**@denialhaag**])
- 💥 Move Python QDMI entities and the neutral-atom specialization to QDMI
  namespaces, expose device registration and opening through
  `mqt.core.qdmi.driver`, retain v3 FoMaC compatibility aliases, and let the
  Qiskit adapter open stable device IDs directly ([#2074]) ([**@burgholzer**])
- 🚀 Reduce ZX diagram growth for multi-controlled X gates with an exact
  ancilla-free quadratic decomposition ([#1984]) ([**@burgholzer**])

### Fixed

- 🐛 Distinguish scalar OpenQASM qubits from one-element qubit registers and
  reject indexing scalar qubits ([#2157]) ([**@DRovara**], [**@burgholzer**])
- 🐛 Preserve the original OpenQASM type error when an assignment's right-hand
  expression cannot be typed ([#2156]) ([**@DRovara**], [**@burgholzer**])

### Removed

- 💥 Remove batch job submission from the QDMI client. `Device::submitJob` now
  states that MQT Core does not support batch jobs ([#2148]) ([**@marcelwa**],
  [**@burgholzer**])
- 💥 Remove QDMI device configuration through `[tool.qdmi]` in `pyproject.toml`
  and the vendored toml++ header ([#2116]) ([**@denialhaag**])

## [3.8.0] - 2026-07-30

_If you are upgrading: please see [`UPGRADING.md`](UPGRADING.md#380)._

### Added

- ✨ Add binary-safe QDMI program submission and retrieval to FoMaC, including
  explicit text and exact-byte APIs and all standard QDMI program formats
  ([#1957]) ([**@burgholzer**])
- ✨ Add versioned, relocatable configuration and stable-ID registration for
  QDMI device libraries, including disabled-ID reservations, fresh device
  sessions, idempotent registration, and external-device target metadata
  ([#1912]) ([**@burgholzer**])
- ✨ Add native relative-phase CCX (`rccx`) support across the IR, DD package,
  ZX diagrams, OpenQASM import/export, and Python/Qiskit bindings ([#1886],
  [#1950]) ([**@simon1hofmann**])
- ✨ Add support for QDMI child devices to the driver and FoMaC libraries
  ([#1897], [#1952]) ([**@burgholzer**])
- ✨ Add typed custom property and result queries to the C++ and Python FoMaC
  libraries ([#1895]) ([**@burgholzer**])
- ✨ Add support for custom job parameters to C++ and Python FoMaC library
  ([#1887]) ([**@flowerthrower**], [**@burgholzer**])
- ✨ Add labeled and ordered output schemas to the QIR runtime ([#1877])
  ([**@rturrado**])
- ✨ Add boolean, integer, floating-point, tuple, and array record output
  functions to the QIR runtime ([#1799]) ([**@rturrado**])
- ✨ Add the reusable in-process `MQT::CoreQIRJIT` library and QIR program
  format support to the DDSIM QDMI device ([#1766]) ([**@rturrado**])

### Changed

- ⬆️ Raise the minimum supported QDMI version to 1.3.2 ([#1897])
  ([**@burgholzer**])

### Removed

- 🔥 Replace the unstable C++ `Driver::addDynamicDeviceLibrary` and Python
  `add_dynamic_device_library` APIs with definition registration and stable-ID
  opening ([#1912]) ([**@burgholzer**])

### Fixed

- 🐛 Allow MQT Core to be embedded as a CMake subproject without target
  collisions and make its bundled QDMI devices individually configurable
  ([#1965]) ([**@burgholzer**])
- 🐛 Fix QIR function names for adjoint gates ([#1830]) ([**@denialhaag**])

## [3.7.0] - 2026-07-09

_If you are upgrading: please see [`UPGRADING.md`](UPGRADING.md#370)._

### Added

- ✨ Add support for IQM's `move` gate in the QDMI Qiskit backend converter
  ([#1844], [#1848]) ([**@burgholzer**], [**@marcelwa**])
- 🚸 Add `const` version of the `CompoundOperation`'s `getOps()` function
  ([#1826]) ([**@ystade**])
- 🚸 Add [CMake presets] to provide a standardized and reproducible way to
  configure builds ([#1660]) ([**@denialhaag**])

### Changed

- ⬆️ Update QDMI to version 1.3.2 ([#1873]) ([**@denialhaag**])
- ♻️ Improve implementation and usability of FoMaC classes ([#1849])
  ([**@MatthiasReumann**])
- ⬆️ Update `nanobind` to version 2.13.0 ([#1817])
- ⬆️ Update [munich-quantum-toolkit/workflows] to version `v2.0.1` ([#1660],
  [#1737]) ([**@denialhaag**])

### Removed

- 📝 Remove support for generating LaTeX documentation ([#1828])
  ([**@denialhaag**])

### Fixed

- 🐛 Fix invalid `prop_type` for `QDMI_DEVICE_PROPERTY_COUPLINGMAP` in QDMI SC
  Device ([#1842]) ([**@MatthiasReumann**])

## [3.6.1] - 2026-05-20

### Changed

- 🚸 Improve native gate support for the Qiskit-to-OpenQASM3 conversion in the
  QDMI-Qiskit interface ([#1719]) ([**@burgholzer**])

### Fixed

- 🏁 Fix dynamic loading of QDMI device DLLs on Windows when an absolute path is
  provided ([#1720]) ([**@burgholzer**])

## [3.6.0] - 2026-05-13

_If you are upgrading: please see [`UPGRADING.md`](UPGRADING.md#360)._

### Added

- 🚸 Add a measurement instruction to the default SC QDMI device ([#1694])
  ([**@burgholzer**])
- ✨ Add support for multi-controlled gates to the QDMI Qiskit backend converter
  ([#1694]) ([**@burgholzer**])

### Changed

- ♻️ Build all built-in QDMI devices as shared libraries ([#1694])
  ([**@burgholzer**])
- ⬆️ Update minimum supported Qiskit version to 1.1.0 ([#1694])
  ([**@burgholzer**])

### Fixed

- 🐛 Fix missing `nlohmann_json.natvis` in Windows component-based CMake
  installs ([#1702]) ([**@burgholzer**])
- 🐛 Fix segfault in DD `sample` method when idle classical bits are present
  ([#1694]) ([**@burgholzer**])

### Removed

- 🔥 Remove shared library wrappers for QDMI devices ([#1694])
  ([**@burgholzer**])

## [3.5.1] - 2026-04-23

_If you are upgrading: please see [`UPGRADING.md`](UPGRADING.md#351)._

### Fixed

- 🐛 Fix malformed include directories in exported `nlohmann_json` CMake targets
  for component-based installs ([#1662]) ([**@burgholzer**])

## [3.5.0] - 2026-04-21

_If you are upgrading: please see [`UPGRADING.md`](UPGRADING.md#350)._

### Added

- ✨ Add support for multi-controlled gates to ZX package ([#1380])
  ([**@keefehuang**], [**@denialhaag**])
- ✨ Add Sampler and Estimator primitives to the QDMI-Qiskit interface ([#1507])
  ([**@marcelwa**])

### Changed

- ⬆️ Update `nanobind` to version 2.12.0 ([#1528])
- ⬆️ Update QDMI to version 1.3.0 ([#1652]) ([**@burgholzer**])
- 📦 Switch to component-based installation for the MQT Core Python package
  ([#1596]) ([**@burgholzer**])
- ⬆️ Update QDMI to latest version from stable `v1.2.x` branch ([#1593])
  ([**@burgholzer**])
- ⬆️ Update `clang-tidy` to version 22 ([#1564]) ([**@denialhaag**],
  [**@burgholzer**])
- 👷 Build on `macos-26`/`macos-26-intel` by default and
  `macos-15`/`macos-15-intel` for extensive tests ([#1571]) ([**@denialhaag**])

## [3.4.1] - 2026-02-01

### Changed

- ⬆️ Update `nanobind` to version 2.11.0 ([#1481]) ([**@denialhaag**])
- ⬆️ Update Boost to version 1.89.0 ([#1453]) ([**@burgholzer**])
- ⬆️ Update QDMI to latest version from stable `v1.2.x` branch ([#1453])
  ([**@burgholzer**])
- ⬆️ Update `spdlog` to version 1.17.0 ([#1453]) ([**@burgholzer**])
- ♻️ Use `llc` instead of random `clang` for compiling QIR test circuits to
  improve robustness and handle opaque pointers correctly across LLVM versions
  ([#1447]) ([**@burgholzer**])
- ♻️ Extract singleton pattern into reusable template base class for QDMI
  devices and driver ([#1444]) ([**@ystade**], [**@burgholzer**])
- 🚚 Reorganize QDMI code structure by moving devices into dedicated
  subdirectories and separating driver and common utilities ([#1444])
  ([**@ystade**])

### Removed

- 🔥 No longer actively type check Python code with `mypy` and solely rely on
  `ty` ([#1437]) ([**@burgholzer**])

## [3.4.0] - 2026-01-08

_If you are upgrading: please see [`UPGRADING.md`](UPGRADING.md#340)._

### Added

- ✨ Return device handle from `add_dynamic_device_library` for direct backend
  creation ([#1381]) ([**@marcelwa**])
- ✨ Add IQM JSON support for job submission in Qiskit-QDMI Backend ([#1375],
  [#1382]) ([**@marcelwa**], [**@burgholzer**])
- ✨ Add authentication support for QDMI sessions with token, username/password,
  auth file, auth URL, and project ID parameters ([#1355]) ([**@marcelwa**])
- ✨ Add a new QDMI device that represents a superconducting architecture
  featuring a coupling map ([#1328]) ([**@ystade**])
- ✨ Add bi-directional iterator that traverses the def-use chain of a qubit
  value ([#1310]) ([**@MatthiasReumann**])
- ✨ Add `OptionalDependencyTester` to lazily handle optional Python
  dependencies like Qiskit ([#1243]) ([**@marcelwa**], [**@burgholzer**])
- ✨ Expose the QDMI job interface through FoMaC ([#1243]) ([**@marcelwa**],
  [**@burgholzer**])
- ✨ Add Qiskit backend wrapper with job submission support for QDMI devices
  through a provider interface ([#1243], [#1385]) ([**@marcelwa**],
  [**@burgholzer**])
- ✨ Support `QDMI_DEVICE_PROPERTY_SUPPORTEDPROGRAMFORMATS` in the NA QDMI
  Device and the DDSIM QDMI Device ([#1243]) ([**@marcelwa**],
  [**@burgholzer**])
- ✨ Support `QDMI_DEVICE_JOB_PROPERTY_PROGRAM` in the NA QDMI Device ([#1243])
  ([**@marcelwa**], [**@burgholzer**])

### Changed

- 📦🏁 Build Windows x86 wheels on `windows-2025` runner for newer compiler
  ([#1415]) ([**@burgholzer**])
- 👷 Build on `macos-15`/`windows-2025` by default and `macos-14`/`windows-2022`
  for extensive tests ([#1414]) ([**@burgholzer**])
- 📦🍎 Build macOS arm64 wheels on macos-15 runner for newer compiler ([#1413])
  ([**@burgholzer**])
- ⚡ Improve uv build caching by removing unconditional `reinstall-package` and
  configuring dedicated `cache-keys` ([#1412]) ([**@burgholzer**])
- 👨‍💻📦 Build `spdlog` and QDMI generators as shared libraries in Python package
  builds ([#1411], [#1403]) ([**@burgholzer**])
- ♻️🏁 Remove Windows-specific restrictions for dynamic QDMI device library
  handling ([#1406]) ([**@burgholzer**])
- ♻️ Migrate Python bindings from `pybind11` to `nanobind` ([#1383])
  ([**@denialhaag**], [**@burgholzer**])
- 📦️ Provide Stable ABI wheels for Python 3.12+ ([#1383]) ([**@burgholzer**],
  [**@denialhaag**])
- 🚚 Create dedicated `mqt.core.na` submodule to closely follow the structure of
  other submodules ([#1383]) ([**@burgholzer**])
- ✨ Add common definitions and utilities for QDMI ([#1355]) ([**@burgholzer**])
- 🚚 Move `NA` QDMI device in its right place next to other QDMI devices
  ([#1355]) ([**@burgholzer**])
- ♻️ Allow repeated loading of QDMI device library with potentially different
  session configurations ([#1355]) ([**@burgholzer**])
- ♻️ Enable thread-safe reference counting for QDMI devices singletons ([#1355])
  ([**@burgholzer**])
- ♻️ Refactor `FoMaC` singleton to instantiable `Session` class with
  configurable authentication parameters ([#1355]) ([**@marcelwa**])
- 👷 Stop testing on `ubuntu-22.04` and `ubuntu-22.04-arm` runners ([#1359])
  ([**@denialhaag**], [**@burgholzer**])
- 👷 Stop testing with `clang-19` and start testing with `clang-21` ([#1359])
  ([**@denialhaag**], [**@burgholzer**])
- 👷 Fix macOS tests with Homebrew Clang via new
  `munich-quantum-toolkit/workflows` version ([#1359]) ([**@denialhaag**],
  [**@burgholzer**])
- 👷 Re-enable macOS tests with GCC by disabling module scanning ([#1359])
  ([**@denialhaag**], [**@burgholzer**])
- ♻️ Group circuit operations into scheduling units for MLIR routing ([#1301])
  ([**@MatthiasReumann**])
- 👷 Use `munich-quantum-software/setup-mlir` to set up MLIR ([#1294])
  ([**@denialhaag**])
- ♻️ Preserve tuple structure and improve site type clarity of the MQT NA
  Default QDMI Device ([#1299]) ([**@marcelwa**])
- ♻️ Move DD package evaluation module to standalone script ([#1327])
  ([**@burgholzer**])
- ⬆️ Bump QDMI version to 1.2.0 ([#1243]) ([**@marcelwa**], [**@burgholzer**])

### Fixed

- 🔧 Install all available QDMI device targets in Python package builds
  ([#1403]) ([**@burgholzer**])
- 🐛 Fix operation validation in Qiskit backend to handle device-specific gate
  naming conventions ([#1384]) ([**@marcelwa**])
- 🐛 Fix conditional branch handling when importing MLIR from
  `QuantumComputation` ([#1378]) ([**@lirem101**])
- 🐛 Fix custom QDMI property and parameter handling in SC and NA devices
  ([#1355]) ([**@burgholzer**])
- 🚨 Fix argument naming of `QuantumComputation` and `CompoundOperation` dunder
  methods for properly implementing the `MutableSequence` protocol ([#1338])
  ([**@burgholzer**])
- 🐛 Fix memory management in dynamic QDMI device by making it explicit
  ([#1336]) ([**@ystade**])

### Removed

- 🔥 Remove wheel builds for Python 3.13t ([#1371]) ([**@burgholzer**])
- 🔥 Remove the `evaluation` extra from the MQT Core Python package ([#1327])
  ([**@burgholzer**])
- 🔥 Remove the `mqt-core-dd-compare` entry point from the MQT Core Python
  package ([#1327]) ([**@burgholzer**])

## [3.3.3] - 2025-11-10

### Added

- ✨ Add support for bridge gates for the neutral atom hybrid mapper ([#1293])
  ([**@lsschmid**])

### Fixed

- 🐛 Revert change to `opTypeFromString()` signature made in [#1283] ([#1300])
  ([**@denialhaag**])

## [3.3.2] - 2025-11-04

### Added

- ✨ Add DD-based simulator QDMI device ([#1287]) ([**@burgholzer**])
- ✨ A `--reuse-qubits` pass implementing an advanced form of qubit reuse to
  reduce the qubit count of quantum circuits ([#1108]) ([**@DRovara**])
- ✨ A `--lift-measurements` pass that attempts to move measurements up as much
  as possible, used for instance to enable better qubit reuse ([#1108])
  ([**@DRovara**])
- ✨ Add native support for `R(theta, phi)` gate ([#1283]) ([**@burgholzer**])
- ✨ Add A\*-search-based routing algorithm to MLIR transpilation routines
  ([#1237], [#1271], [#1279]) ([**@MatthiasReumann**])

### Fixed

- 🐛 Fix edge-case in validation of `NAComputation` ([#1276]) ([**@ystade**])
- 🐛 Allow integer QASM version declarations ([#1269]) ([**@denialhaag**])

## [3.3.1] - 2025-10-14

### Fixed

- 🐛 Ensure `spdlog` dependency can be found from `mqt-core` install ([#1263])
  ([**@burgholzer**])

## [3.3.0] - 2025-10-13

_If you are upgrading: please see [`UPGRADING.md`](UPGRADING.md#330)._

### Added

- 👷 Enable testing on Python 3.14 ([#1246]) ([**@denialhaag**])
- ✨ Add dedicated `PlacementPass` to MLIR transpilation routines ([#1232])
  ([**@MatthiasReumann**])
- ✨ Add an NA-specific FoMaC implementation ([#1223], [#1236]) ([**@ystade**],
  [**@burgholzer**])
- ✨ Enable import of BarrierOp into MQTRef ([#1224]) ([**@denialhaag**])
- ✨ Add naive quantum program routing MLIR pass ([#1148])
  ([**@MatthiasReumann**])
- ✨ Add QIR runtime using DD-based simulation ([#1210]) ([**@ystade**],
  [**@burgholzer**])
- ✨ Add SWAP reconstruction patterns to the newly-named
  `SwapReconstructionAndElision` MLIR pass ([#1207]) ([**@taminob**],
  [**@burgholzer**])
- ✨ Add two-way conversions between MQTRef and QIR ([#1091])
  ([**@li-mingbao**])
- 🚸 Define custom assembly formats for MLIR operations ([#1209])
  ([**@denialhaag**])
- ✨ Add support for translating `IfElseOperation`s to the `MQTRef` MLIR dialect
  ([#1164]) ([**@denialhaag**], [**@burgholzer**])
- ✨ Add MQT's implementation of a generic FoMaC with Python bindings ([#1150],
  [#1186], [#1223]) ([**@ystade**])
- ✨ Add new MLIR pass `ElidePermutations` for SWAP gate elimination ([#1151])
  ([**@taminob**])
- ✨ Add new pattern to MLIR pass `GateElimination` for identity gate removal
  ([#1140]) ([**@taminob**])
- ✨ Add Clifford block collection pass to `CircuitOptimizer` module ([#885])
  ([**jannikpflieger**], [**@burgholzer**])
- ✨ Add `isControlled()` method to the `UnitaryInterface` MLIR class ([#1157])
  ([**@taminob**], [**@burgholzer**])
- 📝 Integrate generated MLIR documentation ([#1147]) ([**@denialhaag**],
  [**@burgholzer**])
- ✨ Add `IfElseOperation` to C++ library and Python package to support Qiskit's
  `IfElseOp` ([#1117]) ([**@denialhaag**], [**@burgholzer**],
  [**@lavanya-m-k**])
- ✨ Add `allocQubit` and `deallocQubit` operations for dynamically working with
  single qubits to the MLIR dialects ([#1139]) ([**@DRovara**],
  [**@burgholzer**])
- ✨ Add `qubit` operation for static qubit addressing to the MLIR dialects
  ([#1098], [#1116]) ([**@MatthiasReumann**])
- ✨ Add MQT's implementation of a QDMI Driver ([#1010]) ([**@ystade**])
- ✨ Add MQT's implementation of a QDMI Device for neutral atom-based quantum
  computing ([#996], [#1010], [#1100]) ([**@ystade**], [**@burgholzer**])
- ✨ Add translation from `QuantumComputation` to the `MQTRef` MLIR dialect
  ([#1099]) ([**@denialhaag**], [**@burgholzer**])
- ✨ Add `reset` operations to the MLIR dialects ([#1106]) ([**@DRovara**])

### Changed

- ♻️ Replace custom `AllocOp`, `DeallocOp`, `ExtractOp`, and `InsertOp` with
  MLIR-native `memref` operations ([#1211]) ([**@denialhaag**])
- 🚚 Rename MLIR pass `ElidePermutations` to `SwapReconstructionAndElision`
  ([#1207]) ([**@taminob**])
- ⬆️ Require LLVM 21 for building the MLIR library ([#1180]) ([**@denialhaag**])
- ⬆️ Update to version 21 of `clang-tidy` ([#1180]) ([**@denialhaag**])
- 🚚 Rename MLIR pass `CancelConsecutiveInverses` to `GateElimination` ([#1140])
  ([**@taminob**])
- 🚚 Rename `xxminusyy` to `xx_minus_yy` and `xxplusyy` to `xx_plus_yy` in MLIR
  dialects ([#1071]) ([**@BertiFlorea**], [**@denialhaag**])
- 🚸 Add custom assembly format for operations in the MLIR dialects ([#1139])
  ([**@burgholzer**])
- 🚸 Enable `InferTypeOpInterface` in the MLIR dialects to reduce explicit type
  information ([#1139]) ([**@burgholzer**])
- 🚚 Rename `check-quantum-opt` test target to `mqt-core-mlir-lit-test`
  ([#1139]) ([**@burgholzer**])
- ♻️ Update the `measure` operations in the MLIR dialects to no longer support
  more than one qubit being measured at once ([#1106]) ([**@DRovara**])
- 🚚 Rename `XXminusYY` to `XXminusYYOp` and `XXplusYY` to `XXplusYYOp` in MLIR
  dialects ([#1099]) ([**@denialhaag**])
- 🚚 Rename `MQTDyn` MLIR dialect to `MQTRef` ([#1098]) ([**@MatthiasReumann**])

### Removed

- 🔥 Drop support for Python 3.9 ([#1181]) ([**@denialhaag**])
- 🔥 Remove `ClassicControlledOperation` from C++ library and Python package
  ([#1117]) ([**@denialhaag**])

### Fixed

- 🐛 Fix CMake installation to make `find_package(mqt-core CONFIG)` succeed
  ([#1247]) ([**@burgholzer**], [**@denialhaag**])
- 🏁 Fix stack overflows in OpenQASM layout parsing on Windows for large
  circuits ([#1235]) ([**@burgholzer**])
- ✨ Add missing `StandardOperation` conversions in MLIR roundtrip pass
  ([#1071]) ([**@BertiFlorea**], [**@denialhaag**])

## [3.2.1] - 2025-08-01

### Fixed

- 🐛 Fix usage of `std::accumulate` by changing accumulator parameter from
  reference to value ([#1089]) ([**@denialhaag**])
- 🐛 Fix erroneous `contains` check in DD package ([#1088]) ([**@denialhaag**])

## [3.2.0] - 2025-07-31

_If you are upgrading: please see [`UPGRADING.md`](UPGRADING.md#320)._

### Added

- 🐍 Start building CPython 3.14 wheels ([#1076]) ([**@denialhaag**])
- ✨ Add MQT-internal MLIR dialect conversions ([#1001]) ([**@li-mingbao**])

### Changed

- ✨ Expose enums to Python via `pybind11`'s new (`enum.Enum`-compatible)
  `py::native_enum` ([#1075]) ([**@denialhaag**])
- ⬆️ Require C++20 ([#897]) ([**@burgholzer**], [**@denialhaag**])

## [3.1.0] - 2025-07-11

_If you are upgrading: please see [`UPGRADING.md`](UPGRADING.md#310)._

### Added

- ✨ Add MLIR pass for merging rotation gates ([#1019]) ([**@denialhaag**])
- ✨ Add functions to generate random vector DDs ([#975])
  ([**@MatthiasReumann**])
- ✨ Add function to approximate decision diagrams ([#908])
  ([**@MatthiasReumann**])
- 📦 Add Windows ARM64 wheels ([#926]) ([**@burgholzer**])
- 📝 Add documentation page for MLIR ([#931]) ([**@ystade**])
- ✨ Initial implementation of the mqtdyn Dialect ([#900]) ([**@DRovara**],
  [**@ystade**])

### Fixed

- 🐛 Fix bug in MLIR roundtrip passes caused by accessing an invalidated
  iterator after erasure in a loop ([#932]) ([**@flowerthrower**])
- 🐛 Add missing support for `sxdg` gates in Qiskit circuit import ([#930])
  ([**@burgholzer**])
- 🐛 Fix bug related to initialization of operations with duplicate operands
  ([#964]) ([**@ystade**])
- 🐛 Open issue for Qiskit upstream test only when the test is actually failing
  not when it was cancelled ([#973]) ([**@ystade**])
- 🐛 Fix parsing of `GPhase` in the `MQTOpt` MLIR dialect ([#1042])
  ([**@ystade**], [**@DRovara**])

### Changed

- ⬆️ Bump shared library ABI version from `3.0` to `3.1` ([#1047])
  ([**@denialhaag**])
- ♻️ Switch from reference counting to mark-and-sweep garbage collection in
  decision diagram package ([#1020]) ([**@MatthiasReumann**], [**burgholzer**],
  [**q-inho**])
- ♻️ Move the C++ code for the Python bindings to the top-level `bindings`
  directory ([#982]) ([**@denialhaag**])
- ♻️ Move all Python code (no tests) to the top-level `python` directory
  ([#982]) ([**@denialhaag**])
- ⚡ Improve performance of getNqubits for StandardOperations ([#959])
  ([**@ystade**])
- ♻️ Move Make-State Functionality To StateGeneration ([#984])
  ([**@MatthiasReumann**])
- ♻️ Outsource definition of standard operations from MLIR dialects to reduce
  redundancy ([#933]) ([**@ystade**])
- ♻️ Unify operands and results in MLIR dialects ([#931]) ([**@ystade**])
- ⏪️ Restore support for (MLIR and) LLVM v19 ([#934]) ([**@flowerthrower**],
  [**@ystade**])
- ⬆️ Update nlohmann_json to `v3.12.0` ([#921]) ([**@burgholzer**])

## [3.0.2] - 2025-04-07

### Added

- 📝 Add JOSS journal reference and citation information ([#913])
  ([**@burgholzer**])
- 📝 Add new links to Python package metadata ([#911]) ([**@burgholzer**])

### Fixed

- 📝 Fix old links in Python package metadata ([#911]) ([**@burgholzer**])

## [3.0.1] - 2025-04-07

### Fixed

- 🐛 Fix doxygen build on RtD to include C++ API docs ([#912])
  ([**@burgholzer**])

## [3.0.0] - 2025-04-06

_If you are upgrading: please see [`UPGRADING.md`](UPGRADING.md#300)._

### Added

- ✨ Ship shared C++ libraries with `mqt-core` Python package ([#662])
  ([**@burgholzer**])
- ✨ Add Python bindings for the DD package ([#838]) ([**@burgholzer**])
- ✨ Add direct MQT `QuantumComputation` to Qiskit `QuantumCircuit` export
  ([#859]) ([**@burgholzer**])
- ✨ Support for Qiskit 2.0+ ([#860]) ([**@burgholzer**])
- ✨ Add initial infrastructure for MLIR within the MQT ([#878], [#879], [#892],
  [#893], [#895]) ([**@burgholzer**], [**@ystade**], [**@DRovara**],
  [**@flowerthrower**], [**@BertiFlorea**])
- ✨ Add State Preparation Algorithm ([#543]) ([**@M-J-Hochreiter**])
- 🚸 Add support for indexed identifiers to OpenQASM 3 parser ([#832])
  ([**@burgholzer**])
- 🚸 Allow indexed registers as operation arguments ([#839]) ([**@burgholzer**])
- 📝 Add documentation for the DD package ([#831]) ([**@burgholzer**])
- 📝 Add documentation for the ZX package ([#817]) ([**@pehamTom**])
- 📝 Add C++ API docs setup ([#817]) ([**@pehamTom**], [**@burgholzer**])

### Changed

- **Breaking**: 🚚 MQT Core has moved to the [munich-quantum-toolkit] GitHub
  organization
- **Breaking**: ✨ Adopt [PEP 735] dependency groups ([#762])
  ([**@burgholzer**])
- **Breaking**: ♻️ Encapsulate the OpenQASM parser in its own library ([#822])
  ([**@burgholzer**])
- **Breaking**: ♻️ Replace `Config` template from DD package with constructor
  argument ([#886]) ([**@burgholzer**])
- **Breaking**: ♻️ Remove template parameters from `MemoryManager` and adjacent
  classes ([#866]) ([**@rotmanjanez**])
- **Breaking**: ♻️ Refactor algorithms to use factory functions instead of
  inheritance ([**@a9b7e70**]) ([**@burgholzer**])
- **Breaking**: ♻️ Change pointer parameters to references in DD package
  ([#798]) ([**@burgholzer**])
- **Breaking**: ♻️ Change registers from typedef to actual type ([#807])
  ([**@burgholzer**])
- **Breaking**: ♻️ Refactor `NAComputation` class hierarchy ([#846], [#877])
  ([**@ystade**])
- **Breaking**: ⬆️ Bump minimum required CMake version to `3.24.0` ([#879])
  ([**@burgholzer**])
- **Breaking**: ⬆️ Bump minimum required `uv` version to `0.5.20` ([#802])
  ([**@burgholzer**])
- 📝 Rework existing project documentation ([#789], [#842]) ([**@burgholzer**])
- 📄 Use [PEP 639] license expressions ([#847]) ([**@burgholzer**])

### Removed

- **Breaking**: 🔥 Remove the `Teleportation` gate from the IR ([#882])
  ([**@burgholzer**])
- **Breaking**: 🔥 Remove parsers for `.real`, `.qc`, `.tfc`, and `GRCS` files
  ([#822]) ([**@burgholzer**])
- **Breaking**: 🔥 Remove tensor dump functionality ([#798]) ([**@burgholzer**])
- **Breaking**: 🔥 Remove `extract_probability_vector` functionality ([#883])
  ([**@burgholzer**])

### Fixed

- 🐛 Fix Qiskit layout import and handling ([#849], [#858]) ([**@burgholzer**])
- 🐛 Properly handle timing literals in QASM parser ([#724]) ([**@burgholzer**])
- 🐛 Fix stripping of idle qubits ([#763]) ([**@burgholzer**])
- 🐛 Fix permutation handling in OpenQASM dump ([#810]) ([**@burgholzer**])
- 🐛 Fix out-of-bounds error in ZX `EdgeIterator` ([#758]) ([**@burgholzer**])
- 🐛 Fix endianness in DCX and XX_minus_YY gate matrix definition ([#741])
  ([**@burgholzer**])
- 🐛 Fix needless dummy register in empty circuit construction ([#758])
  ([**@burgholzer**])

## [2.7.0] - 2024-10-08

_📚 Refer to the
[GitHub Release Notes](https://github.com/munich-quantum-toolkit/core/releases)
for previous changelogs._

<!-- Version links -->

[unreleased]: https://github.com/munich-quantum-toolkit/core/compare/v4.0.0...HEAD
[4.0.0]: https://github.com/munich-quantum-toolkit/core/releases/tag/v4.0.0
[3.10.0]: https://github.com/munich-quantum-toolkit/core/releases/tag/v3.10.0
[3.9.2]: https://github.com/munich-quantum-toolkit/core/releases/tag/v3.9.2
[3.9.1]: https://github.com/munich-quantum-toolkit/core/releases/tag/v3.9.1
[3.9.0]: https://github.com/munich-quantum-toolkit/core/releases/tag/v3.9.0
[3.8.0]: https://github.com/munich-quantum-toolkit/core/releases/tag/v3.8.0
[3.7.0]: https://github.com/munich-quantum-toolkit/core/releases/tag/v3.7.0
[3.6.1]: https://github.com/munich-quantum-toolkit/core/releases/tag/v3.6.1
[3.6.0]: https://github.com/munich-quantum-toolkit/core/releases/tag/v3.6.0
[3.5.1]: https://github.com/munich-quantum-toolkit/core/releases/tag/v3.5.1
[3.5.0]: https://github.com/munich-quantum-toolkit/core/releases/tag/v3.5.0
[3.4.1]: https://github.com/munich-quantum-toolkit/core/releases/tag/v3.4.1
[3.4.0]: https://github.com/munich-quantum-toolkit/core/releases/tag/v3.4.0
[3.3.3]: https://github.com/munich-quantum-toolkit/core/releases/tag/v3.3.3
[3.3.2]: https://github.com/munich-quantum-toolkit/core/releases/tag/v3.3.2
[3.3.1]: https://github.com/munich-quantum-toolkit/core/releases/tag/v3.3.1
[3.3.0]: https://github.com/munich-quantum-toolkit/core/releases/tag/v3.3.0
[3.2.1]: https://github.com/munich-quantum-toolkit/core/releases/tag/v3.2.1
[3.2.0]: https://github.com/munich-quantum-toolkit/core/releases/tag/v3.2.0
[3.1.0]: https://github.com/munich-quantum-toolkit/core/releases/tag/v3.1.0
[3.0.2]: https://github.com/munich-quantum-toolkit/core/releases/tag/v3.0.2
[3.0.1]: https://github.com/munich-quantum-toolkit/core/releases/tag/v3.0.1
[3.0.0]: https://github.com/munich-quantum-toolkit/core/releases/tag/v3.0.0
[2.7.0]: https://github.com/munich-quantum-toolkit/core/releases/tag/v2.7.0

<!-- PR links -->

[#2538]: https://github.com/munich-quantum-toolkit/core/pull/2538
[#2537]: https://github.com/munich-quantum-toolkit/core/pull/2537
[#2535]: https://github.com/munich-quantum-toolkit/core/pull/2535
[#2531]: https://github.com/munich-quantum-toolkit/core/pull/2531
[#2526]: https://github.com/munich-quantum-toolkit/core/pull/2526
[#2525]: https://github.com/munich-quantum-toolkit/core/pull/2525
[#2524]: https://github.com/munich-quantum-toolkit/core/pull/2524
[#2523]: https://github.com/munich-quantum-toolkit/core/pull/2523
[#2519]: https://github.com/munich-quantum-toolkit/core/pull/2519
[#2518]: https://github.com/munich-quantum-toolkit/core/pull/2518
[#2517]: https://github.com/munich-quantum-toolkit/core/pull/2517
[#2516]: https://github.com/munich-quantum-toolkit/core/pull/2516
[#2515]: https://github.com/munich-quantum-toolkit/core/pull/2515
[#2514]: https://github.com/munich-quantum-toolkit/core/pull/2514
[#2513]: https://github.com/munich-quantum-toolkit/core/pull/2513
[#2512]: https://github.com/munich-quantum-toolkit/core/pull/2512
[#2511]: https://github.com/munich-quantum-toolkit/core/pull/2511
[#2510]: https://github.com/munich-quantum-toolkit/core/pull/2510
[#2509]: https://github.com/munich-quantum-toolkit/core/pull/2509
[#2508]: https://github.com/munich-quantum-toolkit/core/pull/2508
[#2507]: https://github.com/munich-quantum-toolkit/core/pull/2507
[#2506]: https://github.com/munich-quantum-toolkit/core/pull/2506
[#2505]: https://github.com/munich-quantum-toolkit/core/pull/2505
[#2503]: https://github.com/munich-quantum-toolkit/core/pull/2503
[#2502]: https://github.com/munich-quantum-toolkit/core/pull/2502
[#2501]: https://github.com/munich-quantum-toolkit/core/pull/2501
[#2500]: https://github.com/munich-quantum-toolkit/core/pull/2500
[#2499]: https://github.com/munich-quantum-toolkit/core/pull/2499
[#2498]: https://github.com/munich-quantum-toolkit/core/pull/2498
[#2497]: https://github.com/munich-quantum-toolkit/core/pull/2497
[#2496]: https://github.com/munich-quantum-toolkit/core/pull/2496
[#2495]: https://github.com/munich-quantum-toolkit/core/pull/2495
[#2494]: https://github.com/munich-quantum-toolkit/core/pull/2494
[#2493]: https://github.com/munich-quantum-toolkit/core/pull/2493
[#2492]: https://github.com/munich-quantum-toolkit/core/pull/2492
[#2491]: https://github.com/munich-quantum-toolkit/core/pull/2491
[#2490]: https://github.com/munich-quantum-toolkit/core/pull/2490
[#2489]: https://github.com/munich-quantum-toolkit/core/pull/2489
[#2488]: https://github.com/munich-quantum-toolkit/core/pull/2488
[#2486]: https://github.com/munich-quantum-toolkit/core/pull/2486
[#2485]: https://github.com/munich-quantum-toolkit/core/pull/2485
[#2484]: https://github.com/munich-quantum-toolkit/core/pull/2484
[#2483]: https://github.com/munich-quantum-toolkit/core/pull/2483
[#2482]: https://github.com/munich-quantum-toolkit/core/pull/2482
[#2481]: https://github.com/munich-quantum-toolkit/core/pull/2481
[#2479]: https://github.com/munich-quantum-toolkit/core/pull/2479
[#2478]: https://github.com/munich-quantum-toolkit/core/pull/2478
[#2477]: https://github.com/munich-quantum-toolkit/core/pull/2477
[#2475]: https://github.com/munich-quantum-toolkit/core/pull/2475
[#2474]: https://github.com/munich-quantum-toolkit/core/pull/2474
[#2473]: https://github.com/munich-quantum-toolkit/core/pull/2473
[#2472]: https://github.com/munich-quantum-toolkit/core/pull/2472
[#2470]: https://github.com/munich-quantum-toolkit/core/pull/2470
[#2468]: https://github.com/munich-quantum-toolkit/core/pull/2468
[#2467]: https://github.com/munich-quantum-toolkit/core/pull/2467
[#2466]: https://github.com/munich-quantum-toolkit/core/pull/2466
[#2465]: https://github.com/munich-quantum-toolkit/core/pull/2465
[#2464]: https://github.com/munich-quantum-toolkit/core/pull/2464
[#2463]: https://github.com/munich-quantum-toolkit/core/pull/2463
[#2462]: https://github.com/munich-quantum-toolkit/core/pull/2462
[#2461]: https://github.com/munich-quantum-toolkit/core/pull/2461
[#2460]: https://github.com/munich-quantum-toolkit/core/pull/2460
[#2459]: https://github.com/munich-quantum-toolkit/core/pull/2459
[#2458]: https://github.com/munich-quantum-toolkit/core/pull/2458
[#2457]: https://github.com/munich-quantum-toolkit/core/pull/2457
[#2456]: https://github.com/munich-quantum-toolkit/core/pull/2456
[#2455]: https://github.com/munich-quantum-toolkit/core/pull/2455
[#2454]: https://github.com/munich-quantum-toolkit/core/pull/2454
[#2453]: https://github.com/munich-quantum-toolkit/core/pull/2453
[#2452]: https://github.com/munich-quantum-toolkit/core/pull/2452
[#2451]: https://github.com/munich-quantum-toolkit/core/pull/2451
[#2449]: https://github.com/munich-quantum-toolkit/core/pull/2449
[#2448]: https://github.com/munich-quantum-toolkit/core/pull/2448
[#2447]: https://github.com/munich-quantum-toolkit/core/pull/2447
[#2446]: https://github.com/munich-quantum-toolkit/core/pull/2446
[#2444]: https://github.com/munich-quantum-toolkit/core/pull/2444
[#2443]: https://github.com/munich-quantum-toolkit/core/pull/2443
[#2441]: https://github.com/munich-quantum-toolkit/core/pull/2441
[#2440]: https://github.com/munich-quantum-toolkit/core/pull/2440
[#2439]: https://github.com/munich-quantum-toolkit/core/pull/2439
[#2436]: https://github.com/munich-quantum-toolkit/core/pull/2436
[#2435]: https://github.com/munich-quantum-toolkit/core/pull/2435
[#2426]: https://github.com/munich-quantum-toolkit/core/pull/2426
[#2421]: https://github.com/munich-quantum-toolkit/core/pull/2421
[#2410]: https://github.com/munich-quantum-toolkit/core/pull/2410
[#2409]: https://github.com/munich-quantum-toolkit/core/pull/2409
[#2404]: https://github.com/munich-quantum-toolkit/core/pull/2404
[#2402]: https://github.com/munich-quantum-toolkit/core/pull/2402
[#2399]: https://github.com/munich-quantum-toolkit/core/pull/2399
[#2398]: https://github.com/munich-quantum-toolkit/core/pull/2398
[#2380]: https://github.com/munich-quantum-toolkit/core/pull/2380
[#2369]: https://github.com/munich-quantum-toolkit/core/pull/2369
[#2368]: https://github.com/munich-quantum-toolkit/core/pull/2368
[#2358]: https://github.com/munich-quantum-toolkit/core/pull/2358
[#2357]: https://github.com/munich-quantum-toolkit/core/pull/2357
[#2354]: https://github.com/munich-quantum-toolkit/core/pull/2354
[#2351]: https://github.com/munich-quantum-toolkit/core/pull/2351
[#2349]: https://github.com/munich-quantum-toolkit/core/pull/2349
[#2348]: https://github.com/munich-quantum-toolkit/core/pull/2348
[#2344]: https://github.com/munich-quantum-toolkit/core/pull/2344
[#2343]: https://github.com/munich-quantum-toolkit/core/pull/2343
[#2342]: https://github.com/munich-quantum-toolkit/core/pull/2342
[#2340]: https://github.com/munich-quantum-toolkit/core/pull/2340
[#2339]: https://github.com/munich-quantum-toolkit/core/pull/2339
[#2338]: https://github.com/munich-quantum-toolkit/core/pull/2338
[#2337]: https://github.com/munich-quantum-toolkit/core/pull/2337
[#2336]: https://github.com/munich-quantum-toolkit/core/pull/2336
[#2335]: https://github.com/munich-quantum-toolkit/core/pull/2335
[#2334]: https://github.com/munich-quantum-toolkit/core/pull/2334
[#2328]: https://github.com/munich-quantum-toolkit/core/pull/2328
[#2324]: https://github.com/munich-quantum-toolkit/core/pull/2324
[#2323]: https://github.com/munich-quantum-toolkit/core/pull/2323
[#2322]: https://github.com/munich-quantum-toolkit/core/pull/2322
[#2320]: https://github.com/munich-quantum-toolkit/core/pull/2320
[#2316]: https://github.com/munich-quantum-toolkit/core/pull/2316
[#2315]: https://github.com/munich-quantum-toolkit/core/pull/2315
[#2308]: https://github.com/munich-quantum-toolkit/core/pull/2308
[#2307]: https://github.com/munich-quantum-toolkit/core/pull/2307
[#2304]: https://github.com/munich-quantum-toolkit/core/pull/2304
[#2302]: https://github.com/munich-quantum-toolkit/core/pull/2302
[#2301]: https://github.com/munich-quantum-toolkit/core/pull/2301
[#2300]: https://github.com/munich-quantum-toolkit/core/pull/2300
[#2299]: https://github.com/munich-quantum-toolkit/core/pull/2299
[#2298]: https://github.com/munich-quantum-toolkit/core/pull/2298
[#2297]: https://github.com/munich-quantum-toolkit/core/pull/2297
[#2296]: https://github.com/munich-quantum-toolkit/core/pull/2296
[#2295]: https://github.com/munich-quantum-toolkit/core/pull/2295
[#2294]: https://github.com/munich-quantum-toolkit/core/pull/2294
[#2293]: https://github.com/munich-quantum-toolkit/core/pull/2293
[#2291]: https://github.com/munich-quantum-toolkit/core/pull/2291
[#2290]: https://github.com/munich-quantum-toolkit/core/pull/2290
[#2288]: https://github.com/munich-quantum-toolkit/core/pull/2288
[#2285]: https://github.com/munich-quantum-toolkit/core/pull/2285
[#2284]: https://github.com/munich-quantum-toolkit/core/pull/2284
[#2283]: https://github.com/munich-quantum-toolkit/core/pull/2283
[#2282]: https://github.com/munich-quantum-toolkit/core/pull/2282
[#2281]: https://github.com/munich-quantum-toolkit/core/pull/2281
[#2278]: https://github.com/munich-quantum-toolkit/core/pull/2278
[#2272]: https://github.com/munich-quantum-toolkit/core/pull/2272
[#2270]: https://github.com/munich-quantum-toolkit/core/pull/2270
[#2266]: https://github.com/munich-quantum-toolkit/core/pull/2266
[#2262]: https://github.com/munich-quantum-toolkit/core/pull/2262
[#2259]: https://github.com/munich-quantum-toolkit/core/pull/2259
[#2258]: https://github.com/munich-quantum-toolkit/core/pull/2258
[#2257]: https://github.com/munich-quantum-toolkit/core/pull/2257
[#2256]: https://github.com/munich-quantum-toolkit/core/pull/2256
[#2249]: https://github.com/munich-quantum-toolkit/core/pull/2249
[#2246]: https://github.com/munich-quantum-toolkit/core/pull/2246
[#2240]: https://github.com/munich-quantum-toolkit/core/pull/2240
[#2232]: https://github.com/munich-quantum-toolkit/core/pull/2232
[#2228]: https://github.com/munich-quantum-toolkit/core/pull/2228
[#2224]: https://github.com/munich-quantum-toolkit/core/pull/2224
[#2223]: https://github.com/munich-quantum-toolkit/core/pull/2223
[#2220]: https://github.com/munich-quantum-toolkit/core/pull/2220
[#2219]: https://github.com/munich-quantum-toolkit/core/pull/2219
[#2218]: https://github.com/munich-quantum-toolkit/core/pull/2218
[#2217]: https://github.com/munich-quantum-toolkit/core/pull/2217
[#2216]: https://github.com/munich-quantum-toolkit/core/pull/2216
[#2215]: https://github.com/munich-quantum-toolkit/core/pull/2215
[#2214]: https://github.com/munich-quantum-toolkit/core/pull/2214
[#2213]: https://github.com/munich-quantum-toolkit/core/pull/2213
[#2212]: https://github.com/munich-quantum-toolkit/core/pull/2212
[#2211]: https://github.com/munich-quantum-toolkit/core/pull/2211
[#2210]: https://github.com/munich-quantum-toolkit/core/pull/2210
[#2209]: https://github.com/munich-quantum-toolkit/core/pull/2209
[#2205]: https://github.com/munich-quantum-toolkit/core/pull/2205
[#2203]: https://github.com/munich-quantum-toolkit/core/pull/2203
[#2194]: https://github.com/munich-quantum-toolkit/core/pull/2194
[#2193]: https://github.com/munich-quantum-toolkit/core/pull/2193
[#2189]: https://github.com/munich-quantum-toolkit/core/pull/2189
[#2185]: https://github.com/munich-quantum-toolkit/core/pull/2185
[#2184]: https://github.com/munich-quantum-toolkit/core/pull/2184
[#2181]: https://github.com/munich-quantum-toolkit/core/pull/2181
[#2180]: https://github.com/munich-quantum-toolkit/core/pull/2180
[#2179]: https://github.com/munich-quantum-toolkit/core/pull/2179
[#2178]: https://github.com/munich-quantum-toolkit/core/pull/2178
[#2176]: https://github.com/munich-quantum-toolkit/core/pull/2176
[#2175]: https://github.com/munich-quantum-toolkit/core/pull/2175
[#2169]: https://github.com/munich-quantum-toolkit/core/pull/2169
[#2168]: https://github.com/munich-quantum-toolkit/core/pull/2168
[#2165]: https://github.com/munich-quantum-toolkit/core/pull/2165
[#2163]: https://github.com/munich-quantum-toolkit/core/pull/2163
[#2162]: https://github.com/munich-quantum-toolkit/core/pull/2162
[#2158]: https://github.com/munich-quantum-toolkit/core/pull/2158
[#2157]: https://github.com/munich-quantum-toolkit/core/pull/2157
[#2156]: https://github.com/munich-quantum-toolkit/core/pull/2156
[#2154]: https://github.com/munich-quantum-toolkit/core/pull/2154
[#2150]: https://github.com/munich-quantum-toolkit/core/pull/2150
[#2149]: https://github.com/munich-quantum-toolkit/core/pull/2149
[#2148]: https://github.com/munich-quantum-toolkit/core/pull/2148
[#2147]: https://github.com/munich-quantum-toolkit/core/pull/2147
[#2145]: https://github.com/munich-quantum-toolkit/core/pull/2145
[#2141]: https://github.com/munich-quantum-toolkit/core/pull/2141
[#2140]: https://github.com/munich-quantum-toolkit/core/pull/2140
[#2138]: https://github.com/munich-quantum-toolkit/core/pull/2138
[#2137]: https://github.com/munich-quantum-toolkit/core/pull/2137
[#2136]: https://github.com/munich-quantum-toolkit/core/pull/2136
[#2135]: https://github.com/munich-quantum-toolkit/core/pull/2135
[#2133]: https://github.com/munich-quantum-toolkit/core/pull/2133
[#2128]: https://github.com/munich-quantum-toolkit/core/pull/2128
[#2127]: https://github.com/munich-quantum-toolkit/core/pull/2127
[#2125]: https://github.com/munich-quantum-toolkit/core/pull/2125
[#2124]: https://github.com/munich-quantum-toolkit/core/pull/2124
[#2120]: https://github.com/munich-quantum-toolkit/core/pull/2120
[#2119]: https://github.com/munich-quantum-toolkit/core/pull/2119
[#2118]: https://github.com/munich-quantum-toolkit/core/pull/2118
[#2116]: https://github.com/munich-quantum-toolkit/core/pull/2116
[#2115]: https://github.com/munich-quantum-toolkit/core/pull/2115
[#2114]: https://github.com/munich-quantum-toolkit/core/pull/2114
[#2112]: https://github.com/munich-quantum-toolkit/core/pull/2112
[#2111]: https://github.com/munich-quantum-toolkit/core/pull/2111
[#2108]: https://github.com/munich-quantum-toolkit/core/pull/2108
[#2106]: https://github.com/munich-quantum-toolkit/core/pull/2106
[#2105]: https://github.com/munich-quantum-toolkit/core/pull/2105
[#2084]: https://github.com/munich-quantum-toolkit/core/pull/2084
[#2083]: https://github.com/munich-quantum-toolkit/core/pull/2083
[#2082]: https://github.com/munich-quantum-toolkit/core/pull/2082
[#2079]: https://github.com/munich-quantum-toolkit/core/pull/2079
[#2078]: https://github.com/munich-quantum-toolkit/core/pull/2078
[#2077]: https://github.com/munich-quantum-toolkit/core/pull/2077
[#2075]: https://github.com/munich-quantum-toolkit/core/pull/2075
[#2074]: https://github.com/munich-quantum-toolkit/core/pull/2074
[#2072]: https://github.com/munich-quantum-toolkit/core/pull/2072
[#2066]: https://github.com/munich-quantum-toolkit/core/pull/2066
[#2065]: https://github.com/munich-quantum-toolkit/core/pull/2065
[#2060]: https://github.com/munich-quantum-toolkit/core/pull/2060
[#2058]: https://github.com/munich-quantum-toolkit/core/pull/2058
[#2054]: https://github.com/munich-quantum-toolkit/core/pull/2054
[#2050]: https://github.com/munich-quantum-toolkit/core/pull/2050
[#2049]: https://github.com/munich-quantum-toolkit/core/pull/2049
[#2047]: https://github.com/munich-quantum-toolkit/core/pull/2047
[#2046]: https://github.com/munich-quantum-toolkit/core/pull/2046
[#2044]: https://github.com/munich-quantum-toolkit/core/pull/2044
[#2043]: https://github.com/munich-quantum-toolkit/core/pull/2043
[#2042]: https://github.com/munich-quantum-toolkit/core/pull/2042
[#2041]: https://github.com/munich-quantum-toolkit/core/pull/2041
[#2039]: https://github.com/munich-quantum-toolkit/core/pull/2039
[#2038]: https://github.com/munich-quantum-toolkit/core/pull/2038
[#2036]: https://github.com/munich-quantum-toolkit/core/pull/2036
[#2035]: https://github.com/munich-quantum-toolkit/core/pull/2035
[#2034]: https://github.com/munich-quantum-toolkit/core/pull/2034
[#2031]: https://github.com/munich-quantum-toolkit/core/pull/2031
[#2030]: https://github.com/munich-quantum-toolkit/core/pull/2030
[#2028]: https://github.com/munich-quantum-toolkit/core/pull/2028
[#2026]: https://github.com/munich-quantum-toolkit/core/pull/2026
[#2025]: https://github.com/munich-quantum-toolkit/core/pull/2025
[#2018]: https://github.com/munich-quantum-toolkit/core/pull/2018
[#2017]: https://github.com/munich-quantum-toolkit/core/pull/2017
[#2016]: https://github.com/munich-quantum-toolkit/core/pull/2016
[#2015]: https://github.com/munich-quantum-toolkit/core/pull/2015
[#2014]: https://github.com/munich-quantum-toolkit/core/pull/2014
[#2011]: https://github.com/munich-quantum-toolkit/core/pull/2011
[#2010]: https://github.com/munich-quantum-toolkit/core/pull/2010
[#2008]: https://github.com/munich-quantum-toolkit/core/pull/2008
[#2007]: https://github.com/munich-quantum-toolkit/core/pull/2007
[#2006]: https://github.com/munich-quantum-toolkit/core/pull/2006
[#2005]: https://github.com/munich-quantum-toolkit/core/pull/2005
[#2004]: https://github.com/munich-quantum-toolkit/core/pull/2004
[#2003]: https://github.com/munich-quantum-toolkit/core/pull/2003
[#2002]: https://github.com/munich-quantum-toolkit/core/pull/2002
[#2001]: https://github.com/munich-quantum-toolkit/core/pull/2001
[#2000]: https://github.com/munich-quantum-toolkit/core/pull/2000
[#1999]: https://github.com/munich-quantum-toolkit/core/pull/1999
[#1998]: https://github.com/munich-quantum-toolkit/core/pull/1998
[#1997]: https://github.com/munich-quantum-toolkit/core/pull/1997
[#1996]: https://github.com/munich-quantum-toolkit/core/pull/1996
[#1995]: https://github.com/munich-quantum-toolkit/core/pull/1995
[#1994]: https://github.com/munich-quantum-toolkit/core/pull/1994
[#1993]: https://github.com/munich-quantum-toolkit/core/pull/1993
[#1992]: https://github.com/munich-quantum-toolkit/core/pull/1992
[#1989]: https://github.com/munich-quantum-toolkit/core/pull/1989
[#1988]: https://github.com/munich-quantum-toolkit/core/pull/1988
[#1987]: https://github.com/munich-quantum-toolkit/core/pull/1987
[#1986]: https://github.com/munich-quantum-toolkit/core/pull/1986
[#1985]: https://github.com/munich-quantum-toolkit/core/pull/1985
[#1984]: https://github.com/munich-quantum-toolkit/core/pull/1984
[#1983]: https://github.com/munich-quantum-toolkit/core/pull/1983
[#1980]: https://github.com/munich-quantum-toolkit/core/pull/1980
[#1979]: https://github.com/munich-quantum-toolkit/core/pull/1979
[#1978]: https://github.com/munich-quantum-toolkit/core/pull/1978
[#1976]: https://github.com/munich-quantum-toolkit/core/pull/1976
[#1975]: https://github.com/munich-quantum-toolkit/core/pull/1975
[#1974]: https://github.com/munich-quantum-toolkit/core/pull/1974
[#1973]: https://github.com/munich-quantum-toolkit/core/pull/1973
[#1972]: https://github.com/munich-quantum-toolkit/core/pull/1972
[#1967]: https://github.com/munich-quantum-toolkit/core/pull/1967
[#1965]: https://github.com/munich-quantum-toolkit/core/pull/1965
[#1961]: https://github.com/munich-quantum-toolkit/core/pull/1961
[#1960]: https://github.com/munich-quantum-toolkit/core/pull/1960
[#1959]: https://github.com/munich-quantum-toolkit/core/pull/1959
[#1957]: https://github.com/munich-quantum-toolkit/core/pull/1957
[#1956]: https://github.com/munich-quantum-toolkit/core/pull/1956
[#1954]: https://github.com/munich-quantum-toolkit/core/pull/1954
[#1953]: https://github.com/munich-quantum-toolkit/core/pull/1953
[#1952]: https://github.com/munich-quantum-toolkit/core/pull/1952
[#1951]: https://github.com/munich-quantum-toolkit/core/pull/1951
[#1950]: https://github.com/munich-quantum-toolkit/core/pull/1950
[#1944]: https://github.com/munich-quantum-toolkit/core/pull/1944
[#1939]: https://github.com/munich-quantum-toolkit/core/pull/1939
[#1938]: https://github.com/munich-quantum-toolkit/core/pull/1938
[#1936]: https://github.com/munich-quantum-toolkit/core/pull/1936
[#1935]: https://github.com/munich-quantum-toolkit/core/pull/1935
[#1934]: https://github.com/munich-quantum-toolkit/core/pull/1934
[#1933]: https://github.com/munich-quantum-toolkit/core/pull/1933
[#1930]: https://github.com/munich-quantum-toolkit/core/pull/1930
[#1927]: https://github.com/munich-quantum-toolkit/core/pull/1927
[#1925]: https://github.com/munich-quantum-toolkit/core/pull/1925
[#1924]: https://github.com/munich-quantum-toolkit/core/pull/1924
[#1923]: https://github.com/munich-quantum-toolkit/core/pull/1923
[#1915]: https://github.com/munich-quantum-toolkit/core/pull/1915
[#1914]: https://github.com/munich-quantum-toolkit/core/pull/1914
[#1912]: https://github.com/munich-quantum-toolkit/core/pull/1912
[#1911]: https://github.com/munich-quantum-toolkit/core/pull/1911
[#1910]: https://github.com/munich-quantum-toolkit/core/pull/1910
[#1909]: https://github.com/munich-quantum-toolkit/core/pull/1909
[#1908]: https://github.com/munich-quantum-toolkit/core/pull/1908
[#1907]: https://github.com/munich-quantum-toolkit/core/pull/1907
[#1905]: https://github.com/munich-quantum-toolkit/core/pull/1905
[#1904]: https://github.com/munich-quantum-toolkit/core/pull/1904
[#1899]: https://github.com/munich-quantum-toolkit/core/pull/1899
[#1897]: https://github.com/munich-quantum-toolkit/core/pull/1897
[#1895]: https://github.com/munich-quantum-toolkit/core/pull/1895
[#1887]: https://github.com/munich-quantum-toolkit/core/pull/1887
[#1886]: https://github.com/munich-quantum-toolkit/core/pull/1886
[#1877]: https://github.com/munich-quantum-toolkit/core/pull/1877
[#1873]: https://github.com/munich-quantum-toolkit/core/pull/1873
[#1872]: https://github.com/munich-quantum-toolkit/core/pull/1872
[#1870]: https://github.com/munich-quantum-toolkit/core/pull/1870
[#1869]: https://github.com/munich-quantum-toolkit/core/pull/1869
[#1865]: https://github.com/munich-quantum-toolkit/core/pull/1865
[#1850]: https://github.com/munich-quantum-toolkit/core/pull/1850
[#1849]: https://github.com/munich-quantum-toolkit/core/pull/1849
[#1848]: https://github.com/munich-quantum-toolkit/core/pull/1848
[#1844]: https://github.com/munich-quantum-toolkit/core/pull/1844
[#1842]: https://github.com/munich-quantum-toolkit/core/pull/1842
[#1836]: https://github.com/munich-quantum-toolkit/core/pull/1836
[#1832]: https://github.com/munich-quantum-toolkit/core/pull/1832
[#1830]: https://github.com/munich-quantum-toolkit/core/pull/1830
[#1828]: https://github.com/munich-quantum-toolkit/core/pull/1828
[#1826]: https://github.com/munich-quantum-toolkit/core/pull/1826
[#1824]: https://github.com/munich-quantum-toolkit/core/pull/1824
[#1823]: https://github.com/munich-quantum-toolkit/core/pull/1823
[#1817]: https://github.com/munich-quantum-toolkit/core/pull/1817
[#1815]: https://github.com/munich-quantum-toolkit/core/pull/1815
[#1814]: https://github.com/munich-quantum-toolkit/core/pull/1814
[#1810]: https://github.com/munich-quantum-toolkit/core/pull/1810
[#1809]: https://github.com/munich-quantum-toolkit/core/pull/1809
[#1808]: https://github.com/munich-quantum-toolkit/core/pull/1808
[#1807]: https://github.com/munich-quantum-toolkit/core/pull/1807
[#1806]: https://github.com/munich-quantum-toolkit/core/pull/1806
[#1805]: https://github.com/munich-quantum-toolkit/core/pull/1805
[#1803]: https://github.com/munich-quantum-toolkit/core/pull/1803
[#1802]: https://github.com/munich-quantum-toolkit/core/pull/1802
[#1799]: https://github.com/munich-quantum-toolkit/core/pull/1799
[#1787]: https://github.com/munich-quantum-toolkit/core/pull/1787
[#1786]: https://github.com/munich-quantum-toolkit/core/pull/1786
[#1782]: https://github.com/munich-quantum-toolkit/core/pull/1782
[#1781]: https://github.com/munich-quantum-toolkit/core/pull/1781
[#1780]: https://github.com/munich-quantum-toolkit/core/pull/1780
[#1776]: https://github.com/munich-quantum-toolkit/core/pull/1776
[#1774]: https://github.com/munich-quantum-toolkit/core/pull/1774
[#1773]: https://github.com/munich-quantum-toolkit/core/pull/1773
[#1766]: https://github.com/munich-quantum-toolkit/core/pull/1766
[#1765]: https://github.com/munich-quantum-toolkit/core/pull/1765
[#1762]: https://github.com/munich-quantum-toolkit/core/pull/1762
[#1756]: https://github.com/munich-quantum-toolkit/core/pull/1756
[#1755]: https://github.com/munich-quantum-toolkit/core/pull/1755
[#1751]: https://github.com/munich-quantum-toolkit/core/pull/1751
[#1749]: https://github.com/munich-quantum-toolkit/core/pull/1749
[#1748]: https://github.com/munich-quantum-toolkit/core/pull/1748
[#1737]: https://github.com/munich-quantum-toolkit/core/pull/1737
[#1730]: https://github.com/munich-quantum-toolkit/core/pull/1730
[#1728]: https://github.com/munich-quantum-toolkit/core/pull/1728
[#1720]: https://github.com/munich-quantum-toolkit/core/pull/1720
[#1719]: https://github.com/munich-quantum-toolkit/core/pull/1719
[#1718]: https://github.com/munich-quantum-toolkit/core/pull/1718
[#1717]: https://github.com/munich-quantum-toolkit/core/pull/1717
[#1716]: https://github.com/munich-quantum-toolkit/core/pull/1716
[#1710]: https://github.com/munich-quantum-toolkit/core/pull/1710
[#1709]: https://github.com/munich-quantum-toolkit/core/pull/1709
[#1706]: https://github.com/munich-quantum-toolkit/core/pull/1706
[#1705]: https://github.com/munich-quantum-toolkit/core/pull/1705
[#1702]: https://github.com/munich-quantum-toolkit/core/pull/1702
[#1700]: https://github.com/munich-quantum-toolkit/core/pull/1700
[#1694]: https://github.com/munich-quantum-toolkit/core/pull/1694
[#1687]: https://github.com/munich-quantum-toolkit/core/pull/1687
[#1676]: https://github.com/munich-quantum-toolkit/core/pull/1676
[#1675]: https://github.com/munich-quantum-toolkit/core/pull/1675
[#1674]: https://github.com/munich-quantum-toolkit/core/pull/1674
[#1673]: https://github.com/munich-quantum-toolkit/core/pull/1673
[#1672]: https://github.com/munich-quantum-toolkit/core/pull/1672
[#1664]: https://github.com/munich-quantum-toolkit/core/pull/1664
[#1662]: https://github.com/munich-quantum-toolkit/core/pull/1662
[#1660]: https://github.com/munich-quantum-toolkit/core/pull/1660
[#1652]: https://github.com/munich-quantum-toolkit/core/pull/1652
[#1648]: https://github.com/munich-quantum-toolkit/core/pull/1648
[#1638]: https://github.com/munich-quantum-toolkit/core/pull/1638
[#1637]: https://github.com/munich-quantum-toolkit/core/pull/1637
[#1635]: https://github.com/munich-quantum-toolkit/core/pull/1635
[#1627]: https://github.com/munich-quantum-toolkit/core/pull/1627
[#1626]: https://github.com/munich-quantum-toolkit/core/pull/1626
[#1624]: https://github.com/munich-quantum-toolkit/core/pull/1624
[#1623]: https://github.com/munich-quantum-toolkit/core/pull/1623
[#1620]: https://github.com/munich-quantum-toolkit/core/pull/1620
[#1605]: https://github.com/munich-quantum-toolkit/core/pull/1605
[#1603]: https://github.com/munich-quantum-toolkit/core/pull/1603
[#1602]: https://github.com/munich-quantum-toolkit/core/pull/1602
[#1600]: https://github.com/munich-quantum-toolkit/core/pull/1600
[#1596]: https://github.com/munich-quantum-toolkit/core/pull/1596
[#1593]: https://github.com/munich-quantum-toolkit/core/pull/1593
[#1588]: https://github.com/munich-quantum-toolkit/core/pull/1588
[#1583]: https://github.com/munich-quantum-toolkit/core/pull/1583
[#1581]: https://github.com/munich-quantum-toolkit/core/pull/1581
[#1580]: https://github.com/munich-quantum-toolkit/core/pull/1580
[#1574]: https://github.com/munich-quantum-toolkit/core/pull/1574
[#1573]: https://github.com/munich-quantum-toolkit/core/pull/1573
[#1572]: https://github.com/munich-quantum-toolkit/core/pull/1572
[#1571]: https://github.com/munich-quantum-toolkit/core/pull/1571
[#1570]: https://github.com/munich-quantum-toolkit/core/pull/1570
[#1569]: https://github.com/munich-quantum-toolkit/core/pull/1569
[#1568]: https://github.com/munich-quantum-toolkit/core/pull/1568
[#1567]: https://github.com/munich-quantum-toolkit/core/pull/1567
[#1565]: https://github.com/munich-quantum-toolkit/core/pull/1565
[#1564]: https://github.com/munich-quantum-toolkit/core/pull/1564
[#1555]: https://github.com/munich-quantum-toolkit/core/pull/1555
[#1554]: https://github.com/munich-quantum-toolkit/core/pull/1554
[#1550]: https://github.com/munich-quantum-toolkit/core/pull/1550
[#1549]: https://github.com/munich-quantum-toolkit/core/pull/1549
[#1548]: https://github.com/munich-quantum-toolkit/core/pull/1548
[#1547]: https://github.com/munich-quantum-toolkit/core/pull/1547
[#1544]: https://github.com/munich-quantum-toolkit/core/pull/1544
[#1542]: https://github.com/munich-quantum-toolkit/core/pull/1542
[#1537]: https://github.com/munich-quantum-toolkit/core/pull/1537
[#1528]: https://github.com/munich-quantum-toolkit/core/pull/1528
[#1521]: https://github.com/munich-quantum-toolkit/core/pull/1521
[#1513]: https://github.com/munich-quantum-toolkit/core/pull/1513
[#1510]: https://github.com/munich-quantum-toolkit/core/pull/1510
[#1507]: https://github.com/munich-quantum-toolkit/core/pull/1507
[#1506]: https://github.com/munich-quantum-toolkit/core/pull/1506
[#1481]: https://github.com/munich-quantum-toolkit/core/pull/1481
[#1479]: https://github.com/munich-quantum-toolkit/core/pull/1479
[#1475]: https://github.com/munich-quantum-toolkit/core/pull/1475
[#1474]: https://github.com/munich-quantum-toolkit/core/pull/1474
[#1472]: https://github.com/munich-quantum-toolkit/core/pull/1472
[#1471]: https://github.com/munich-quantum-toolkit/core/pull/1471
[#1470]: https://github.com/munich-quantum-toolkit/core/pull/1470
[#1466]: https://github.com/munich-quantum-toolkit/core/pull/1466
[#1465]: https://github.com/munich-quantum-toolkit/core/pull/1465
[#1464]: https://github.com/munich-quantum-toolkit/core/pull/1464
[#1458]: https://github.com/munich-quantum-toolkit/core/pull/1458
[#1453]: https://github.com/munich-quantum-toolkit/core/pull/1453
[#1447]: https://github.com/munich-quantum-toolkit/core/pull/1447
[#1446]: https://github.com/munich-quantum-toolkit/core/pull/1446
[#1444]: https://github.com/munich-quantum-toolkit/core/pull/1444
[#1443]: https://github.com/munich-quantum-toolkit/core/pull/1443
[#1437]: https://github.com/munich-quantum-toolkit/core/pull/1437
[#1436]: https://github.com/munich-quantum-toolkit/core/pull/1436
[#1430]: https://github.com/munich-quantum-toolkit/core/pull/1430
[#1428]: https://github.com/munich-quantum-toolkit/core/pull/1428
[#1426]: https://github.com/munich-quantum-toolkit/core/pull/1426
[#1415]: https://github.com/munich-quantum-toolkit/core/pull/1415
[#1414]: https://github.com/munich-quantum-toolkit/core/pull/1414
[#1413]: https://github.com/munich-quantum-toolkit/core/pull/1413
[#1412]: https://github.com/munich-quantum-toolkit/core/pull/1412
[#1411]: https://github.com/munich-quantum-toolkit/core/pull/1411
[#1407]: https://github.com/munich-quantum-toolkit/core/pull/1407
[#1406]: https://github.com/munich-quantum-toolkit/core/pull/1406
[#1403]: https://github.com/munich-quantum-toolkit/core/pull/1403
[#1402]: https://github.com/munich-quantum-toolkit/core/pull/1402
[#1385]: https://github.com/munich-quantum-toolkit/core/pull/1385
[#1384]: https://github.com/munich-quantum-toolkit/core/pull/1384
[#1383]: https://github.com/munich-quantum-toolkit/core/pull/1383
[#1382]: https://github.com/munich-quantum-toolkit/core/pull/1382
[#1381]: https://github.com/munich-quantum-toolkit/core/pull/1381
[#1380]: https://github.com/munich-quantum-toolkit/core/pull/1380
[#1378]: https://github.com/munich-quantum-toolkit/core/pull/1378
[#1375]: https://github.com/munich-quantum-toolkit/core/pull/1375
[#1371]: https://github.com/munich-quantum-toolkit/core/pull/1371
[#1359]: https://github.com/munich-quantum-toolkit/core/pull/1359
[#1356]: https://github.com/munich-quantum-toolkit/core/pull/1356
[#1355]: https://github.com/munich-quantum-toolkit/core/pull/1355
[#1338]: https://github.com/munich-quantum-toolkit/core/pull/1338
[#1336]: https://github.com/munich-quantum-toolkit/core/pull/1336
[#1330]: https://github.com/munich-quantum-toolkit/core/pull/1330
[#1328]: https://github.com/munich-quantum-toolkit/core/pull/1328
[#1327]: https://github.com/munich-quantum-toolkit/core/pull/1327
[#1310]: https://github.com/munich-quantum-toolkit/core/pull/1310
[#1301]: https://github.com/munich-quantum-toolkit/core/pull/1301
[#1300]: https://github.com/munich-quantum-toolkit/core/pull/1300
[#1299]: https://github.com/munich-quantum-toolkit/core/pull/1299
[#1294]: https://github.com/munich-quantum-toolkit/core/pull/1294
[#1293]: https://github.com/munich-quantum-toolkit/core/pull/1293
[#1287]: https://github.com/munich-quantum-toolkit/core/pull/1287
[#1283]: https://github.com/munich-quantum-toolkit/core/pull/1283
[#1279]: https://github.com/munich-quantum-toolkit/core/pull/1279
[#1276]: https://github.com/munich-quantum-toolkit/core/pull/1276
[#1271]: https://github.com/munich-quantum-toolkit/core/pull/1271
[#1269]: https://github.com/munich-quantum-toolkit/core/pull/1269
[#1264]: https://github.com/munich-quantum-toolkit/core/pull/1264
[#1263]: https://github.com/munich-quantum-toolkit/core/pull/1263
[#1247]: https://github.com/munich-quantum-toolkit/core/pull/1247
[#1246]: https://github.com/munich-quantum-toolkit/core/pull/1246
[#1243]: https://github.com/munich-quantum-toolkit/core/pull/1243
[#1237]: https://github.com/munich-quantum-toolkit/core/pull/1237
[#1236]: https://github.com/munich-quantum-toolkit/core/pull/1236
[#1235]: https://github.com/munich-quantum-toolkit/core/pull/1235
[#1232]: https://github.com/munich-quantum-toolkit/core/pull/1232
[#1224]: https://github.com/munich-quantum-toolkit/core/pull/1224
[#1223]: https://github.com/munich-quantum-toolkit/core/pull/1223
[#1211]: https://github.com/munich-quantum-toolkit/core/pull/1211
[#1210]: https://github.com/munich-quantum-toolkit/core/pull/1210
[#1209]: https://github.com/munich-quantum-toolkit/core/pull/1209
[#1207]: https://github.com/munich-quantum-toolkit/core/pull/1207
[#1186]: https://github.com/munich-quantum-toolkit/core/pull/1186
[#1181]: https://github.com/munich-quantum-toolkit/core/pull/1181
[#1180]: https://github.com/munich-quantum-toolkit/core/pull/1180
[#1164]: https://github.com/munich-quantum-toolkit/core/pull/1164
[#1157]: https://github.com/munich-quantum-toolkit/core/pull/1157
[#1151]: https://github.com/munich-quantum-toolkit/core/pull/1151
[#1150]: https://github.com/munich-quantum-toolkit/core/pull/1150
[#1148]: https://github.com/munich-quantum-toolkit/core/pull/1148
[#1147]: https://github.com/munich-quantum-toolkit/core/pull/1147
[#1140]: https://github.com/munich-quantum-toolkit/core/pull/1140
[#1139]: https://github.com/munich-quantum-toolkit/core/pull/1139
[#1117]: https://github.com/munich-quantum-toolkit/core/pull/1117
[#1116]: https://github.com/munich-quantum-toolkit/core/pull/1116
[#1108]: https://github.com/munich-quantum-toolkit/core/pull/1108
[#1106]: https://github.com/munich-quantum-toolkit/core/pull/1106
[#1100]: https://github.com/munich-quantum-toolkit/core/pull/1100
[#1099]: https://github.com/munich-quantum-toolkit/core/pull/1099
[#1098]: https://github.com/munich-quantum-toolkit/core/pull/1098
[#1091]: https://github.com/munich-quantum-toolkit/core/pull/1091
[#1089]: https://github.com/munich-quantum-toolkit/core/pull/1089
[#1088]: https://github.com/munich-quantum-toolkit/core/pull/1088
[#1076]: https://github.com/munich-quantum-toolkit/core/pull/1076
[#1075]: https://github.com/munich-quantum-toolkit/core/pull/1075
[#1071]: https://github.com/munich-quantum-toolkit/core/pull/1071
[#1047]: https://github.com/munich-quantum-toolkit/core/pull/1047
[#1042]: https://github.com/munich-quantum-toolkit/core/pull/1042
[#1020]: https://github.com/munich-quantum-toolkit/core/pull/1020
[#1019]: https://github.com/munich-quantum-toolkit/core/pull/1019
[#1010]: https://github.com/munich-quantum-toolkit/core/pull/1010
[#1001]: https://github.com/munich-quantum-toolkit/core/pull/1001
[#996]: https://github.com/munich-quantum-toolkit/core/pull/996
[#984]: https://github.com/munich-quantum-toolkit/core/pull/984
[#982]: https://github.com/munich-quantum-toolkit/core/pull/982
[#975]: https://github.com/munich-quantum-toolkit/core/pull/975
[#973]: https://github.com/munich-quantum-toolkit/core/pull/973
[#964]: https://github.com/munich-quantum-toolkit/core/pull/964
[#959]: https://github.com/munich-quantum-toolkit/core/pull/959
[#934]: https://github.com/munich-quantum-toolkit/core/pull/934
[#933]: https://github.com/munich-quantum-toolkit/core/pull/933
[#932]: https://github.com/munich-quantum-toolkit/core/pull/932
[#931]: https://github.com/munich-quantum-toolkit/core/pull/931
[#930]: https://github.com/munich-quantum-toolkit/core/pull/930
[#926]: https://github.com/munich-quantum-toolkit/core/pull/926
[#921]: https://github.com/munich-quantum-toolkit/core/pull/921
[#913]: https://github.com/munich-quantum-toolkit/core/pull/913
[#912]: https://github.com/munich-quantum-toolkit/core/pull/912
[#911]: https://github.com/munich-quantum-toolkit/core/pull/911
[#908]: https://github.com/munich-quantum-toolkit/core/pull/908
[#900]: https://github.com/munich-quantum-toolkit/core/pull/900
[#897]: https://github.com/munich-quantum-toolkit/core/pull/897
[#895]: https://github.com/munich-quantum-toolkit/core/pull/895
[#893]: https://github.com/munich-quantum-toolkit/core/pull/893
[#892]: https://github.com/munich-quantum-toolkit/core/pull/892
[#886]: https://github.com/munich-quantum-toolkit/core/pull/886
[#885]: https://github.com/munich-quantum-toolkit/core/pull/885
[#883]: https://github.com/munich-quantum-toolkit/core/pull/883
[#882]: https://github.com/munich-quantum-toolkit/core/pull/882
[#879]: https://github.com/munich-quantum-toolkit/core/pull/879
[#878]: https://github.com/munich-quantum-toolkit/core/pull/878
[#877]: https://github.com/munich-quantum-toolkit/core/pull/877
[#866]: https://github.com/munich-quantum-toolkit/core/pull/866
[#860]: https://github.com/munich-quantum-toolkit/core/pull/860
[#859]: https://github.com/munich-quantum-toolkit/core/pull/859
[#858]: https://github.com/munich-quantum-toolkit/core/pull/858
[#849]: https://github.com/munich-quantum-toolkit/core/pull/849
[#847]: https://github.com/munich-quantum-toolkit/core/pull/847
[#846]: https://github.com/munich-quantum-toolkit/core/pull/846
[#842]: https://github.com/munich-quantum-toolkit/core/pull/842
[#839]: https://github.com/munich-quantum-toolkit/core/pull/839
[#838]: https://github.com/munich-quantum-toolkit/core/pull/838
[#832]: https://github.com/munich-quantum-toolkit/core/pull/832
[#831]: https://github.com/munich-quantum-toolkit/core/pull/831
[#822]: https://github.com/munich-quantum-toolkit/core/pull/822
[#817]: https://github.com/munich-quantum-toolkit/core/pull/817
[#810]: https://github.com/munich-quantum-toolkit/core/pull/810
[#807]: https://github.com/munich-quantum-toolkit/core/pull/807
[#802]: https://github.com/munich-quantum-toolkit/core/pull/802
[#798]: https://github.com/munich-quantum-toolkit/core/pull/798
[#789]: https://github.com/munich-quantum-toolkit/core/pull/789
[#763]: https://github.com/munich-quantum-toolkit/core/pull/763
[#762]: https://github.com/munich-quantum-toolkit/core/pull/762
[#758]: https://github.com/munich-quantum-toolkit/core/pull/758
[#741]: https://github.com/munich-quantum-toolkit/core/pull/741
[#724]: https://github.com/munich-quantum-toolkit/core/pull/724
[#662]: https://github.com/munich-quantum-toolkit/core/pull/662
[#543]: https://github.com/munich-quantum-toolkit/core/pull/543
[**@a9b7e70**]: https://github.com/munich-quantum-toolkit/core/pull/798/commits/a9b7e70aaeb532fe8e1e31a7decca86d81eb523f

<!-- Contributor -->

[**@burgholzer**]: https://github.com/burgholzer
[**@ystade**]: https://github.com/ystade
[**@DRovara**]: https://github.com/DRovara
[**@flowerthrower**]: https://github.com/flowerthrower
[**@BertiFlorea**]: https://github.com/BertiFlorea
[**@M-J-Hochreiter**]: https://github.com/M-J-Hochreiter
[**@rotmanjanez**]: https://github.com/rotmanjanez
[**@pehamTom**]: https://github.com/pehamTom
[**@MatthiasReumann**]: https://github.com/MatthiasReumann
[**@denialhaag**]: https://github.com/denialhaag
[**q-inho**]: https://github.com/q-inho
[**@li-mingbao**]: https://github.com/li-mingbao
[**@lavanya-m-k**]: https://github.com/lavanya-m-k
[**@taminob**]: https://github.com/taminob
[**@lsschmid**]: https://github.com/lsschmid
[**@marcelwa**]: https://github.com/marcelwa
[**@lirem101**]: https://github.com/lirem101
[**@Ectras**]: https://github.com/Ectras
[**@simon1hofmann**]: https://github.com/simon1hofmann
[**@keefehuang**]: https://github.com/keefehuang
[**@J4MMlE**]: https://github.com/J4MMlE
[**@rturrado**]: https://github.com/rturrado

<!-- General links -->

[Keep a Changelog]: https://keepachangelog.com/en/1.1.0/
[Common Changelog]: https://common-changelog.org
[QDMI-on-IQM]: https://github.com/iqm-finland/QDMI-on-IQM
[Semantic Versioning]: https://semver.org/spec/v2.0.0.html
[munich-quantum-toolkit]: https://github.com/munich-quantum-toolkit
[PEP 639]: https://peps.python.org/pep-0639/
[PEP 735]: https://peps.python.org/pep-0735/
[CMake presets]: https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html
[munich-quantum-toolkit/workflows]: https://github.com/munich-quantum-toolkit/workflows
[MQT QMAP]: https://github.com/munich-quantum-toolkit/qmap
[MQT QCEC]: https://github.com/munich-quantum-toolkit/qcec

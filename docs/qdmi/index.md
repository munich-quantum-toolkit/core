# QDMI in the MQT

The
[Quantum Device Management Interface (QDMI)](https://munich-quantum-software-stack.github.io/QDMI/)
provides a standardized interface for describing and interacting with quantum
devices. MQT Core supplies a driver, C++ and Python client interfaces, device
implementations, and SDK and HPC integrations.

## What QDMI standardizes

QDMI defines a C interface between software clients and device implementations.
Its [specification](https://munich-quantum-software-stack.github.io/QDMI/)
covers three parts of an interaction:

- **Sessions:** configure and initialize access to a device.
- **Queries:** inspect device properties, sites, supported operations, and
  available calibration data.
- **Jobs:** set a program and its parameters, submit it, check its status,
  cancel it when supported, and retrieve results.

A QDMI device can represent a simulator, a physical QPU, or a cloud service. The
interface exposes each device's capabilities and accepted program formats;
clients must use those queries to prepare compatible jobs.

## How MQT Core uses QDMI

| Layer                                                                   | Responsibility                                                                          |
| ----------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| {doc}`Compiler <../mlir/target_compilation>`                            | Query a target's capabilities and compile a compatible program.                         |
| {doc}`Driver and client interfaces <driver>`                            | Load device libraries and provide C++ and Python access to sessions, queries, and jobs. |
| Device implementation                                                   | Translate QDMI calls into simulator operations or a provider's API.                     |
| {doc}`SDK adapters <qdmi_backend>` and {doc}`Slurm integration <slurm>` | Connect user workflows and resource management to QDMI devices.                         |

Compilation and submission remain separate: `compile_program` prepares a
`CompiledProgram`; `submit_program` submits it to a matching execution device.
The {doc}`compilation and execution guide <../compilation/index>` introduces
this workflow. The
{doc}`hardware compilation tutorial <../tutorials/hardware_compilation>` shows
both steps with bundled DDSIM; the {doc}`QIR guide <../qir/index>` also
retrieves QIR output records through the same job API.

Work through the {doc}`QDMI tutorial <../tutorials/qdmi_execution>` to discover
capabilities, reuse a compiled program, and relate shots, counts, and simulator
states.

## Choose a guide

- **Run a program:** start with {doc}`../getting_started` and the
  {doc}`DDSIM device <ddsim_device>`, which executes supported OpenQASM and QIR.
- **Discover or configure devices:** use the {doc}`driver` and
  {doc}`configuration` guides.
- **Compile for hardware:** the {doc}`superconducting models <sc_device>` supply
  topology, native operations, and calibration. They do not execute programs.
- **Use another interface:** connect through {doc}`Qiskit <qdmi_backend>`,
  {doc}`PennyLane <pennylane_device>`, or {doc}`Slurm <slurm>`.
- **Implement a device:** follow the
  [QDMI device interface](https://munich-quantum-software-stack.github.io/QDMI/)
  and use MQT Core's bundled devices as implementation examples.

## Further reading and device implementations

The
[QDMI specification and examples](https://munich-quantum-software-stack.github.io/QDMI/)
are the reference for implementing the interface. These case studies explain how
that interface maps to real services and hardware:

- **Amazon Braket:**
  [Standardizing Access to Heterogeneous Quantum Backends](https://arxiv.org/abs/2603.05138)
  treats the cloud service as one QDMI device, with backend selection and a job
  lifecycle covering authentication, submission, and result retrieval. It also
  discusses constraints imposed by the underlying service.
- **IQM systems:**
  [Practical HPCQC Integration with QDMI](https://arxiv.org/abs/2604.19869)
  describes capability and calibration queries, job handling, Qiskit workflows,
  and Slurm integration for cloud and on-premise deployment. The
  [QDMI-on-IQM implementation](https://github.com/iqm-finland/QDMI-on-IQM)
  provides the corresponding device library.

These integrations share the QDMI boundary while retaining provider-specific
program formats, authentication, and scheduling constraints. See
{doc}`configuration` to register an external device library with MQT Core.

```{toctree}
:maxdepth: 1
:caption: Table of Contents

QDMI Driver <driver>
QDMI device configuration <configuration>
DDSIM QDMI Device <ddsim_device>
SC QDMI Device <sc_device>
Slurm integration <slurm>
QDMI-Qiskit Backend <qdmi_backend>
PennyLane interface for QDMI devices <pennylane_device>
```

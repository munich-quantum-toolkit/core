# QDMI in the MQT

The
[Quantum Device Management Interface (QDMI)](https://munich-quantum-software-stack.github.io/QDMI/)
provides a standardized interface for describing and interacting with quantum
devices. MQT Core supplies a driver, C++ and Python client interfaces, device
implementations, and SDK and HPC integrations.

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

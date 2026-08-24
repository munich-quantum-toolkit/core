# QDMI in the MQT

The
[Quantum Device Management Interface (QDMI)](https://munich-quantum-software-stack.github.io/QDMI/)
provides a standardized interface for describing and interacting with quantum
devices. This part of MQT Core contains the implementation of QDMI's different
components, such as a [QDMI driver](driver.md), a
[QDMI device for Superconducting Systems](sc_device.md), and a
[QDMI device for a Classical Quantum Circuit Simulator](ddsim_device).

```{toctree}
:maxdepth: 1
:caption: Table of Contents

SC QDMI Device <sc_device>
DDSIM QDMI Device <ddsim_device>
QDMI Client and Driver runtime <driver>
QDMI device configuration <configuration>
Slurm integration <slurm>
QDMI-Qiskit Backend <qdmi_backend>
PennyLane interface for QDMI devices <pennylane_device>
```

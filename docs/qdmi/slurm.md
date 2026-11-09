# Slurm integration with MQSC Orchestrator

The Slurm Playground has moved to
[MQSC Orchestrator](https://github.com/munich-quantum-software/orchestrator),
where development continues. Starting with MQT Core 4.2, the playground, its
deployment infrastructure, and its integration tests live in that repository.

Use MQSC Orchestrator for deployment instructions, examples, and validation of
hybrid quantum-classical workflows with Slurm. Follow its setup guide from a
separate checkout instead of using `examples/slurm` in MQT Core.

MQT Core provides the {doc}`QDMI APIs <driver>` and {doc}`Qiskit <qdmi_backend>`
and {doc}`PennyLane <pennylane_device>` adapters used by these workflows.

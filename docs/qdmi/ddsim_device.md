# MQT Core DD-based Simulator QDMI Device

## Objective

MQT Core provides a QDMI device that is powered by a classical quantum circuit simulator based on decision diagrams (see [the documentation of the DD Package](../dd_package.md)).
This functionality is exposed through the QDMI interface as a device, which can be used to classically simulate quantum programs.

## Capabilities

The simulator device supports all operations that our [MQT Core IR](../mqt_core_ir.md) supports and takes programs in either OpenQASM 2 or OpenQASM 3 format.
It can either be used for performing weak simulation, i.e., sampling from the distribution produced by the circuit, or for performing strong simulation, i.e., computing a representation of the full state vector.
To switch between these two modes, either set the `QDMI_DEVICE_JOB_PARAMETER_SHOTSNUM` parameter to the desired number of shots for weak simulation or to `0` for strong simulation.

Under the hood, the QDMI device uses the MQT Core OpenQASM parser (see {cpp:func}`qasm3::Importer::imports`) to parse the program into a {cpp:class}`qc::QuantumComputation` object.
That circuit is then passed either to the {cpp:func}`dd::sample` or {cpp:func}`dd::simulate` function, depending on the mode.
Consult the respective documentation for more details and limitations.

The device implements the full QDMI job interface (except for the `QDMI_JOB_RESULT_SHOTS` result format not supported by the simulator).

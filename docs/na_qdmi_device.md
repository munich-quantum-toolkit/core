---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
---

# MQT Core Neutral Atom QDMI Device

## Objective

Compilation passes throughout MQT need information about the target device.
The Neutral Atom QDMI device provides a uniform way to provide the necessary information.
It defines a representation to easily provide static information in the form of a file.

<!-- todo: Explain how to use the device. -->

## Describing a Device

The basis of a such device implementation is a specification in a JSON file.
The structure of this JSON files is defined via `struct`s in [`Generator.hpp`](https://github.com/munich-quantum-toolkit/core/tree/main/include/mqt-core/na/device/Generator.hpp) that define functions to serialize and deserialize the data using the [nlohmann/json](https://json.nlohmann.me) library.
During compilation, this JSON file is parsed and the corresponding C++ code is produced by an application (see [`App.cpp`](https://github.com/munich-quantum-toolkit/core/tree/main/src/na/device/App.cpp)) for the actual QDMI device implementation.
The C++ code is then compiled to a library that can be used by the QDMI driver.

### JSON Schema

An example instance of a device JSON file can be found in [`json/na/device.json`](https://github.com/munich-quantum-toolkit/core/tree/main/json/na/device.json).

#### Top-Level Fields

- **`name`** _(string)_: The name of the device.
- **`numQubits`** _(integer)_: The total number of qubits in the device.
- **`traps`** _(array)_: A list of traps defining the qubit lattice structure.
- **`minAtomDistance`** _(number)_: The minimum distance between atoms in the lattice.
- **`globalSingleQubitOperations`** _(array)_: A list of global single-qubit operations supported by the device.
- **`globalMultiQubitOperations`** _(array)_: A list of global multi-qubit operations supported by the device.
- **`localSingleQubitOperations`** _(array)_: A list of local single-qubit operations supported by the device.
- **`localMultiQubitOperations`** _(array)_: A list of local multi-qubit operations supported by the device.
- **`shuttlingUnits`** _(array)_: A list of shuttling units for moving qubits.
- **`decoherenceTimes`** _(object)_: Decoherence times for the qubits.
- **`lengthUnit`** _(object)_: The unit of length used in the file.
- **`timeUnit`** _(object)_: The unit of time used in the file.

---

#### Detailed Field Descriptions

##### `traps` _(array of objects)_:

- **`latticeOrigin`** _(object)_: The origin of the lattice.
  - **`x`** _(number)_: X-coordinate of the origin.
  - **`y`** _(number)_: Y-coordinate of the origin.
- **`latticeVector1`** _(object)_: Defines one lattice vector.
  - **`x`** _(number)_: X-component of the vector.
  - **`y`** _(number)_: Y-component of the vector.
- **`latticeVector2`** _(object)_: Defines the other lattice vector.
  - **`x`** _(number)_: X-component of the vector.
  - **`y`** _(number)_: Y-component of the vector.
- **`sublatticeOffsets`** _(array of objects)_: Offsets for sublattices.
  - **`x`** _(number)_: X-offset.
  - **`y`** _(number)_: Y-offset.
- **`extent`** _(object)_: The extent of the trap area.
  - **`origin`** _(object)_: The origin of the trap area.
    - **`x`** _(number)_: X-coordinate of the origin.
    - **`y`** _(number)_: Y-coordinate of the origin.
  - **`size`** _(object)_: Size of the trap area.
    - **`width`** _(number)_: Width of the trap area.
    - **`height`** _(number)_: Height of the trap area.

##### `globalSingleQubitOperations` and `localSingleQubitOperations` _(array of objects)_:

- **`name`** _(string)_: The name of the operation.
- **`region`** _(object)_: The region where the operation is applied.
  - **`origin`** _(object)_: The origin of the region.
    - **`x`** _(number)_: X-coordinate of the origin.
    - **`y`** _(number)_: Y-coordinate of the origin.
  - **`size`** _(object)_: The size of the region.
    - **`width`** _(number)_: Width of the region.
    - **`height`** _(number)_: Height of the region.
- **`duration`** _(number)_: Duration of the operation.
- **`fidelity`** _(number)_: Fidelity of the operation.
- **`numParameters`** _(integer)_: Number of parameters for the operation.
- **`numRows`** and **`numRows`** _(integer, only `localSingleQubitOperations`)_: Number of rows and columns, respectively, in the operation.

##### `globalMultiQubitOperations` and `localMultiQubitOperations` _(array of objects)_:

- **`name`** _(string)_: The name of the operation.
- **`region`** _(object)_: The region where the operation is applied (same structure as above).
- **`interactionRadius`** _(number)_: Radius of interaction for the operation.
- **`blockingRadius`** _(number)_: Blocking radius for the operation.
- **`numQubits`** _(integer)_: Number of qubits involved in the operation.
- **`numParameters`** _(integer)_: Number of parameters for the operation.
- **`duration`** _(number)_: Duration of the operation.
- **`fidelity`** _(number)_: Fidelity of the operation.
- **`idlingFidelity`** _(number, only `globalMultiQubitOperations`)_: Fidelity when qubits are idle.
- **`numRows`** and **`numRows`** _(integer, only `localMultiQubitOperations`)_: Number of rows and columns, respectively, in the operation.

##### `shuttlingUnits` _(array of objects)_:

- **`name`** _(string)_: The name of the shuttling unit.
- **`region`** _(object)_: The region where the shuttling unit operates (same structure as above).
- **`numRows`** _(integer)_: Number of rows in the shuttling unit.
- **`numColumns`** _(integer)_: Number of columns in the shuttling unit.
- **`movingSpeed`** _(number)_: Speed of movement.
- **`loadDuration`** _(number)_: Duration to load a qubit.
- **`storeDuration`** _(number)_: Duration to store a qubit.
- **`loadFidelity`** _(number)_: Fidelity of loading a qubit.
- **`storeFidelity`** _(number)_: Fidelity of storing a qubit.
- **`numParameters`** _(integer)_: Number of parameters for the shuttling unit.

##### `decoherenceTimes` _(object)_:

- **`t1`** _(number)_: Relaxation time (T1) in the specified time unit.
- **`t2`** _(number)_: Dephasing time (T2) in the specified time unit.

##### `lengthUnit` and `timeUnit` _(objects)_:

- **`value`** _(number)_: The scaling factor for the unit.
- **`unit`** _(string)_: The unit of measurement (e.g., "um" for micrometers, "us" for microseconds).

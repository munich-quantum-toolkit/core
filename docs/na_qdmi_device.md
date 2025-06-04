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

## File Format

```yaml
name: <str>
decoherence_times:
  t1: <uint>
  t2: <uint>
min_atom_distance: <uint>
length_unit:
  value: <uint>
  unit: <str>
time_unit:
  value: <uint>
  unit: <str>
traps:
  - lattice_origin:
      x: <int>
      y: <int>
    lattice_vectors:
      - x: <int>
        y: <int>
        repeat: <uint>
      - x: <int>
        y: <int>
        repeat: <uint>
    sublattice_offsets:
      - x: <int>
        y: <int>
      ...
  ...
global_single_qubit_operations:
  - name: <str>
    num_parameters: <uint>
    duration: <float>
    fidelity: <float>
    region:
      origin:
        x: <int>
        y: <int>
      size:
        width: <int>
        height: <int>
  ...
global_multi_qubit_operations:
  - name: <str>
    num_qubits: <num>
    num_parameters: <num>
    duration: <duration>
    fidelity: <fidelity>
    idling_fidelity: <fidelity>
    interaction_radius: <radius>
    blocking_radius: <radius>
    region:
      origin:
        x: <int>
        y: <int>
      size:
        width: <int>
        height: <int>
  ...
local_single_qubit_operations:
  - name: <name>
    num_parameters: <num>
    duration: <duration>
    fidelity: <fidelity>
    num_rows: <num>
    num_columns: <num>
    region:
      origin:
        x: <int>
        y: <int>
      size:
        width: <int>
        height: <int>
  ...
local_multi_qubit_operations:
  - name: <name>
    num_qubits: <num>
    num_parameters: <num>
    duration: <duration>
    fidelity: <fidelity>
    interaction_radius: <radius>
    blocking_radius: <radius>
    num_rows: <num>
    num_columns: <num>
    region:
      origin:
        x: <int>
        y: <int>
      size:
        width: <int>
        height: <int>
  ...
shuttling_units:
  - name: <name>
    num_parameters: <num>
    load_duration: <duration>
    load_fidelity: <fidelity>
    moving_speed: <duration>
    store_duration: <duration>
    store_fidelity: <fidelity>
    num_rows: <num>
    num_columns: <num>
    region:
      origin:
        x: <int>
        y: <int>
      size:
        width: <int>
        height: <int>
  ...
```
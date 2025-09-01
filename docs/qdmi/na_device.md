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
The Neutral Atom [QDMI](https://munich-quantum-software-stack.github.io/QDMI/) device provides a uniform way to provide the necessary information for neutral atom-based quantum devices.
It defines a representation to easily provide static information in the form of a JSON file.

## Describing a Device

The basis of a such device implementation is a specification in a JSON file.
The structure of this JSON file is defined by the {cpp:class}`na::Device` struct.
The struct defines functions to serialize and deserialize the data using the [nlohmann/json](https://json.nlohmann.me) library.
During compilation, this JSON file is parsed and the corresponding C++ code is produced by an application (see `src/na/device/App.cpp`) for the actual QDMI device implementation.
The C++ code is then compiled to a library that can be used by the QDMI driver.
An example instance of a device JSON file can be found in `json/na/device.json`.

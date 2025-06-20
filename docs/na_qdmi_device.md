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

The basis of a device implementation is a specification in a JSON file.
These JSON files are defined on the basis of a protocol buffer schema that can be found in `proto/na/device.proto`.
During compilation, this JSON file is parsed and the corresponding C++ code is produced for the actual QDMI device implementation.
The C++ code is then compiled to a library that can be used by the QDMI driver.

### JSON Schema

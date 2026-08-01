# MQT MLIR Agent Guide

This file supplements the repository-level `AGENTS.md` for work under `mlir/`.

## C++ Identifiers

- Do not use `module` as a C++ variable or parameter name because it conflicts
  with the C++20 keyword. Use `moduleOp` for `mlir::ModuleOp` values.

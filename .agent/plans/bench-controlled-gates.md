# Controlled composite gates

Status: complete.

## Scope and ownership

The versioned circuit adapter must retain the complete definition of controlled
gates whose targets span more wires than their base operation. Generic control
unwrapping applies only when operand widths agree.

Multi-control decomposition expands composite controls only when they meet its
width and target-support conditions. Reuse the existing control-unrolling
implementation and control canonicalization to preserve nested controls, operand
order, and controlled global phases. Add no pipeline passes or early
single-qubit merging: symbolic merging expands classical IR and slows later
passes. Two-qubit controls retain their existing handling.

## Validation

Compiler (230), decomposition (313), Core MLIR Python (720), and Bench (354)
tests pass. Stub generation passed for the adapter change. Repository lint and
full changed-file C++ lint pass.

Against parent b897f04, medians from 21 warmed samples per case differ by −2.5%
to +2.8% for all-to-all compilation/synthesis and −3.6% to +3.8% for linear
routing. Tests cover GHZ, QFT, numeric and symbolic rotations, and native gates
on 12–32 qubits; quantum operation counts match the baseline. Measured with
Python 3.13 and MLIR 23.1.1, MinSizeRel on macOS ARM64, seed 10 and four mapping
trials. No additional pipeline passes or early single-qubit merging remain.
These measurements do not establish zero cost for every circuit.

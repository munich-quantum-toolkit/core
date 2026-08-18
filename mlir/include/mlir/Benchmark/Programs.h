/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <cstdint>

namespace mlir::qc {
class QCProgramBuilder;
} // namespace mlir::qc

namespace mqt::benchmark {

using namespace mlir;

/**
 * @brief Signature of a benchmark program builder
 *
 * @details Each builder adds one benchmark to @p b and returns the values that
 * the program returns, which are its classical registers. The meaning of @p n
 * follows the benchmark's own size parameter, as documented per program.
 */
using ProgramBuilder = SmallVector<Value> (*)(qc::QCProgramBuilder& b,
                                              uint64_t n);

/// A benchmark program and the name it is generated under.
struct Benchmark {
  llvm::StringRef name;
  ProgramBuilder build;
  /// The smallest @p n the program accepts.
  uint64_t minimumSize;
};

/// Returns every benchmark program.
[[nodiscard]] SmallVector<Benchmark> benchmarks();

// --- Programs ------------------------------------------------------------- //

/// GHZ state preparation with a linear chain of CX gates on @p n qubits.
SmallVector<Value> ghzLinear(qc::QCProgramBuilder& b, uint64_t n);

/// GHZ state preparation with all CX gates sharing one control, on @p n qubits.
SmallVector<Value> ghzStar(qc::QCProgramBuilder& b, uint64_t n);

/// Quantum Fourier transform on @p n qubits.
SmallVector<Value> qft(qc::QCProgramBuilder& b, uint64_t n);

/// Quantum phase estimation on @p n qubits: n-1 counting qubits and one
/// eigenstate ancilla.
SmallVector<Value> qpe(qc::QCProgramBuilder& b, uint64_t n);

/// Iterative quantum Fourier transform on one qubit with @p n result bits.
SmallVector<Value> iqft(qc::QCProgramBuilder& b, uint64_t n);

/// Iterative quantum phase estimation on two qubits with @p n bits of
/// precision.
SmallVector<Value> iqpe(qc::QCProgramBuilder& b, uint64_t n);

/// Grover's search algorithm on @p n qubits: n-1 search qubits and one flag.
SmallVector<Value> grover(qc::QCProgramBuilder& b, uint64_t n);

/// Uniformly controlled RY rotations on @p n qubits: n-1 controls and one
/// target.
SmallVector<Value> multiplexer(qc::QCProgramBuilder& b, uint64_t n);

/// Quantum teleportation. The program has a fixed size, so @p n is ignored.
SmallVector<Value> teleportation(qc::QCProgramBuilder& b, uint64_t n);

/// Block encoding of a linear combination of unitaries on @p n qubits: two
/// ancillas holding the weights and n-2 system qubits.
SmallVector<Value> blockEncoding(qc::QCProgramBuilder& b, uint64_t n);

/// A chain of Toffoli gates across @p n qubits.
SmallVector<Value> toffoliHeavy(qc::QCProgramBuilder& b, uint64_t n);

/// Quantum fan-out of one qubit over @p n qubits, followed by a parallel layer.
SmallVector<Value> fanOut(qc::QCProgramBuilder& b, uint64_t n);

/// Draper adder of two quantum registers holding @p n / 2 qubits each.
SmallVector<Value> qftAdderQuantum(qc::QCProgramBuilder& b, uint64_t n);

/// Draper adder of a classical constant to one register of @p n qubits.
SmallVector<Value> qftAdderClassical(qc::QCProgramBuilder& b, uint64_t n);

/// Hardware-efficient VQE ansatz on @p n qubits with a fixed layer count.
SmallVector<Value> vqeAnsatz(qc::QCProgramBuilder& b, uint64_t n);

/// QAOA on a ring of @p n qubits with a fixed layer count.
SmallVector<Value> qaoa(qc::QCProgramBuilder& b, uint64_t n);

/// Controlled multiplication modulo a constant on @p n qubits: one control, one
/// ancilla, and the rest split between the multiplier and the accumulator.
SmallVector<Value> controlledMultiplyModN(qc::QCProgramBuilder& b, uint64_t n);

/// Grover's search on @p n qubits that repeats until a weakly coupled probe
/// reports the marked state.
SmallVector<Value> groverWeakMeasurement(qc::QCProgramBuilder& b, uint64_t n);

/// VQE on @p n qubits, with the optimization loop driven by the measurements.
SmallVector<Value> vqe(qc::QCProgramBuilder& b, uint64_t n);

/// Maximum likelihood amplitude estimation on @p n qubits, with one round per
/// power of the Grover operator.
SmallVector<Value> mlqae(qc::QCProgramBuilder& b, uint64_t n);

} // namespace mqt::benchmark

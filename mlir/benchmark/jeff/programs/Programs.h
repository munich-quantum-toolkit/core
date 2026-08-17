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

namespace mqt::jeff::benchmarks {

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

/// Returns every benchmark the generator knows about.
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

} // namespace mqt::jeff::benchmarks

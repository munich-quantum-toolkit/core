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

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <cstdint>

namespace mlir::qc {
class QCProgramBuilder;
} // namespace mlir::qc

namespace mqt::jeff::benchmarks {

/**
 * @brief Signature of a benchmark program builder
 *
 * @details Each builder adds one benchmark to @p b and returns the values that
 * the program returns, which are its classical registers. The meaning of @p n
 * follows the benchmark's own size parameter, as documented per program.
 */
using ProgramBuilder = llvm::SmallVector<mlir::Value> (*)(
    mlir::qc::QCProgramBuilder& b, uint64_t n);

/// A benchmark program and the name it is generated under.
struct Benchmark {
  llvm::StringRef name;
  ProgramBuilder build;
  /// The smallest @p n the program accepts.
  uint64_t minimumSize;
};

/// Returns every benchmark the generator knows about.
[[nodiscard]] llvm::SmallVector<Benchmark> benchmarks();

// --- Programs ------------------------------------------------------------- //

/// GHZ state preparation with a linear chain of CX gates on @p n qubits.
llvm::SmallVector<mlir::Value> ghzLinear(mlir::qc::QCProgramBuilder& b,
                                         uint64_t n);

/// GHZ state preparation with all CX gates sharing one control, on @p n qubits.
llvm::SmallVector<mlir::Value> ghzStar(mlir::qc::QCProgramBuilder& b,
                                       uint64_t n);

/// Quantum Fourier transform on @p n qubits.
llvm::SmallVector<mlir::Value> qft(mlir::qc::QCProgramBuilder& b, uint64_t n);

/// Quantum phase estimation on @p n qubits: n-1 counting qubits and one
/// eigenstate ancilla.
llvm::SmallVector<mlir::Value> qpe(mlir::qc::QCProgramBuilder& b, uint64_t n);

/// Iterative quantum Fourier transform on one qubit with @p n result bits.
llvm::SmallVector<mlir::Value> iqft(mlir::qc::QCProgramBuilder& b, uint64_t n);

/// Iterative quantum phase estimation on two qubits with @p n bits of
/// precision.
llvm::SmallVector<mlir::Value> iqpe(mlir::qc::QCProgramBuilder& b, uint64_t n);

/// Grover's search algorithm on @p n qubits: n-1 search qubits and one flag.
llvm::SmallVector<mlir::Value> grover(mlir::qc::QCProgramBuilder& b,
                                      uint64_t n);

/// Uniformly controlled RY rotations on @p n qubits: n-1 controls and one
/// target.
llvm::SmallVector<mlir::Value> multiplexer(mlir::qc::QCProgramBuilder& b,
                                           uint64_t n);

/// Quantum teleportation. The program has a fixed size, so @p n is ignored.
llvm::SmallVector<mlir::Value> teleportation(mlir::qc::QCProgramBuilder& b,
                                             uint64_t n);

} // namespace mqt::jeff::benchmarks

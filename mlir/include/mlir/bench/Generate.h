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

#include "mlir/Compiler/Programs.h"

#include <optional>
#include <string>
#include <string_view>

namespace mqt::bench {
class BV;
class GHZ;
class Grover;
class Multiplexer;
class QFT;
class QPE;
class Teleportation;

/// A generated program and the normalized semantic instance that produced it.
struct GeneratedBenchmark {
  std::string benchmarkId;
  std::string caseId;
  std::string manifestJSON;
  mlir::QCProgram program;
};

/// Generate a configured Bernstein--Vazirani benchmark.
[[nodiscard]] std::optional<mlir::QCProgram> generate(const BV& benchmark);

/// Generate the QC program for a configured GHZ benchmark.
[[nodiscard]] std::optional<mlir::QCProgram> generate(const GHZ& benchmark);

/// Generate the QC program for a configured Grover benchmark.
[[nodiscard]] std::optional<mlir::QCProgram> generate(const Grover& benchmark);

/// Generate the QC program for a configured quantum multiplexer benchmark.
[[nodiscard]] std::optional<mlir::QCProgram>
generate(const Multiplexer& benchmark);

/// Generate a configured quantum Fourier-transform benchmark.
[[nodiscard]] std::optional<mlir::QCProgram> generate(const QFT& benchmark);

/// Generate the QC program for a configured QPE benchmark.
[[nodiscard]] std::optional<mlir::QCProgram> generate(const QPE& benchmark);

/// Generate the quantum teleportation benchmark.
[[nodiscard]] std::optional<mlir::QCProgram>
generate(const Teleportation& benchmark);

/// Parse a benchmark instance specification and generate the benchmark.
[[nodiscard]] std::optional<GeneratedBenchmark>
generate(std::string_view instanceSpecificationJSON,
         std::string_view source = "<instance-specification>");

} // namespace mqt::bench

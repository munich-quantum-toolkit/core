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

#include "mqt/Compiler/QDMIAdapter.h"
#include "mqt/Compiler/TargetEnvironment.h"
#include "qdmi/QDMI.hpp"

#include "llvm/ADT/ArrayRef.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace mqt::runtime {

using JobParameter =
    std::variant<std::string, bool, int, double, std::vector<std::byte>>;
using JobParameters = std::array<std::optional<JobParameter>, 5>;
using BorrowedJobParameters =
    std::array<std::optional<qdmi::CustomJobParameter>, 5>;

/// Own a provider session, target snapshot, and default submission parameters.
class Context {
public:
  Context(qdmi::Device device, mlir::TargetEnvironment environment,
          JobParameters parameters = {});
  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;
  [[nodiscard]] const mlir::TargetEnvironment& environment() const noexcept {
    return environment_;
  }
  /// Validate one format group, taking a fresh snapshot for foreign programs.
  [[nodiscard]] llvm::Expected<mlir::TargetEnvironment>
  prepare(llvm::ArrayRef<const mlir::CompiledProgram*> programs,
          bool retained) const;
  /// Submit compatible programs with defaults or a complete per-call override.
  [[nodiscard]] llvm::Expected<std::optional<qdmi::Job>>
  trySubmit(llvm::ArrayRef<const mlir::CompiledProgram*> programs,
            int64_t shots, const BorrowedJobParameters* parameters = nullptr,
            const mlir::TargetEnvironment* destination = nullptr) const;
  [[nodiscard]] llvm::Expected<qdmi::Job>
  submit(const mlir::CompiledProgram& program, int64_t shots) const;

private:
  qdmi::Device device_;
  JobParameters parameters_;
  mlir::TargetEnvironment environment_;
};

} // namespace mqt::runtime

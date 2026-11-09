/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "QDMIContext.h"

#include "qdmi/common/Common.hpp"

#include "llvm/Support/Error.h"

#include <array>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace mqt::runtime {

Context::Context(qdmi::Device device, mlir::TargetEnvironment environment,
                 JobParameters parameters)
    : device_(std::move(device)), parameters_(std::move(parameters)),
      environment_(std::move(environment)) {}

llvm::Expected<mlir::TargetEnvironment>
Context::prepare(llvm::ArrayRef<const mlir::CompiledProgram*> programs,
                 bool retained) const {
  return mlir::prepareProgramSubmission(device_, programs,
                                        retained ? &environment_ : nullptr);
}

llvm::Expected<std::optional<qdmi::Job>>
Context::trySubmit(llvm::ArrayRef<const mlir::CompiledProgram*> programs,
                   int64_t shots, const BorrowedJobParameters* parameters,
                   const mlir::TargetEnvironment* destination) const {
  BorrowedJobParameters defaults;
  if (parameters == nullptr) {
    for (size_t i = 0; i < parameters_.size(); ++i) {
      if (parameters_[i]) {
        defaults[i] = std::visit(
            [](const auto& value) -> qdmi::CustomJobParameter {
              if constexpr (std::is_same_v<std::decay_t<decltype(value)>,
                                           std::vector<std::byte>>) {
                return std::span<const std::byte>(value);
              } else {
                return value;
              }
            },
            *parameters_[i]);
      }
    }
    parameters = &defaults;
  }
  const auto& borrowed = *parameters;
  return mlir::trySubmitPrograms(
      device_, programs, shots, borrowed[0], borrowed[1], borrowed[2],
      borrowed[3], borrowed[4],
      destination == nullptr ? &environment_ : destination);
}

llvm::Expected<qdmi::Job> Context::submit(const mlir::CompiledProgram& program,
                                          int64_t shots) const {
  auto job = trySubmit({&program}, shots);
  if (!job) {
    return job.takeError();
  }
  if (!*job) {
    return llvm::createStringError(
        "Device declined the compiled program before submission.");
  }
  return std::move(**job);
}

} // namespace mqt::runtime

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
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <system_error>
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

BorrowedJobParameters
Context::borrow(const BorrowedJobParameters* parameters) const {
  if (parameters != nullptr) {
    return *parameters;
  }
  BorrowedJobParameters defaults;
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
  return defaults;
}

llvm::Expected<std::optional<qdmi::Job>>
Context::trySubmit(llvm::ArrayRef<const mlir::CompiledProgram*> programs,
                   size_t shots, const BorrowedJobParameters* parameters,
                   const mlir::TargetEnvironment* destination) const {
  return mlir::trySubmitPrograms(
      device_, programs, {.shots = shots, .custom = borrow(parameters)},
      destination == nullptr ? &environment_ : destination);
}

llvm::Expected<bool>
Context::supports(QDMI_Program_Format format, size_t count, size_t shots,
                  const BorrowedJobParameters* parameters) const {
  return device_.supportsJob(format, count,
                             {.shots = shots, .custom = borrow(parameters)});
}

llvm::Expected<qdmi::Job> Context::submit(const mlir::CompiledProgram& program,
                                          size_t shots) const {
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

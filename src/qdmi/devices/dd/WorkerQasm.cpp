/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "WorkerQasm.hpp"

#include "dd/DDDefinitions.hpp"
#include "dd/Export.hpp"
#include "dd/Package.hpp"
#include "mqt/Compiler/Programs.h"
#include "mqt/Dialect/MQT/IR/MQTDialect.h"
#include "mqt/Dialect/QCO/Utils/DDFunctionality.h"
#include "mqt/Target/OpenQASM/TranslateOpenQASMToQC.h"
#include "qdmi/common/Diagnostics.hpp"

#include "WorkerProtocol.hpp"

#include "llvm/Support/Error.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string_view>
#include <utility>

namespace {
[[nodiscard]] auto parseQASMToQCO(const std::string_view source)
    -> llvm::FailureOr<mlir::QCOProgram> {
  auto context = mlir::createCompilerContext();
  auto moduleOp = mlir::qc::translateOpenQASMToQC(source, context.get());
  if (!moduleOp) {
    return llvm::failure();
  }
  auto qcProgram =
      mlir::QCProgram::fromModule(std::move(context), std::move(moduleOp));
  if (llvm::failed(qcProgram)) {
    return llvm::failure();
  }
  return std::move(*qcProgram).intoQCO();
}

} // namespace

void qdmi::dd::mqtCoreRunWorkerQasm(
    const WorkerRequest& request, WorkerResponse& response,
    const std::function<size_t(size_t)>& requestWorkers) {
  auto qcoProgram = parseQASMToQCO(request.program);
  if (llvm::failed(qcoProgram)) {
    return;
  }
  // NOLINTNEXTLINE(misc-const-correctness): MLIR handles remain mutable.
  auto entryPoint = mlir::mqt::getEntryPoint(qcoProgram->module());
  if (entryPoint == nullptr) {
    qdmi::diagnostics::error("QCO program has no entry point");
    return;
  }

  mlir::qco::DDSamplingState retainedState;
  const auto shots = static_cast<size_t>(request.shots);
  if (shots != 0) {
    const auto maxWorkers = static_cast<size_t>(request.maxWorkers);
    const auto workers =
        request.automaticWorkers
            ? std::min(maxWorkers,
                       std::clamp(shots / 256, size_t{1}, size_t{8}))
            : maxWorkers;
    auto counts = mlir::qco::sample(
        entryPoint, shots, request.seed.value_or(0),
        mlir::qco::DDArgumentBindings{}, nullptr, &retainedState,
        {.maxQubits = request.maxQubits}, workers, requestWorkers);
    if (llvm::failed(counts)) {
      return;
    }
    response.counts = std::move(*counts);
  } else {
    retainedState.dd = llvm::cantFail(::dd::Package::create(0));
    auto state = mlir::qco::simulateStatevector(
        entryPoint, *retainedState.dd, mlir::qco::DDArgumentBindings{},
        {.maxQubits = request.maxQubits});
    if (llvm::failed(state)) {
      return;
    }
    retainedState.state = *state;
  }
  if (retainedState.dd) {
    const auto state = retainedState.state;
    response.qubits =
        state.isTerminal() ? 0 : static_cast<uint32_t>(state.p->v) + 1;
    llvm::raw_string_ostream bytes(response.state.emplace());
    ::dd::serialize(state, bytes, true);
  }
  response.succeeded = true;
}

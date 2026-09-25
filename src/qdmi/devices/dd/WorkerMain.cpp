/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/DDDefinitions.hpp"
#include "dd/Export.hpp"
#include "dd/Package.hpp"
#include "mqt/Compiler/Programs.h"
#include "mqt/Dialect/MQT/IR/MQTDialect.h"
#include "mqt/Dialect/QC/Translation/TranslateOpenQASMToQC.h"
#include "mqt/Dialect/QCO/Utils/DDFunctionality.h"
#include "mqt/Dialect/QIR/Execution/JIT/Session.h"
#include "mqt/Dialect/QIR/Execution/Runtime/Runtime.h"
#include "mqt/Support/Diagnostics.h"

#include "WorkerProtocol.hpp"
#include "support/Diagnostics.hpp"

#include "qdmi/constants.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_socket_stream.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace {
[[nodiscard]] auto parseQASMToQCO(const std::string_view source)
    -> std::optional<mlir::QCOProgram> {
  auto context = mlir::createCompilerContext();
  context->disableMultithreading();
  context->getDiagEngine().registerHandler([](mlir::Diagnostic& diagnostic) {
    mqt::emitDiagnostic(mlir::toNativeDiagnostic(
        diagnostic, mqt::ErrorCategory::InvalidArgument));
    return mlir::success();
  });
  auto moduleOp = mlir::qc::translateOpenQASMToQC(source, context.get());
  if (!moduleOp) {
    return std::nullopt;
  }
  auto qcProgram =
      mlir::QCProgram::fromModule(std::move(context), std::move(moduleOp));
  if (!qcProgram) {
    return std::nullopt;
  }
  return std::move(*qcProgram).intoQCO();
}

struct Execution {
  std::string_view program_;
  size_t numShots_;
  std::optional<uint64_t> seed_;
  bool captureQIROutput_;
  std::optional<std::string> qirOutput_;
  std::vector<std::string> shots_;
  std::unique_ptr<dd::Package> dd_;
  dd::VectorDD stateVecDD_{};
  bool qasmProgram() {
    auto qcoProgram = parseQASMToQCO(program_);
    if (!qcoProgram) {
      return false;
    }
    /// NOLINTNEXTLINE(misc-const-correctness): MLIR handles remain mutable.
    auto entryPoint = mlir::mqt::getEntryPoint(qcoProgram->module());
    if (entryPoint == nullptr) {
      std::ignore = mqt::emitError("QCO program has no entry point");
      return false;
    }
    if (numShots_ != 0) {
      mlir::qco::DDSamplingState retainedState;
      if (mlir::failed(mlir::qco::sample(
              entryPoint, numShots_, seed_.value_or(0),
              mlir::qco::DDArgumentBindings{}, &shots_, &retainedState))) {
        return false;
      }
      dd_ = std::move(retainedState.dd);
      stateVecDD_ = retainedState.state;
      return true;
    }
    auto package = dd::Package::create();
    if (mlir::failed(package)) {
      return false;
    }
    dd_ = std::move(*package);
    auto state = mlir::qco::simulateStatevector(entryPoint, *dd_);
    if (mlir::failed(state)) {
      return false;
    }
    stateVecDD_ = *state;
    return true;
  }
  bool qirProgram() {
    const llvm::StringRef irBytes(program_.data(), program_.size());
    const bool sampling = numShots_ != 0;
    std::optional<std::ostringstream> output;
    auto jitSession = qir::JitSession::create(
        irBytes, "QDMI job",
        sampling ? qir::Execution::Sampling : qir::Execution::StateExtraction,
        sampling ? seed_ : std::nullopt);
    if (mlir::failed(jitSession)) {
      return false;
    }
    auto& runtime = (*jitSession)->runtime();
    if (sampling && captureQIROutput_) {
      runtime.setOstream(output.emplace());
    } else {
      runtime.disableOutput();
    }
    bool stateAvailable = !sampling;
    auto rc = sampling
                  ? (*jitSession)->sample(numShots_, shots_, &stateAvailable)
                  : mlir::FailureOr<int64_t>((*jitSession)->run());
    if (mlir::failed(rc)) {
      return false;
    }
    if (*rc != 0) {
      std::ignore = mqt::emitError("QIR program returned exit code " +
                                   std::to_string(*rc));
      return false;
    }
    if (output) {
      qirOutput_ = std::move(*output).str();
    }
    for (auto& shot : shots_) {
      /// QDMI spells the highest-index output bit first.
      std::ranges::reverse(shot);
    }
    if (stateAvailable) {
      auto state = runtime.takeState();
      dd_ = std::move(state.dd);
      stateVecDD_ = state.edge;
    }
    return true;
  }
};

qdmi::dd::WorkerResponse execute(const qdmi::dd::WorkerRequest& request,
                                 llvm::raw_socket_stream& stream) {
  qdmi::dd::WorkerResponse response;
  mqt::ScopedDiagnosticHandler const capture(
      [&](const mqt::Diagnostic& diagnostic) {
        qdmi::dd::WorkerResponse notification;
        notification.completed = false;
        notification.diagnostics.push_back(diagnostic);
        if (!qdmi::dd::writeFrame(stream, qdmi::dd::encode(notification))) {
          std::exit(1);
        }
        return mlir::success();
      });
  Execution execution{
      .program_ = request.program,
      .numShots_ = static_cast<size_t>(request.shots),
      .seed_ = request.seed,
      .captureQIROutput_ = request.captureOutput,
  };
  const bool qasm = request.format == QDMI_PROGRAM_FORMAT_QASM2 ||
                    request.format == QDMI_PROGRAM_FORMAT_QASM3;
  response.succeeded = qasm ? execution.qasmProgram() : execution.qirProgram();
  if (response.succeeded) {
    response.shots = std::move(execution.shots_);
    response.output = std::move(execution.qirOutput_);
    if (execution.dd_) {
      const auto root = execution.stateVecDD_;
      response.qubits =
          root.isTerminal() ? 0 : static_cast<uint32_t>(root.p->v) + 1;
      std::ostringstream bytes(std::ios::binary);
      dd::serialize(root, bytes, true);
      response.state = std::move(bytes).str();
    }
  }
  return response;
}
} // namespace

int main(int argc, char** argv) {
  const llvm::InitLLVM init(argc, argv);
  llvm::sys::DisableSystemDialogsOnCrash();
  if (argc != 2) {
    return 1;
  }
  auto connection =
      llvm::raw_socket_stream::createConnectedUnix(std::span(argv, 2)[1]);
  if (!connection) {
    llvm::consumeError(connection.takeError());
    return 1;
  }
  const qdmi::dd::WorkerStream ownedStream(connection->release());
  auto& stream = *ownedStream;
  stream.SetUnbuffered();
  std::string bytes;
  while (qdmi::dd::readFrame(stream, bytes)) {
    if (bytes.empty()) {
      break;
    }
    qdmi::dd::WorkerRequest request;
    if (!qdmi::dd::decode(bytes, request)) {
      return 1;
    }
    /// execute destroys the JIT, runtime, DDs, and diagnostic scope before
    /// reuse.
    auto const response = execute(request, stream);
    if (!qdmi::dd::writeFrame(stream, qdmi::dd::encode(response))) {
      return 1;
    }
  }
  return 0;
}

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

#include "WorkerProtocol.hpp"

#include "qdmi/constants.h"

#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_socket_stream.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {
[[nodiscard]] auto parseQASMToQCO(const std::string_view source)
    -> std::optional<mlir::QCOProgram> {
  auto context = mlir::createCompilerContext();
  context->disableMultithreading();
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
      std::cerr << "QCO program has no entry point" << '\n';
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
    dd_ = std::make_unique<dd::Package>();
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
    auto jitSession = qir::JitSession(
        irBytes, "QDMI job",
        sampling ? qir::Execution::Sampling : qir::Execution::StateExtraction,
        sampling ? seed_ : std::nullopt);
    auto& runtime = jitSession.runtime();
    if (sampling && captureQIROutput_) {
      runtime.setOstream(output.emplace());
    } else {
      runtime.disableOutput();
    }
    bool stateAvailable = !sampling;
    const auto rc = sampling
                        ? jitSession.sample(numShots_, shots_, &stateAvailable)
                        : jitSession.run();
    if (rc != 0) {
      std::cerr << "QIR program returned exit code " + std::to_string(rc)
                << '\n';
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

qdmi::dd::WorkerResponse execute(const qdmi::dd::WorkerRequest& request) {
  qdmi::dd::WorkerResponse response;
  Execution execution{
      .program_ = request.program,
      .numShots_ = static_cast<size_t>(request.shots),
      .seed_ = request.seed,
      .captureQIROutput_ = request.captureOutput,
  };
  const bool qasm = request.format == QDMI_PROGRAM_FORMAT_QASM2 ||
                    request.format == QDMI_PROGRAM_FORMAT_QASM3;
  try {
    response.succeeded =
        qasm ? execution.qasmProgram() : execution.qirProgram();
  } catch (const std::exception& error) {
    std::cerr << "DDSIM program failed: " << error.what() << '\n';
  }
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
    /// execute destroys the program's JIT, runtime, and DDs before reuse.
    auto const response = execute(request);
    if (!qdmi::dd::writeFrame(stream, qdmi::dd::encode(response))) {
      return 1;
    }
  }
  return 0;
}

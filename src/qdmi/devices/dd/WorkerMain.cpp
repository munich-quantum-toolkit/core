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
#include <future>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
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
  size_t workerSlots_;
  bool automaticWorkers_;
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
      if (mlir::failed(
              mlir::qco::sample(entryPoint, numShots_, seed_.value_or(0),
                                mlir::qco::DDArgumentBindings{}, &shots_,
                                &retainedState, {}, workerSlots_))) {
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
    const bool large = jitSession.canShareCompiledCode() &&
                       jitSession.quantumCallSites() >= 32;
    const size_t automatic =
        std::clamp(numShots_ / (large ? 32 : 256), size_t{1},
                   large ? size_t{32} : size_t{8});
    size_t workers = 1;
    if (sampling && !jitSession.canSampleTerminal()) {
      workers =
          automaticWorkers_ ? std::min(workerSlots_, automatic) : workerSlots_;
    }
    if (workers == 1) {
      const auto rc =
          sampling ? jitSession.sample(numShots_, shots_, &stateAvailable)
                   : jitSession.run();
      if (rc != 0) {
        std::cerr << "QIR program returned exit code " << rc << '\n';
        return false;
      }
    } else {
      std::mt19937_64 rng(seed_.value_or(std::random_device{}()));
      std::vector<uint64_t> workerSeeds(workers);
      for (size_t i = 1; i < workers; ++i) {
        workerSeeds[i] = rng();
      }
      std::vector<std::vector<std::string>> parts(workers);
      std::vector<std::string> records(workers);
      std::vector<std::future<int64_t>> tasks;
      tasks.reserve(workers);
      const bool shareCode = jitSession.canShareCompiledCode();
      for (size_t i = 0; i < workers; ++i) {
        tasks.push_back(std::async(std::launch::async, [&, i] {
          std::unique_ptr<qir::JitSession> peer;
          std::unique_ptr<qir::Runtime> workerRuntime;
          if (i != 0 && shareCode) {
            workerRuntime = jitSession.makeWorkerRuntime(workerSeeds[i]);
          } else if (i != 0) {
            peer = std::make_unique<qir::JitSession>(
                irBytes, "QDMI job", qir::Execution::Sampling, workerSeeds[i]);
          }
          qir::Runtime* worker = &runtime;
          if (workerRuntime) {
            worker = workerRuntime.get();
          } else if (peer) {
            worker = &peer->runtime();
          }
          std::ostringstream localOutput;
          if (i != 0) {
            if (captureQIROutput_) {
              worker->setOstream(localOutput);
            } else {
              worker->disableOutput();
            }
          }
          const size_t count = (numShots_ / workers) +
                               static_cast<size_t>(i < numShots_ % workers);
          const auto code = workerRuntime
                                ? jitSession.sampleWithRuntime(*worker, count,
                                                               parts[i], false)
                                : (i == 0 ? jitSession : *peer)
                                      .sample(count, parts[i], nullptr, i == 0);
          if (i != 0 && captureQIROutput_) {
            records[i] = std::move(localOutput).str();
          }
          return code;
        }));
      }
      int64_t firstError = 0;
      for (size_t i = 0; i < workers; ++i) {
        const auto code = tasks[i].get();
        if (firstError == 0 && code != 0) {
          firstError = code;
        }
        shots_.insert(shots_.end(), parts[i].begin(), parts[i].end());
      }
      if (firstError != 0) {
        std::cerr << "QIR program returned exit code " << firstError << '\n';
        return false;
      }
      if (output) {
        for (size_t i = 1; i < workers; ++i) {
          *output << records[i];
        }
      }
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
      .workerSlots_ = static_cast<size_t>(request.workerSlots),
      .automaticWorkers_ = request.automaticWorkers,
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

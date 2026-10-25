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
#include "mqt/Dialect/QCO/Utils/DDFunctionality.h"
#include "mqt/Dialect/QIR/Execution/JIT/Session.h"
#include "mqt/Dialect/QIR/Execution/Runtime/Runtime.h"
#include "mqt/Target/OpenQASM/TranslateOpenQASMToQC.h"

#include "WorkerProtocol.hpp"

#include "qdmi/constants.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/raw_socket_stream.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {
void appendShot(std::string& packed, std::string_view shot) {
  const auto offset = packed.size();
  packed.append(shot);
  // QDMI spells the highest-index output bit first.
  std::ranges::reverse(packed.begin() + static_cast<ptrdiff_t>(offset),
                       packed.end());
  packed.push_back(',');
}

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

struct Execution {
  llvm::raw_socket_stream* stream_;
  std::string_view program_;
  size_t numShots_;
  std::optional<uint64_t> seed_;
  bool captureQIROutput_;
  size_t maxWorkers_;
  bool automaticWorkers_;
  std::optional<std::string> qirOutput_;
  std::optional<qdmi::dd::OutputTemplate> outputTemplate_;
  std::string shots_;
  size_t shotCount_ = 0;
  std::map<std::string, size_t> counts_;
  std::unique_ptr<dd::Package> dd_;
  dd::VectorDD stateVecDD_{};
  [[nodiscard]] size_t requestWorkers(size_t desired) const {
    std::string bytes;
    qdmi::dd::WorkerSlots slots;
    if (!qdmi::dd::writeFrame(
            *stream_, qdmi::dd::encode(qdmi::dd::WorkerSlots{desired})) ||
        !qdmi::dd::readFrame(*stream_, bytes) ||
        !qdmi::dd::decode(bytes, slots) || slots.count > desired) {
      throw std::runtime_error("DDSIM worker slot exchange failed");
    }
    return slots.count;
  }
  bool qasmProgram() {
    auto qcoProgram = parseQASMToQCO(program_);
    if (llvm::failed(qcoProgram)) {
      return false;
    }
    // NOLINTNEXTLINE(misc-const-correctness): MLIR handles remain mutable.
    auto entryPoint = mlir::mqt::getEntryPoint(qcoProgram->module());
    if (entryPoint == nullptr) {
      std::cerr << "QCO program has no entry point" << '\n';
      return false;
    }
    if (numShots_ != 0) {
      mlir::qco::DDSamplingState retainedState;
      const size_t workers =
          automaticWorkers_
              ? std::min(maxWorkers_,
                         std::clamp(numShots_ / 256, size_t{1}, size_t{8}))
              : maxWorkers_;
      auto counts = mlir::qco::sample(
          entryPoint, numShots_, seed_.value_or(0),
          mlir::qco::DDArgumentBindings{}, nullptr, &retainedState, {}, workers,
          [&](size_t desired) { return requestWorkers(desired); });
      if (llvm::failed(counts)) {
        return false;
      }
      counts_ = std::move(*counts);
      dd_ = std::move(retainedState.dd);
      stateVecDD_ = retainedState.state;
      return true;
    }
    dd_ = std::make_unique<dd::Package>();
    auto state = mlir::qco::simulateStatevector(entryPoint, *dd_);
    if (llvm::failed(state)) {
      return false;
    }
    stateVecDD_ = *state;
    return true;
  }
  bool qirProgram() {
    const llvm::StringRef irBytes(program_.data(), program_.size());
    const bool sampling = numShots_ != 0;
    const bool capture = sampling && captureQIROutput_;
    std::optional<llvm::raw_string_ostream> output;
    auto jitSession = qir::JitSession(
        irBytes, "QDMI job",
        sampling ? qir::Execution::Sampling : qir::Execution::StateExtraction,
        sampling ? seed_ : std::nullopt,
        capture ? &output.emplace(qirOutput_.emplace()) : nullptr);
    auto& runtime = jitSession.runtime();
    if (capture) {
      // Constructors run before sampling; the schema header still comes first.
      const auto constructorRecords = std::exchange(*qirOutput_, {});
      runtime.outputProgramHeader();
      *output << constructorRecords;
    }
    bool stateAvailable = !sampling;
    const bool large = jitSession.canShareCompiledCode() &&
                       jitSession.quantumCallSites() >= 32;
    const size_t automatic =
        std::clamp(numShots_ / (large ? 32 : 256), size_t{1},
                   large ? size_t{32} : size_t{8});
    size_t workers = 1;
    if (sampling && !jitSession.canSampleTerminal() &&
        (capture || !jitSession.canSampleBranches())) {
      workers =
          automaticWorkers_ ? std::min(maxWorkers_, automatic) : maxWorkers_;
      if (workers > 1) {
        workers = requestWorkers(workers);
      }
    }
    if (workers == 1) {
      std::optional<qir::SampledOutput> terminalOutput;
      std::map<std::string, size_t> outcomes;
      const auto emit = [&](std::string_view shot) {
        if (shots_.empty() && jitSession.canSampleTerminal() &&
            numShots_ <= shots_.max_size() / (shot.size() + 1)) {
          shots_.reserve(numShots_ * (shot.size() + 1));
        }
        appendShot(shots_, shot);
      };
      int64_t rc = 0;
      if (!sampling) {
        rc = jitSession.run();
      } else if (capture) {
        rc = jitSession.sample(numShots_, emit, &stateAvailable, false,
                               &terminalOutput);
      } else {
        rc = jitSession.sampleCounts(numShots_, outcomes, &stateAvailable);
      }
      if (rc != 0) {
        std::cerr << "QIR program returned exit code " << rc << '\n';
        return false;
      }
      if (terminalOutput) {
        outputTemplate_.emplace(std::move(terminalOutput->text),
                                std::move(terminalOutput->resultOffsets));
      }
      for (const auto& [shot, count] : outcomes) {
        counts_.emplace(std::string(shot.rbegin(), shot.rend()), count);
      }
    } else {
      std::mt19937_64 rng(seed_ ? *seed_ : qir::Runtime::generateRandomSeed());
      std::vector<uint64_t> workerSeeds(workers);
      for (size_t i = 1; i < workers; ++i) {
        workerSeeds[i] = rng();
      }
      std::vector<std::string> parts(workers);
      std::vector<std::string> records(workers);
      std::vector<std::map<std::string, size_t>> counts(workers);
      std::vector<std::future<int64_t>> tasks;
      tasks.reserve(workers);
      const bool shareCode = jitSession.canShareCompiledCode();
      for (size_t i = 0; i < workers; ++i) {
        tasks.push_back(std::async(std::launch::async, [&, i] {
          llvm::raw_string_ostream localOutput(records[i]);
          std::unique_ptr<qir::JitSession> peer;
          std::unique_ptr<qir::Runtime> workerRuntime;
          if (i != 0 && shareCode) {
            workerRuntime = jitSession.makeWorkerRuntime(workerSeeds[i]);
          } else if (i != 0) {
            peer = std::make_unique<qir::JitSession>(
                irBytes, "QDMI job", qir::Execution::Sampling, workerSeeds[i],
                capture ? &localOutput : nullptr);
          }
          qir::Runtime* worker = &runtime;
          if (workerRuntime) {
            worker = workerRuntime.get();
          } else if (peer) {
            worker = &peer->runtime();
          }
          if (i != 0) {
            if (capture) {
              worker->setOstream(localOutput);
            } else {
              worker->disableOutput();
            }
          }
          const size_t count = (numShots_ / workers) +
                               static_cast<size_t>(i < numShots_ % workers);
          const auto emit = [&](std::string_view shot) {
            if (capture) {
              appendShot(parts[i], shot);
            } else {
              ++counts[i][std::string(shot.rbegin(), shot.rend())];
            }
          };
          return workerRuntime
                     ? jitSession.sampleWithRuntime(*worker, count, emit, false)
                     : (i == 0 ? jitSession : *peer)
                           .sample(count, emit, nullptr, false);
        }));
      }
      int64_t firstError = 0;
      for (auto& task : tasks) {
        const auto code = task.get();
        if (firstError == 0 && code != 0) {
          firstError = code;
        }
      }
      if (firstError != 0) {
        std::cerr << "QIR program returned exit code " << firstError << '\n';
        return false;
      }
      size_t packedSize = 0;
      for (const auto& part : parts) {
        packedSize += part.size();
      }
      shots_.reserve(packedSize);
      for (const auto& part : parts) {
        shots_ += part;
      }
      if (output) {
        for (size_t i = 1; i < workers; ++i) {
          *output << records[i];
        }
      }
      for (const auto& part : counts) {
        for (const auto& [shot, count] : part) {
          counts_[shot] += count;
        }
      }
    }
    shotCount_ = capture ? numShots_ : 0;
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
  Execution execution{
      .stream_ = &stream,
      .program_ = request.program,
      .numShots_ = static_cast<size_t>(request.shots),
      .seed_ = request.seed,
      .captureQIROutput_ = request.captureOutput,
      .maxWorkers_ = static_cast<size_t>(request.maxWorkers),
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
    response.shotCount = execution.shotCount_;
    response.shots = std::move(execution.shots_);
    response.counts = std::move(execution.counts_);
    response.output = std::move(execution.qirOutput_);
    response.outputTemplate = std::move(execution.outputTemplate_);
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
  while (true) {
    std::string bytes;
    if (!qdmi::dd::readFrame(stream, bytes) || bytes.empty()) {
      break;
    }
    qdmi::dd::WorkerRequest request;
    if (!qdmi::dd::decode(bytes, request)) {
      return 1;
    }
    // execute destroys the program's JIT, runtime, and DDs before reuse.
    auto const response = execute(request, stream);
    if (!qdmi::dd::writeFrame(stream, qdmi::dd::encode(response))) {
      return 1;
    }
  }
  return 0;
}

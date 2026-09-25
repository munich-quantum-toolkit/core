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

#include "support/Diagnostics.hpp"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_socket_stream.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace qdmi::dd {
struct CloseWorkerStream {
  void operator()(llvm::raw_socket_stream* stream) const;
};
using WorkerStream =
    std::unique_ptr<llvm::raw_socket_stream, CloseWorkerStream>;

struct WorkerRequest {
  int32_t format = 0;
  std::string program;
  uint64_t shots = 0;
  std::optional<uint64_t> seed;
  bool captureOutput = false;
};
struct WorkerResponse {
  bool completed = true;
  bool succeeded = false;
  std::vector<mqt::Diagnostic> diagnostics;
  std::vector<std::string> shots;
  std::optional<std::string> output;
  uint32_t qubits = 0;
  std::optional<std::string> state;
};

/// Frames are versioned and length-delimited. A failed read invalidates the
/// worker. An empty request frame closes the worker session.
bool writeFrame(llvm::raw_socket_stream& stream, llvm::StringRef bytes);
bool readFrame(llvm::raw_socket_stream& stream, std::string& bytes);
std::string encode(const WorkerRequest& request);
std::string encode(const WorkerResponse& response);
bool decode(llvm::StringRef bytes, WorkerRequest& request);
bool decode(llvm::StringRef bytes, WorkerResponse& response);
} // namespace qdmi::dd

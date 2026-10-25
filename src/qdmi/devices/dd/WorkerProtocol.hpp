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

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_socket_stream.h"

#include <cstddef>
#include <cstdint>
#include <map>
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
  /// Borrows the job's payload or the decoded frame until execution returns.
  llvm::StringRef program;
  uint64_t shots = 0;
  std::optional<uint64_t> seed;
  bool captureOutput = false;
  uint64_t maxWorkers = 1;
  bool automaticWorkers = true;
};
/// One optional request/grant exchange before parallel sampling.
///
/// The worker asks for at most WorkerRequest::maxWorkers. The supervisor grants
/// between one and that number, keeping one slot until the response arrives.
struct WorkerSlots {
  uint64_t count = 1;
};
/// Terminal shot records appended after the header and constructor output.
struct OutputTemplate {
  std::string text;
  std::vector<uint64_t> resultOffsets;
};
struct WorkerResponse {
  bool succeeded = false;
  uint64_t shotCount = 0;
  /// One comma-terminated bit string per ordered QIR shot, including empty
  /// shots.
  std::string shots;
  std::map<std::string, size_t> counts;
  std::optional<std::string> output;
  std::optional<OutputTemplate> outputTemplate;
  uint32_t qubits = 0;
  std::optional<std::string> state;
};

/// Frames are versioned and length-delimited.
///
/// A failed read invalidates the worker. An empty request frame closes the
/// worker session.
bool writeFrame(llvm::raw_socket_stream& stream, llvm::StringRef bytes);
bool readFrame(llvm::raw_socket_stream& stream, std::string& bytes);
std::string encode(const WorkerRequest& request);
std::string encode(const WorkerResponse& response);
std::string encode(WorkerSlots slots);
bool decode(llvm::StringRef bytes, WorkerRequest& request);
bool decode(llvm::StringRef bytes, WorkerResponse& response);
bool decode(llvm::StringRef bytes, WorkerSlots& slots);
} // namespace qdmi::dd

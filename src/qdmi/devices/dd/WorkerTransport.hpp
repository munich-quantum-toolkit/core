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

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#ifndef _WIN32
#include "llvm/Support/raw_socket_stream.h"
#endif

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

namespace qdmi::dd {
#ifdef _WIN32
struct CloseWorkerPipe {
  void operator()(void* pipe) const;
};
using WorkerStream = std::unique_ptr<void, CloseWorkerPipe>;
#else
struct CloseWorkerStream {
  void operator()(llvm::raw_socket_stream* stream) const;
};
using WorkerStream =
    std::unique_ptr<llvm::raw_socket_stream, CloseWorkerStream>;
#endif

size_t readWorker(const WorkerStream& stream, char* data, size_t size);
bool writeWorker(const WorkerStream& stream, llvm::StringRef bytes);

/// A local endpoint for one DDSIM worker. The returned stream owns the
/// connection.
class WorkerListener {
public:
  ~WorkerListener();
  llvm::Error open();
  [[nodiscard]] llvm::StringRef endpoint() const { return endpoint_; }

  /// An empty stream means there was no connection during this poll.
  llvm::Expected<WorkerStream> accept(std::chrono::milliseconds timeout,
                                      uint32_t workerPid);

private:
  llvm::SmallString<128> endpoint_;
#ifdef _WIN32
  WorkerStream pipe_;
#else
  llvm::SmallString<128> directory_;
  std::optional<llvm::ListeningSocket> listener_;
#endif
};

llvm::Expected<WorkerStream> connectWorker(llvm::StringRef endpoint);
} // namespace qdmi::dd

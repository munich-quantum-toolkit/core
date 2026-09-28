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

#include "WorkerProtocol.hpp"

#include "llvm/Support/Program.h"

#include <memory>
#include <mutex>
#include <vector>

namespace qdmi::dd {
/// The supervisor owns the stream; cancellation only touches the process
/// handle.
class Worker {
public:
  ~Worker();
  bool start();
  bool execute(const WorkerRequest& request, WorkerResponse& response);
  /// Close a completed session and allow bounded process cleanup.
  /// No execute call may be active.
  void shutdown();
  void terminate();

private:
  std::mutex mutex_;
  llvm::sys::ProcessInfo process_;
  bool terminated_ = false;
  WorkerStream stream_;
};

/// Owns active workers as well as the bounded idle cache until shutdown.
class WorkerPool {
public:
  ~WorkerPool();
  std::shared_ptr<Worker> acquire();
  void release(const std::shared_ptr<Worker>& worker, bool reusable);
  void shutdown();
  void initialize();

private:
  std::mutex mutex_;
  bool accepting_ = true;
  std::vector<std::shared_ptr<Worker>> active_;
  std::vector<std::shared_ptr<Worker>> idle_;
};
} // namespace qdmi::dd

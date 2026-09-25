/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "Worker.hpp"

#include "qdmi/common/DeviceConfiguration.hpp"

#include "WorkerProtocol.hpp"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_socket_stream.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#else
#include <signal.h> // NOLINT(modernize-deprecated-headers): POSIX kill and SIGKILL.
#include <sys/wait.h>
#endif

namespace qdmi::dd {
namespace {
const char MODULE_ANCHOR = 0;

bool hasExited(const llvm::sys::ProcessInfo& process) {
#ifdef _WIN32
  return llvm::sys::Wait(process, 0, nullptr, nullptr, true).Pid != 0;
#else
  /// LLVM 23's timed Wait changes the host's SIGALRM handler, even for polling.
  int status = 0;
  /// sys/wait.h owns WNOHANG; the suggested glibc bits header is private.
  /// NOLINTNEXTLINE(misc-include-cleaner)
  const auto exited = waitpid(process.Pid, &status, WNOHANG);
  return exited == process.Pid || (exited == -1 && errno == ECHILD);
#endif
}
} // namespace
Worker::~Worker() { terminate(); }
void Worker::shutdown() {
  {
    const std::scoped_lock lock(mutex_);
    terminated_ = true;
    if (process_.Pid == 0) {
      return;
    }
    if (stream_) {
      std::ignore = writeFrame(*stream_, {});
      stream_.reset();
    }
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(1);
    do {
      if (hasExited(process_)) {
        process_ = {};
        return;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    } while (std::chrono::steady_clock::now() < deadline);
  }
  terminate();
}
void Worker::terminate() {
  const std::scoped_lock lock(mutex_);
  terminated_ = true;
  if (process_.Pid == 0) {
    return;
  }
#ifdef _WIN32
  TerminateProcess(process_.Process, 1);
#else
  kill(process_.Pid, SIGKILL);
#endif
  llvm::sys::Wait(process_, std::nullopt);
  process_ = {};
}
bool Worker::start() {
  llvm::SmallString<128> directory;
  if (const auto error =
          llvm::sys::fs::createUniqueDirectory("mqt-dd", directory)) {
    std::cerr << "Cannot create DDSIM worker socket: " + error.message()
              << '\n';
    return false;
  }
  const llvm::scope_exit cleanup(
      [&] { std::ignore = llvm::sys::fs::remove_directories(directory); });
  auto socketPath = directory;
  llvm::sys::path::append(socketPath, "socket");
  auto listener = llvm::ListeningSocket::createUnix(socketPath);
  if (!listener) {
    std::cerr << llvm::toString(listener.takeError()) << '\n';
    return false;
  }
  const auto executable =
      (qdmi::detail::moduleDirectory(&MODULE_ANCHOR) / MQT_DDSIM_WORKER_NAME)
          .string();
  {
    const std::scoped_lock lock(mutex_);
    if (terminated_) {
      return false;
    }
    std::string error;
    process_ = llvm::sys::ExecuteNoWait(executable, {executable, socketPath},
                                        std::nullopt, {}, 0, &error);
    if (process_.Pid == 0) {
      std::cerr << "Cannot start DDSIM worker: " + error << '\n';
      return false;
    }
  }
  /// Bound startup, while making cancellation responsive before connection.
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (std::chrono::steady_clock::now() < deadline) {
    auto connection = listener->accept(std::chrono::milliseconds(50));
    if (connection) {
      stream_.reset(connection->release());
      stream_->SetUnbuffered();
      return true;
    }
    llvm::consumeError(connection.takeError());
    const std::scoped_lock lock(mutex_);
    if (terminated_) {
      return false;
    }
    if (hasExited(process_)) {
      process_ = {};
      return false;
    }
  }
  std::cerr << "DDSIM worker did not connect before the startup deadline."
            << '\n';
  return false;
}
bool Worker::execute(const WorkerRequest& request, WorkerResponse& response) {
  if (!stream_ && !start()) {
    return false;
  }
  std::string bytes;
  if (writeFrame(*stream_, encode(request)) && readFrame(*stream_, bytes) &&
      decode(bytes, response)) {
    return true;
  }

  std::cerr << "DDSIM worker exited or returned an incomplete response."
            << '\n';
  return false;
}
WorkerPool::~WorkerPool() { shutdown(); }
std::shared_ptr<Worker> WorkerPool::acquire() {
  std::shared_ptr<Worker> worker;
  const std::scoped_lock lock(mutex_);
  if (!accepting_) {
    return {};
  }
  worker = idle_.empty() ? std::make_shared<Worker>() : std::move(idle_.back());
  if (!idle_.empty()) {
    idle_.pop_back();
  }
  active_.push_back(worker);
  return worker;
}
void WorkerPool::release(const std::shared_ptr<Worker>& worker, bool reusable) {
  {
    const std::scoped_lock lock(mutex_);
    const auto it = std::ranges::find(active_, worker);
    if (it == active_.end()) {
      return;
    }
    active_.erase(it);
    if (reusable && accepting_) {
      idle_.push_back(worker);
      return;
    }
  }
  worker->shutdown();
}
void WorkerPool::initialize() {
  const std::scoped_lock lock(mutex_);
  accepting_ = true;
}
void WorkerPool::shutdown() {
  std::vector<std::shared_ptr<Worker>> active;
  std::vector<std::shared_ptr<Worker>> idle;
  {
    const std::scoped_lock lock(mutex_);
    accepting_ = false;
    active.swap(active_);
    idle.swap(idle_);
  }
  for (const auto& worker : active) {
    worker->terminate();
  }
  for (const auto& worker : idle) {
    worker->shutdown();
  }
}
} // namespace qdmi::dd

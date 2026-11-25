/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "WorkerTransport.hpp"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <system_error>
#include <utility>

#ifdef _WIN32
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/RandomNumberGenerator.h"

#include <algorithm>
#include <array>
#include <thread>
#include <windows.h>
#else
#include "llvm/Support/Errno.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_socket_stream.h"

#include <sys/socket.h>
#include <tuple>
#endif

namespace qdmi::dd {
namespace {
#ifdef _WIN32
constexpr llvm::StringLiteral PIPE_PREFIX = R"(\\.\pipe\mqt-dd-)";
constexpr size_t PIPE_TOKEN_BYTES = 16;
// Buffer short frames so writes can finish before the peer reads.
constexpr DWORD PIPE_BUFFER_BYTES = 4096;

llvm::Error pipeError(const char* message, DWORD error = GetLastError()) {
  return llvm::createStringError(
      std::error_code(static_cast<int>(error), std::system_category()),
      message);
}

DWORD ioSize(size_t size) {
  return static_cast<DWORD>(std::min<size_t>(size, MAXDWORD));
}
#else
WorkerStream unbuffered(llvm::raw_socket_stream* stream) {
  stream->SetUnbuffered();
  return WorkerStream(stream);
}
#endif
} // namespace

#ifdef _WIN32
void CloseWorkerPipe::operator()(void* pipe) const { CloseHandle(pipe); }

size_t readWorker(const WorkerStream& stream, char* data, size_t size) {
  DWORD received = 0;
  if (!ReadFile(stream.get(), data, ioSize(size), &received, nullptr)) {
    return 0;
  }
  return received;
}

bool writeWorker(const WorkerStream& stream, llvm::StringRef bytes) {
  while (!bytes.empty()) {
    DWORD sent = 0;
    if (!WriteFile(stream.get(), bytes.data(), ioSize(bytes.size()), &sent,
                   nullptr) ||
        sent == 0) {
      return false;
    }
    bytes = bytes.drop_front(sent);
  }
  return true;
}

WorkerListener::~WorkerListener() = default;

llvm::Error WorkerListener::open() {
  std::array<uint8_t, PIPE_TOKEN_BYTES> random{};
  if (const auto error = llvm::getRandomBytes(random.data(), random.size())) {
    return llvm::createStringError(error, "Cannot name DDSIM worker pipe");
  }
  endpoint_ = PIPE_PREFIX;
  endpoint_ += llvm::toHex(llvm::ArrayRef(random));
  // Poll during startup so the supervisor can observe child exit or
  // cancellation.
  auto* pipe = CreateNamedPipeA(
      endpoint_.c_str(), PIPE_ACCESS_DUPLEX | FILE_FLAG_FIRST_PIPE_INSTANCE,
      PIPE_TYPE_BYTE | PIPE_NOWAIT | PIPE_REJECT_REMOTE_CLIENTS, 1,
      PIPE_BUFFER_BYTES, PIPE_BUFFER_BYTES, 0, nullptr);
  if (pipe == INVALID_HANDLE_VALUE) {
    return pipeError("Cannot create DDSIM worker pipe");
  }
  pipe_.reset(pipe);
  return llvm::Error::success();
}

llvm::Expected<WorkerStream>
WorkerListener::accept(std::chrono::milliseconds timeout, uint32_t workerPid) {
  const auto result = ConnectNamedPipe(pipe_.get(), nullptr)
                          ? ERROR_PIPE_LISTENING
                          : GetLastError();
  if (result == ERROR_PIPE_LISTENING) {
    std::this_thread::sleep_for(timeout);
    return WorkerStream{};
  }
  if (result != ERROR_PIPE_CONNECTED) {
    return pipeError("Cannot accept DDSIM worker pipe", result);
  }
  ULONG clientPid = 0;
  if (!GetNamedPipeClientProcessId(pipe_.get(), &clientPid)) {
    return pipeError("Cannot identify DDSIM worker pipe peer");
  }
  if (clientPid != workerPid) {
    return llvm::createStringError("Unexpected DDSIM worker pipe peer");
  }
  DWORD mode = PIPE_WAIT;
  if (!SetNamedPipeHandleState(pipe_.get(), &mode, nullptr, nullptr)) {
    return pipeError("Cannot set DDSIM worker pipe to blocking mode");
  }
  return std::move(pipe_);
}

llvm::Expected<WorkerStream> connectWorker(llvm::StringRef endpoint) {
  const auto localName = endpoint;
  if (!endpoint.consume_front(PIPE_PREFIX) ||
      endpoint.size() != 2 * PIPE_TOKEN_BYTES ||
      !std::ranges::all_of(endpoint, llvm::isHexDigit)) {
    return llvm::createStringError("Invalid DDSIM worker pipe name");
  }
  auto* pipe = CreateFileA(
      localName.str().c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr,
      OPEN_EXISTING, SECURITY_SQOS_PRESENT | SECURITY_IDENTIFICATION, nullptr);
  if (pipe == INVALID_HANDLE_VALUE) {
    return pipeError("Cannot connect to DDSIM worker pipe");
  }
  return WorkerStream(pipe);
}
#else
void CloseWorkerStream::operator()(llvm::raw_socket_stream* stream) const {
  // A transport failure already fails the job. It must not abort the host.
  stream->close();
  stream->clear_error();
  std::default_delete<llvm::raw_socket_stream>{}(stream);
}

size_t readWorker(const WorkerStream& stream, char* data, size_t size) {
  const auto count = stream->read(data, size);
  return count > 0 ? static_cast<size_t>(count) : 0;
}

bool writeWorker(const WorkerStream& stream, llvm::StringRef bytes) {
  // LLVM exposes the descriptor only to subclasses. Use its inherited member
  // pointer without downcasting the stream.
  struct SocketAccess : llvm::raw_socket_stream {
    static int descriptor(const llvm::raw_socket_stream& socket) {
      return (socket.*&SocketAccess::get_fd)();
    }
  };
  const auto socket = SocketAccess::descriptor(*stream);
  while (!bytes.empty()) {
    // Suppress SIGPIPE per send: Darwin delivers it to the process, so masking
    // only this thread cannot protect other host threads.
    const auto count = llvm::sys::RetryAfterSignal(
        -1, ::send, socket, bytes.data(), bytes.size(), MSG_NOSIGNAL);
    if (count <= 0) {
      return false;
    }
    bytes = bytes.drop_front(static_cast<size_t>(count));
  }
  return true;
}

WorkerListener::~WorkerListener() {
  listener_.reset();
  if (!directory_.empty()) {
    std::ignore = llvm::sys::fs::remove(endpoint_);
    std::ignore = llvm::sys::fs::remove(directory_);
  }
}

llvm::Error WorkerListener::open() {
  if (const auto error =
          llvm::sys::fs::createUniqueDirectory("mqt-dd", directory_)) {
    directory_.clear();
    return llvm::createStringError(error,
                                   "Cannot create DDSIM socket directory");
  }
  endpoint_ = directory_;
  llvm::sys::path::append(endpoint_, "socket");
  auto listener = llvm::ListeningSocket::createUnix(endpoint_);
  if (!listener) {
    return listener.takeError();
  }
  listener_.emplace(std::move(*listener));
  return llvm::Error::success();
}

llvm::Expected<WorkerStream>
WorkerListener::accept(std::chrono::milliseconds timeout,
                       uint32_t /*workerPid*/) {
  auto stream = listener_->accept(timeout);
  if (!stream) {
    // Worker startup checks for child exit and retries until its deadline.
    llvm::consumeError(stream.takeError());
    return WorkerStream{};
  }
  return unbuffered(stream->release());
}

llvm::Expected<WorkerStream> connectWorker(llvm::StringRef endpoint) {
  auto stream = llvm::raw_socket_stream::createConnectedUnix(endpoint);
  if (!stream) {
    return stream.takeError();
  }
  return unbuffered(stream->release());
}
#endif
} // namespace qdmi::dd

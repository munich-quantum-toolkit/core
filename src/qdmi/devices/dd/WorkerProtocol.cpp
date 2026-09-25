/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "WorkerProtocol.hpp"

#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/Shared/SimplePackedSerialization.h"
#include "llvm/Support/Errno.h"
#include "llvm/Support/raw_socket_stream.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#ifndef _WIN32
#include <sys/socket.h>
#endif

namespace qdmi::dd {
namespace {
using namespace llvm::orc::shared;
constexpr uint32_t MAGIC = 0x4d515444;
constexpr uint32_t VERSION = 1;
using Header = SPSArgList<uint32_t, uint32_t, uint64_t>;
using Request =
    SPSArgList<int32_t, SPSString, uint64_t, SPSOptional<uint64_t>, uint8_t>;
using Response = SPSArgList<uint8_t, SPSOptional<SPSString>, uint32_t,
                            SPSOptional<SPSString>>;

/// Validate the presence byte before SPS can deserialize it as a C++ bool.
template <class Tag, class T>
bool readOptional(SPSInputBuffer& buffer, std::optional<T>& value) {
  uint8_t present = 0;
  if (!SPSArgList<uint8_t>::deserialize(buffer, present) || present > 1) {
    return false;
  }
  if (present == 0) {
    value.reset();
    return true;
  }
  return SPSArgList<Tag>::deserialize(buffer, value.emplace());
}

bool readAll(llvm::raw_socket_stream& stream, char* data, size_t size) {
  auto remaining = std::span(data, size);
  while (!remaining.empty()) {
    const auto count = stream.read(remaining.data(), remaining.size());
    if (count <= 0) {
      return false;
    }
    remaining = remaining.subspan(static_cast<size_t>(count));
  }
  return true;
}
/// NOLINTNEXTLINE(misc-const-correctness): Windows writes through the stream.
bool writeAll(llvm::raw_socket_stream& stream, llvm::StringRef bytes) {
#ifdef _WIN32
  stream.write(bytes.data(), bytes.size());
  stream.flush();
  const bool ok = !stream.has_error();
  stream.clear_error();
  return ok;
#else
  /// LLVM exposes the descriptor only to subclasses. Use its inherited member
  /// pointer without downcasting the stream.
  struct SocketAccess : llvm::raw_socket_stream {
    static int descriptor(const llvm::raw_socket_stream& socket) {
      return (socket.*&SocketAccess::get_fd)();
    }
  };
  const auto socket = SocketAccess::descriptor(stream);
  while (!bytes.empty()) {
    /// Suppress SIGPIPE per send: Darwin delivers it to the process, so masking
    /// only this thread cannot protect other host threads.
    const auto count = llvm::sys::RetryAfterSignal(
        -1, ::send, socket, bytes.data(), bytes.size(), MSG_NOSIGNAL);
    if (count <= 0) {
      return false;
    }
    bytes = bytes.drop_front(static_cast<size_t>(count));
  }
  return true;
#endif
}
} // namespace

void CloseWorkerStream::operator()(llvm::raw_socket_stream* stream) const {
  /// Close before LLVM 23's Winsock guard is destroyed. Transport failures
  /// already fail the job; a pending stream error must not abort the host.
  stream->close();
  stream->clear_error();
  std::default_delete<llvm::raw_socket_stream>{}(stream);
}

bool writeFrame(llvm::raw_socket_stream& stream, llvm::StringRef bytes) {
  std::array<char, 16> header{};
  SPSOutputBuffer buffer(header.data(), header.size());
  Header::serialize(buffer, MAGIC, VERSION,
                    static_cast<uint64_t>(bytes.size()));
  return writeAll(stream, llvm::StringRef(header.data(), header.size())) &&
         writeAll(stream, bytes);
}

bool readFrame(llvm::raw_socket_stream& stream, std::string& bytes) {
  std::array<char, 16> header{};
  if (!readAll(stream, header.data(), header.size())) {
    return false;
  }
  SPSInputBuffer buffer(header.data(), header.size());
  uint32_t magic = 0;
  uint32_t version = 0;
  uint64_t size = 0;
  if (!Header::deserialize(buffer, magic, version, size) || magic != MAGIC ||
      version != VERSION || size > bytes.max_size()) {
    return false;
  }
  bytes.clear();
  /// Allocate only for bytes actually received, not an untrusted length.
  std::array<char, 65536> chunk{};
  while (size != 0) {
    const auto count =
        stream.read(chunk.data(), std::min<uint64_t>(size, chunk.size()));
    if (count <= 0) {
      return false;
    }
    bytes.append(chunk.data(), static_cast<size_t>(count));
    size -= static_cast<uint64_t>(count);
  }
  return true;
}

std::string encode(const WorkerRequest& request) {
  std::string bytes(Request::size(request.format, request.program,
                                  request.shots, request.seed,
                                  static_cast<uint8_t>(request.captureOutput)),
                    '\0');
  SPSOutputBuffer buffer(bytes.data(), bytes.size());
  Request::serialize(buffer, request.format, request.program, request.shots,
                     request.seed, static_cast<uint8_t>(request.captureOutput));
  return bytes;
}

bool decode(llvm::StringRef bytes, WorkerRequest& request) {
  SPSInputBuffer buffer(bytes.data(), bytes.size());
  llvm::StringRef program;
  uint8_t capture = 0;
  if (!SPSArgList<int32_t, SPSString, uint64_t>::deserialize(
          buffer, request.format, program, request.shots) ||
      !readOptional<uint64_t>(buffer, request.seed) ||
      !SPSArgList<uint8_t>::deserialize(buffer, capture) || capture > 1 ||
      buffer.data() != bytes.end()) {
    return false;
  }
  request.captureOutput = capture != 0;
  request.program = program.str();
  return true;
}

std::string encode(const WorkerResponse& response) {
  auto size = Response::size(static_cast<uint8_t>(response.succeeded),
                             response.output, response.qubits, response.state) +
              sizeof(uint64_t);
  for (const auto& shot : response.shots) {
    size += SPSArgList<SPSString>::size(shot);
  }
  std::string bytes(size, '\0');
  SPSOutputBuffer buffer(bytes.data(), bytes.size());
  Response::serialize(buffer, static_cast<uint8_t>(response.succeeded),
                      response.output, response.qubits, response.state);
  SPSArgList<uint64_t>::serialize(buffer,
                                  static_cast<uint64_t>(response.shots.size()));
  for (const auto& shot : response.shots) {
    SPSArgList<SPSString>::serialize(buffer, shot);
  }
  return bytes;
}

bool decode(llvm::StringRef bytes, WorkerResponse& response) {
  SPSInputBuffer buffer(bytes.data(), bytes.size());
  std::optional<llvm::StringRef> output;
  std::optional<llvm::StringRef> state;
  uint8_t succeeded = 0;
  if (!SPSArgList<uint8_t>::deserialize(buffer, succeeded) ||
      !readOptional<SPSString>(buffer, output) ||
      !SPSArgList<uint32_t>::deserialize(buffer, response.qubits) ||
      !readOptional<SPSString>(buffer, state) || succeeded > 1) {
    return false;
  }
  response.succeeded = succeeded != 0;
  if (output) {
    response.output = output->str();
  }
  if (state) {
    response.state = state->str();
  }
  uint64_t count = 0;
  if (!SPSArgList<uint64_t>::deserialize(buffer, count) ||
      count > bytes.size() / sizeof(uint64_t)) {
    return false;
  }
  for (uint64_t i = 0; i < count; ++i) {
    llvm::StringRef shot;
    if (!SPSArgList<SPSString>::deserialize(buffer, shot) ||
        shot.find_first_not_of("01") != llvm::StringRef::npos) {
      return false;
    }
    response.shots.push_back(shot.str());
  }
  return buffer.data() == bytes.end();
}
} // namespace qdmi::dd

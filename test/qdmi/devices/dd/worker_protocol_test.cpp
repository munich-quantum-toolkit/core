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

#include "gtest/gtest.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_socket_stream.h"

#include <chrono>
#include <cstddef>
#include <string>
#include <tuple>

namespace qdmi::dd {
TEST(WorkerProtocol, SocketRoundTripAndDisconnect) {
  llvm::SmallString<128> directory;
  ASSERT_FALSE(
      llvm::sys::fs::createUniqueDirectory("mqt-transport", directory));
  const llvm::scope_exit cleanup(
      [&] { std::ignore = llvm::sys::fs::remove_directories(directory); });
  auto socketPath = directory;
  llvm::sys::path::append(socketPath, "socket");
  WorkerStream client;
  WorkerStream server;
  {
    auto listener = llvm::ListeningSocket::createUnix(socketPath);
    ASSERT_TRUE(!!listener) << llvm::toString(listener.takeError());
    auto connection = llvm::raw_socket_stream::createConnectedUnix(socketPath);
    ASSERT_TRUE(!!connection) << llvm::toString(connection.takeError());
    client.reset(connection->release());
    auto accepted = listener->accept(std::chrono::seconds(5));
    ASSERT_TRUE(!!accepted) << llvm::toString(accepted.takeError());
    server.reset(accepted->release());
  }

  std::string payload;
  for (int byte = 0; byte < 256; ++byte) {
    payload += static_cast<char>(byte);
  }
  ASSERT_TRUE(writeFrame(*client, payload));
  std::string received;
  ASSERT_TRUE(readFrame(*server, received));
  EXPECT_EQ(received, payload);
  ASSERT_TRUE(writeFrame(*client, {}));
  ASSERT_TRUE(readFrame(*server, received));
  EXPECT_TRUE(received.empty());
  char byte = 0;
  EXPECT_EQ(client->read(&byte, 1, std::chrono::milliseconds(1)), -1);
  EXPECT_TRUE(client->has_error());
  client.reset();
  EXPECT_FALSE(readFrame(*server, received));
  /// Writes may succeed locally after a peer disconnects, notably on Windows.
  /// They must not signal or terminate the host.
  std::ignore = writeFrame(*server, "closed");
  server.reset();
}

TEST(WorkerProtocol, PreservesExactPayloadAndResultMetadata) {
  const WorkerRequest request{
      .format = 7,
      .program = std::string("QIR\0bytes", 9),
      .shots = 19,
      .seed = 42,
      .captureOutput = true,
  };
  WorkerRequest decoded;
  ASSERT_TRUE(decode(encode(request), decoded));
  EXPECT_EQ(decoded.format, request.format);
  EXPECT_EQ(decoded.program, request.program);
  EXPECT_EQ(decoded.shots, request.shots);
  EXPECT_EQ(decoded.seed, request.seed);
  EXPECT_EQ(decoded.captureOutput, request.captureOutput);

  WorkerResponse response;
  response.succeeded = true;
  response.shots = {"10", "01", "10"};
  response.output = "OUTPUT\tBOOL\ttrue\n";
  response.qubits = 2;
  response.state = std::string("DD\0bytes", 8);
  WorkerResponse result;
  ASSERT_TRUE(decode(encode(response), result));
  EXPECT_TRUE(result.succeeded);
  EXPECT_EQ(result.shots, response.shots);
  EXPECT_EQ(result.output, response.output);
  EXPECT_EQ(result.qubits, response.qubits);
  EXPECT_EQ(result.state, response.state);
}

TEST(WorkerProtocol, RejectsTruncationTrailingBytesAndInvalidResults) {
  WorkerResponse response;
  response.succeeded = true;
  response.shots = {"01"};
  const auto bytes = encode(response);
  for (size_t size = 0; size < bytes.size(); ++size) {
    WorkerResponse result;
    EXPECT_FALSE(decode(llvm::StringRef(bytes.data(), size), result));
  }
  WorkerResponse result;
  auto invalidFlag = bytes;
  invalidFlag[0] = 2;
  EXPECT_FALSE(decode(invalidFlag, result));
  EXPECT_FALSE(decode(bytes + "x", result));
  response.shots = {"02"};
  EXPECT_FALSE(decode(encode(response), result));
  const auto request = encode(WorkerRequest{
      .format = 7,
      .program = "payload",
      .shots = 1,
      .seed = {},
      .captureOutput = false,
  });
  for (size_t size = 0; size < request.size(); ++size) {
    WorkerRequest parsed;
    EXPECT_FALSE(decode(llvm::StringRef(request.data(), size), parsed));
  }
  WorkerRequest parsed;
  EXPECT_FALSE(decode(request + "x", parsed));
}
} // namespace qdmi::dd

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
#include <cstdint>
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
  // Writes may succeed locally after a peer disconnects, notably on Windows.
  // They must not signal or terminate the host.
  std::ignore = writeFrame(*server, "closed");
  server.reset();
}

TEST(WorkerProtocol, PreservesExactPayloadAndResultMetadata) {
  const WorkerRequest request{
      .format = 7,
      .program = llvm::StringRef("QIR\0bytes", 9),
      .shots = 19,
      .seed = 42,
      .captureOutput = true,
      .maxWorkers = 3,
      .automaticWorkers = false,
  };
  const auto frame = encode(request);
  WorkerRequest decoded;
  ASSERT_TRUE(decode(frame, decoded));
  EXPECT_EQ(decoded.format, request.format);
  EXPECT_EQ(decoded.program, request.program);
  EXPECT_EQ(decoded.shots, request.shots);
  EXPECT_EQ(decoded.seed, request.seed);
  EXPECT_EQ(decoded.captureOutput, request.captureOutput);
  EXPECT_EQ(decoded.maxWorkers, request.maxWorkers);
  EXPECT_EQ(decoded.automaticWorkers, request.automaticWorkers);

  WorkerResponse response;
  response.succeeded = true;
  response.shotCount = 5;
  response.shots = "10,01,10,,,";
  response.output = "OUTPUT\tBOOL\ttrue\n";
  response.qubits = 2;
  response.state = std::string("DD\0bytes", 8);
  WorkerResponse result;
  ASSERT_TRUE(decode(encode(response), result));
  EXPECT_TRUE(result.succeeded);
  EXPECT_EQ(result.shotCount, response.shotCount);
  EXPECT_EQ(result.shots, response.shots);
  EXPECT_EQ(result.output, response.output);
  EXPECT_FALSE(result.outputTemplate);
  EXPECT_EQ(result.qubits, response.qubits);
  EXPECT_EQ(result.state, response.state);

  response.shotCount = 3;
  response.shots = "10,01,10,";
  response.outputTemplate =
      OutputTemplate{.text = "0\n1\n", .resultOffsets = {0, 2}};
  ASSERT_TRUE(decode(encode(response), result));
  EXPECT_EQ(result.shots, response.shots);
  ASSERT_TRUE(result.outputTemplate);
  EXPECT_EQ(result.outputTemplate->text, response.outputTemplate->text);
  EXPECT_EQ(result.outputTemplate->resultOffsets,
            response.outputTemplate->resultOffsets);

  response.shotCount = 2;
  response.shots = ",,";
  response.outputTemplate = OutputTemplate{.text = "42\n", .resultOffsets = {}};
  ASSERT_TRUE(decode(encode(response), result));
  EXPECT_EQ(result.shots, response.shots);
  ASSERT_TRUE(result.outputTemplate);
  EXPECT_EQ(result.outputTemplate->text, "42\n");
  EXPECT_TRUE(result.outputTemplate->resultOffsets.empty());

  WorkerResponse compressed;
  compressed.succeeded = true;
  compressed.counts = {{"0", 2}, {"1", 1}};
  WorkerResponse decodedCounts;
  ASSERT_TRUE(decode(encode(compressed), decodedCounts));
  EXPECT_TRUE(decodedCounts.shots.empty());
  EXPECT_EQ(decodedCounts.counts, compressed.counts);

  WorkerSlots slots;
  ASSERT_TRUE(decode(encode(WorkerSlots{3}), slots));
  EXPECT_EQ(slots.count, 3U);
  EXPECT_FALSE(decode(encode(WorkerSlots{3}), result));
  EXPECT_FALSE(decode(encode(response), slots));
}

TEST(WorkerProtocol, RejectsTruncationTrailingBytesAndInvalidResults) {
  WorkerResponse response;
  response.succeeded = true;
  response.shotCount = 1;
  response.shots = "01,";
  response.output = "prefix";
  response.outputTemplate =
      OutputTemplate{.text = "0\n1\n", .resultOffsets = {0, 2}};
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
  response.shotCount = 2;
  response.shots = "0,111,";
  EXPECT_FALSE(decode(encode(response), result));
  response.shotCount = 1;
  response.shots = "01,";
  for (const auto& record : {
           OutputTemplate{.text = "0\n1\n", .resultOffsets = {0, 4}},
           OutputTemplate{.text = "0\n1\n", .resultOffsets = {0, 0}},
           OutputTemplate{.text = "0\n1\n", .resultOffsets = {0, 1}},
           OutputTemplate{.text = "0\n1\n", .resultOffsets = {0}},
       }) {
    response.outputTemplate = record;
    EXPECT_FALSE(decode(encode(response), result));
  }
  response.outputTemplate.reset();
  response.shots = "02,";
  EXPECT_FALSE(decode(encode(response), result));
  response.shots.clear();
  response.shotCount = 0;
  response.counts = {{"02", 1}};
  EXPECT_FALSE(decode(encode(response), result));
  response.counts = {{"01", 0}};
  EXPECT_FALSE(decode(encode(response), result));
  response.counts = {{"0", 1}, {"1", 1}};
  auto duplicate = encode(response);
  duplicate[duplicate.size() - sizeof(uint64_t) - 1] = '0';
  WorkerResponse duplicateResult;
  EXPECT_FALSE(decode(duplicate, duplicateResult));
  response.shotCount = 1;
  response.shots = "01,";
  response.counts = {{"01", 1}};
  EXPECT_FALSE(decode(encode(response), result));
  response.counts.clear();
  response.shots = "01";
  EXPECT_FALSE(decode(encode(response), result));
  response.shots = "01,,";
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
  WorkerSlots slots;
  const auto grant = encode(WorkerSlots{3});
  for (size_t size = 0; size < grant.size(); ++size) {
    EXPECT_FALSE(decode(llvm::StringRef(grant.data(), size), slots));
  }
  EXPECT_FALSE(decode(grant + "x", slots));
  EXPECT_FALSE(decode(encode(WorkerSlots{0}), slots));
  auto invalidWorkers = request;
  invalidWorkers.back() = 2;
  EXPECT_FALSE(decode(invalidWorkers, parsed));
}
} // namespace qdmi::dd

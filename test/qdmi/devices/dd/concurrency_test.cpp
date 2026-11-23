/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/*
 * DDSIM QDMI Device - Concurrency tests
 */
#include "mqt_ddsim_qdmi/constants.h"
#include "mqt_ddsim_qdmi/device.h"

#include "helpers/circuits.hpp"
#include "helpers/controlled_job.hpp"
#include "helpers/test_utils.hpp"
#include "support/TestSupport.hpp"

#include "gtest/gtest.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Threading.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iterator>
#include <memory>
#include <numbers>
#include <numeric>
#include <ranges>
#include <string>
#include <system_error>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#ifndef _WIN32
// SIGXFSZ is a POSIX extension, not part of <csignal>.
#include <signal.h> // NOLINT(modernize-deprecated-headers)
#include <sys/resource.h>
#endif

namespace {

unsigned workerLimit() {
  auto limit = llvm::heavyweight_hardware_concurrency().compute_thread_count();
  if (const char* raw = std::getenv("MQT_CORE_DD_WORKER_BUDGET")) {
    unsigned configured = 0;
    const auto* last = std::next(raw, static_cast<ptrdiff_t>(std::strlen(raw)));
    const auto [end, error] = std::from_chars(raw, last, configured);
    if (error == std::errc{} && end == last && configured > 0) {
      limit = std::min(limit, configured);
    }
  }
  return limit;
}

std::vector<std::array<uint64_t, 5>> readProgramTrace(const std::string& path) {
  const auto buffer = llvm::MemoryBuffer::getFile(path);
  EXPECT_TRUE(buffer);
  if (!buffer) {
    return {};
  }
  llvm::SmallVector<llvm::StringRef> lines;
  (*buffer)->getBuffer().split(lines, '\n', -1, false);
  std::vector<std::array<uint64_t, 5>> records;
  for (const auto line : lines) {
    llvm::SmallVector<llvm::StringRef> fields;
    line.split(fields, '\t');
    EXPECT_EQ(fields.size(), 5);
    if (fields.size() != 5) {
      return {};
    }
    std::array<uint64_t, 5> values{};
    for (size_t i = 0; i < fields.size(); ++i) {
      EXPECT_FALSE(fields[i].getAsInteger(10, values[i]));
    }
    records.push_back(values);
  }
  return records;
}

} // namespace

#ifdef __linux__
TEST(Concurrency, FinalizeReapsIdleWorkers) {
  const auto count = std::min(2U, workerLimit());
  const qdmi_test::SessionGuard session{};
  {
    const qdmi_test::JobGuard job{session.session};
    // The barrier keeps programs active together before returning them idle.
    qdmi_test::ControlledJob running{job.job, count};
    running.release();
    ASSERT_EQ(MQT_DDSIM_QDMI_device_job_wait(job.job, 0), QDMI_SUCCESS);
  }
  std::vector<int> children;
  for (const auto& task :
       std::filesystem::directory_iterator("/proc/self/task")) {
    const auto buffer = llvm::MemoryBuffer::getFileAsStream(
        (task.path() / "children").string());
    ASSERT_TRUE(buffer);
    llvm::SmallVector<llvm::StringRef> pids;
    (*buffer)->getBuffer().split(pids, ' ', -1, false);
    for (const auto pid : pids) {
      int child = 0;
      ASSERT_FALSE(pid.getAsInteger(10, child));
      children.push_back(child);
    }
  }
  ASSERT_GE(children.size(), count);
  MQT_DDSIM_QDMI_device_finalize();
  for (const auto child : children) {
    EXPECT_EQ(kill(child, 0), -1);
    EXPECT_EQ(errno, ESRCH);
  }
}
#endif

using ColdVectorReads =
    testing::TestWithParam<std::tuple<QDMI_Job_Result, size_t>>;

TEST_P(ColdVectorReads, ConcurrentProbabilityAndStatevectorReads) {
  const auto [query, shots] = GetParam();
  const qdmi_test::SessionGuard s{};
  const qdmi_test::JobGuard j{s.session};
  ASSERT_EQ(qdmi_test::setProgram(j.job, QDMI_PROGRAM_FORMAT_QASM3,
                                  shots == 0 ? qdmi_test::QASM3_BELL_STATE
                                             : qdmi_test::QASM3_BELL_SAMPLING),
            QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::setShots(j.job, shots), QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::submitAndWait(j.job, 0), QDMI_SUCCESS);

  std::barrier start{4};
  const auto worker = [&](const QDMI_Job_Result result) {
    const bool probabilities =
        result == QDMI_JOB_RESULT_PROBABILITIES_DENSE ||
        result == QDMI_JOB_RESULT_PROBABILITIES_SPARSE_VALUES;
    const bool sparse = result == QDMI_JOB_RESULT_STATEVECTOR_SPARSE_VALUES ||
                        result == QDMI_JOB_RESULT_PROBABILITIES_SPARSE_VALUES;
    const size_t stride = probabilities ? 1 : 2;
    std::vector<double> buffer((sparse ? 2 : 4) * stride, -1.);
    start.arrive_and_wait();
    ASSERT_EQ(MQT_DDSIM_QDMI_device_job_get_results(
                  j.job, 0, result, buffer.size() * sizeof(double),
                  buffer.data(), nullptr),
              QDMI_SUCCESS);
    const auto nonzero = probabilities ? 0.5 : 1.0 / std::numbers::sqrt2;
    for (size_t i = 0; i < buffer.size(); ++i) {
      const auto expected =
          i == 0 || i == (sparse ? 1U : 3U) * stride ? nonzero : 0.0;
      EXPECT_NEAR(buffer[i], expected, 1e-12);
    }
  };

  const std::jthread t1(worker, query);
  const std::jthread t2(worker, query);
  const std::jthread t3(worker, QDMI_JOB_RESULT_PROBABILITIES_DENSE);
  const std::jthread t4(worker, QDMI_JOB_RESULT_PROBABILITIES_DENSE);
}

INSTANTIATE_TEST_SUITE_P(
    Concurrency, ColdVectorReads,
    testing::Combine(
        testing::Values(QDMI_JOB_RESULT_STATEVECTOR_DENSE,
                        QDMI_JOB_RESULT_PROBABILITIES_DENSE,
                        QDMI_JOB_RESULT_STATEVECTOR_SPARSE_VALUES,
                        QDMI_JOB_RESULT_PROBABILITIES_SPARSE_VALUES),
        testing::Values(size_t{0}, size_t{128})));

TEST(Concurrency, ConcurrentHistogramReads) {
  const qdmi_test::SessionGuard s{};
  const qdmi_test::JobGuard j{s.session};
  ASSERT_EQ(qdmi_test::setProgram(j.job, QDMI_PROGRAM_FORMAT_QASM3,
                                  qdmi_test::QASM3_BELL_SAMPLING),
            QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::setShots(j.job, 1024), QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::submitAndWait(j.job, 0), QDMI_SUCCESS);

  std::barrier start{4};
  const auto worker = [&] {
    start.arrive_and_wait();
    const auto [keys, values] = qdmi_test::getHistogram(j.job);
    EXPECT_EQ(keys, (std::vector<std::string>{"00", "11"}));
    EXPECT_EQ(std::accumulate(values.begin(), values.end(), size_t{0}), 1024);
  };
  const std::jthread t1(worker);
  const std::jthread t2(worker);
  const std::jthread t3(worker);
  const std::jthread t4(worker);
}

TEST(Concurrency, ConcurrentCheckDuringRun) {
  const qdmi_test::SessionGuard s{};
  const qdmi_test::JobGuard j{s.session};
  ASSERT_EQ(qdmi_test::setProgram(j.job, QDMI_PROGRAM_FORMAT_QASM3,
                                  qdmi_test::QASM3_BELL_SAMPLING),
            QDMI_SUCCESS);
  constexpr size_t shots = 4096;
  ASSERT_EQ(qdmi_test::setShots(j.job, shots), QDMI_SUCCESS);
  ASSERT_EQ(MQT_DDSIM_QDMI_device_job_submit(j.job), QDMI_SUCCESS);

  std::atomic<bool> done{false};
  std::thread poller([&] {
    QDMI_Job_Status s0 = QDMI_JOB_STATUS_CREATED;
    while (!done.load()) {
      ASSERT_EQ(MQT_DDSIM_QDMI_device_job_check(j.job, &s0), QDMI_SUCCESS);
      if (s0 == QDMI_JOB_STATUS_DONE || s0 == QDMI_JOB_STATUS_FAILED) {
        break;
      }
    }
  });

  ASSERT_EQ(MQT_DDSIM_QDMI_device_job_wait(j.job, 0), QDMI_SUCCESS);
  done.store(true);
  poller.join();
}

TEST(Concurrency, ConcurrentQIRJobsOwnTheirRuntimeState) {
  constexpr size_t numJobs = 4;
  constexpr size_t shots = 1024;
  const qdmi_test::SessionGuard session{};
  const auto program = qdmi_test::getQIRProgram("BellPairStatic.ll");
  std::vector<std::unique_ptr<qdmi_test::JobGuard>> jobs;
  jobs.reserve(numJobs);

  for (size_t i = 0; i < numJobs; ++i) {
    auto job = std::make_unique<qdmi_test::JobGuard>(session.session);
    ASSERT_EQ(qdmi_test::setProgram(job->job, QDMI_PROGRAM_FORMAT_QIRBASESTRING,
                                    program),
              QDMI_SUCCESS);
    ASSERT_EQ(qdmi_test::setShots(job->job, shots), QDMI_SUCCESS);
    ASSERT_EQ(MQT_DDSIM_QDMI_device_job_submit(job->job), QDMI_SUCCESS);
    jobs.emplace_back(std::move(job));
  }

  for (const auto& job : jobs) {
    ASSERT_EQ(MQT_DDSIM_QDMI_device_job_wait(job->job, 0), QDMI_SUCCESS);
    const auto [keys, values] = qdmi_test::getHistogram(job->job);
    ASSERT_EQ(keys.size(), values.size());
    EXPECT_EQ(std::accumulate(values.cbegin(), values.cend(), size_t{0}),
              shots);
    EXPECT_TRUE(std::ranges::all_of(
        keys, [](const auto& key) { return key == "00" || key == "11"; }));
  }
}

TEST(Concurrency, NativeProgramsRunConcurrentlyAndCancelQueuedWork) {
  const mqt::test::TemporaryDirectory directory;
  const auto path = (directory.path() / "canceled.tsv").string();
  const mqt::test::ScopedEnvironmentVariable trace(
      "MQT_CORE_DDSIM_PROGRAM_TRACE", path);
  const auto limit = workerLimit();
  const qdmi_test::SessionGuard session{};
  const qdmi_test::JobGuard job{session.session};
  const size_t count = static_cast<size_t>(limit) + 1;
  qdmi_test::ControlledJob running{job.job, count, limit};
  std::vector<QDMI_Job_Status> statuses(count);
  for (size_t i = 0; i < statuses.size(); ++i) {
    ASSERT_EQ(
        MQT_DDSIM_QDMI_device_job_get_program_status(job.job, i, &statuses[i]),
        QDMI_SUCCESS);
  }
  EXPECT_EQ(
      std::count(statuses.begin(), statuses.end(), QDMI_JOB_STATUS_RUNNING),
      limit);
  EXPECT_EQ(statuses.back(), QDMI_JOB_STATUS_QUEUED);
  ASSERT_EQ(MQT_DDSIM_QDMI_device_job_cancel(job.job), QDMI_SUCCESS);
  running.canceled();
  for (size_t i = 0; i < statuses.size(); ++i) {
    ASSERT_EQ(
        MQT_DDSIM_QDMI_device_job_get_program_status(job.job, i, &statuses[i]),
        QDMI_SUCCESS);
  }
  EXPECT_TRUE(std::ranges::all_of(statuses, [](auto status) {
    return status == QDMI_JOB_STATUS_CANCELED;
  }));
  const auto records = readProgramTrace(path);
  ASSERT_EQ(records.size(), count);
  std::vector<bool> seen(count);
  for (const auto& record : records) {
    ASSERT_LT(record[1], count);
    EXPECT_FALSE(seen[record[1]]);
    seen[record[1]] = true;
    EXPECT_EQ(record[2], QDMI_JOB_STATUS_CANCELED);
    EXPECT_EQ(record[3], limit);
    EXPECT_EQ(record[4], record[1] == count - 1 ? 0U : 1U);
  }
}

TEST(Concurrency, ProgramTraceRecordsTerminalTransitionsWithoutOverwriting) {
  const mqt::test::TemporaryDirectory directory;
  const auto path = (directory.path() / "programs.tsv").string();
  const mqt::test::ScopedEnvironmentVariable trace(
      "MQT_CORE_DDSIM_PROGRAM_TRACE", path);
  const qdmi_test::SessionGuard session{};
  const qdmi_test::JobGuard job{session.session};
  const auto* program = qdmi_test::QASM3_BELL_SAMPLING;
  const std::array<const void*, 2> programs{program, program};
  const std::array<size_t, 2> sizes{
      std::strlen(program) + 1,
      std::strlen(program) + 1,
  };
  ASSERT_EQ(MQT_DDSIM_QDMI_device_job_set_programs(
                job.job, QDMI_PROGRAM_FORMAT_QASM3, programs.size(),
                sizes.data(), programs.data()),
            QDMI_SUCCESS);
  ASSERT_EQ(qdmi_test::setShots(job.job, 64), QDMI_SUCCESS);
  const auto now = [] {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
  };
  const auto before = now();
  ASSERT_EQ(qdmi_test::submitAndWait(job.job, 0), QDMI_SUCCESS);
  const auto after = now();
  const auto buffer = llvm::MemoryBuffer::getFile(path);
  ASSERT_TRUE(buffer);
  const auto records = readProgramTrace(path);
  ASSERT_EQ(records.size(), programs.size());
  std::array<bool, 2> seen{};
  auto previous = before;
  for (const auto& values : records) {
    EXPECT_GE(values[0], previous);
    EXPECT_LE(values[0], after);
    previous = values[0];
    ASSERT_LT(values[1], seen.size());
    EXPECT_FALSE(seen[values[1]]);
    seen[values[1]] = true;
    EXPECT_EQ(values[2], QDMI_JOB_STATUS_DONE);
    EXPECT_GE(values[3], 1U);
    EXPECT_GE(values[4], 1U);
    EXPECT_LE(values[4], values[3]);
  }
  const auto probabilities = qdmi_test::getDenseProbabilities(job.job);
  ASSERT_EQ(probabilities.size(), 4);
  for (size_t i = 0; i < probabilities.size(); ++i) {
    EXPECT_NEAR(probabilities[i], i == 0 || i == 3 ? 0.5 : 0.0, 1e-12);
  }

  const qdmi_test::JobGuard another{session.session};
  ASSERT_EQ(
      qdmi_test::setProgram(another.job, QDMI_PROGRAM_FORMAT_QASM3, program),
      QDMI_SUCCESS);
  QDMI_Job_Status status = QDMI_JOB_STATUS_RUNNING;
  for (const auto* invalidPath : {"", "-"}) {
    const mqt::test::ScopedEnvironmentVariable invalidTrace(
        "MQT_CORE_DDSIM_PROGRAM_TRACE", invalidPath);
    // The Windows CRT removes a variable when assigned an empty value.
    if (*invalidPath == '\0' &&
        std::getenv("MQT_CORE_DDSIM_PROGRAM_TRACE") == nullptr) {
      continue;
    }
    EXPECT_EQ(MQT_DDSIM_QDMI_device_job_submit(another.job),
              QDMI_ERROR_INVALIDARGUMENT);
    ASSERT_EQ(MQT_DDSIM_QDMI_device_job_check(another.job, &status),
              QDMI_SUCCESS);
    EXPECT_EQ(status, QDMI_JOB_STATUS_CREATED);
  }
  EXPECT_EQ(MQT_DDSIM_QDMI_device_job_submit(another.job), QDMI_ERROR_FATAL);
  ASSERT_EQ(MQT_DDSIM_QDMI_device_job_check(another.job, &status),
            QDMI_SUCCESS);
  EXPECT_EQ(status, QDMI_JOB_STATUS_CREATED);
  const auto unchanged = llvm::MemoryBuffer::getFile(path);
  ASSERT_TRUE(unchanged);
  EXPECT_EQ((*unchanged)->getBuffer(), (*buffer)->getBuffer());
}

TEST(Concurrency, TraceWriteFailurePreservesQuantumResults) {
#ifdef _WIN32
  GTEST_SKIP() << "Requires POSIX file-size limits";
#else
  const mqt::test::TemporaryDirectory directory;
  const auto path = (directory.path() / "failed.tsv").string();
  const mqt::test::ScopedEnvironmentVariable trace(
      "MQT_CORE_DDSIM_PROGRAM_TRACE", path);
  const qdmi_test::SessionGuard session{};
  const qdmi_test::JobGuard job{session.session};
  ASSERT_EQ(qdmi_test::setProgram(job.job, QDMI_PROGRAM_FORMAT_QASM3,
                                  qdmi_test::QASM3_BELL_STATE),
            QDMI_SUCCESS);
  rlimit previous{};
  ASSERT_EQ(getrlimit(RLIMIT_FSIZE, &previous), 0);
  int result = QDMI_ERROR_FATAL;
  {
    const auto handler = ::signal(SIGXFSZ, SIG_IGN);
    ASSERT_NE(handler, SIG_ERR);
    const llvm::scope_exit restore([&] {
      EXPECT_EQ(setrlimit(RLIMIT_FSIZE, &previous), 0);
      EXPECT_NE(::signal(SIGXFSZ, handler), SIG_ERR);
    });
    const rlimit limited{.rlim_cur = 0, .rlim_max = previous.rlim_max};
    ASSERT_EQ(setrlimit(RLIMIT_FSIZE, &limited), 0);
    result = qdmi_test::submitAndWait(job.job, 0);
  }
  ASSERT_EQ(result, QDMI_SUCCESS);
  QDMI_Job_Status status = QDMI_JOB_STATUS_CREATED;
  ASSERT_EQ(MQT_DDSIM_QDMI_device_job_check(job.job, &status), QDMI_SUCCESS);
  EXPECT_EQ(status, QDMI_JOB_STATUS_DONE);
  const auto probabilities = qdmi_test::getDenseProbabilities(job.job);
  ASSERT_EQ(probabilities.size(), 4);
  for (size_t i = 0; i < probabilities.size(); ++i) {
    EXPECT_NEAR(probabilities[i], i == 0 || i == 3 ? 0.5 : 0.0, 1e-12);
  }
  const auto buffer = llvm::MemoryBuffer::getFile(path);
  ASSERT_TRUE(buffer);
  EXPECT_TRUE((*buffer)->getBuffer().empty());
#endif
}

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

#include "mqt_ddsim_qdmi/constants.h"
#include "mqt_ddsim_qdmi/device.h"

#include "helpers/test_utils.hpp"

#include "gtest/gtest.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"

#include <chrono>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <vector>

namespace qdmi_test {
/// A file barrier works across the DDSIM process boundary and proves execution
/// reached the entry point before lifecycle assertions begin.
struct ControlledJob {
private:
  std::filesystem::path directory_;
  MQT_DDSIM_QDMI_Device_Job job_;
  bool released_ = false;
  size_t count_ = 1;

  static std::string constant(const std::string& name,
                              const std::string& text) {
    constexpr std::string_view hex = "0123456789ABCDEF";
    std::string encoded;
    for (unsigned char const c : text) {
      encoded += '\\';
      encoded += hex[c >> 4U];
      encoded += hex[c & 15U];
    }
    return "@" + name + " = private constant [" +
           std::to_string(text.size() + 1) + " x i8] c\"" + encoded +
           "\\00\"\n";
  }
  void awaitFile(const std::string& name) const {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (!std::filesystem::exists(directory_ / name)) {
      if (std::chrono::steady_clock::now() >= deadline) {
        throw std::runtime_error(
            "Controlled QIR job did not reach its file barrier");
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

public:
  explicit ControlledJob(MQT_DDSIM_QDMI_Device_Job job, size_t count = 1,
                         size_t started = 0)
      : job_(job), count_(count) {
    llvm::SmallString<128> path;
    if (llvm::sys::fs::createUniqueDirectory("mqt-barrier", path)) {
      throw std::runtime_error("Could not create QIR barrier directory");
    }
    directory_ = path.str().str();
    std::vector<std::string> programs;
    for (size_t i = 0; i < count; ++i) {
      const auto index = std::to_string(i);
      programs.push_back(
          constant("started", (directory_ / ("started" + index)).string()) +
          constant("release", (directory_ / "release").string()) +
          constant("finished", (directory_ / ("finished" + index)).string()) +
          constant("read", "r") + constant("write", "w") + R"(
      declare ptr @fopen(ptr, ptr)
      declare i32 @fclose(ptr)
      define i64 @main() #0 {
        %started = call ptr @fopen(ptr @started, ptr @write)
        call i32 @fclose(ptr %started)
        br label %wait
      wait:
        %release = call ptr @fopen(ptr @release, ptr @read)
        %ready = icmp ne ptr %release, null
        br i1 %ready, label %done, label %wait
      done:
        call i32 @fclose(ptr %release)
        %finished = call ptr @fopen(ptr @finished, ptr @write)
        call i32 @fclose(ptr %finished)
        ret i64 0
      }
      attributes #0 = { "entry_point" "qir_profiles"="adaptive_profile" }
    )");
    }
    std::vector<size_t> sizes;
    std::vector<const void*> pointers;
    for (const auto& program : programs) {
      sizes.push_back(program.size() + 1);
      pointers.push_back(program.c_str());
    }
    const auto format = QDMI_PROGRAM_FORMAT_QIRADAPTIVESTRING;
    if (MQT_DDSIM_QDMI_device_job_set_programs(job, &format, count,
                                               sizes.data(), pointers.data()) !=
            QDMI_SUCCESS ||
        setShots(job, 1) != QDMI_SUCCESS ||
        MQT_DDSIM_QDMI_device_job_submit(job) != QDMI_SUCCESS) {
      throw std::runtime_error("Could not submit controlled QIR job");
    }
    try {
      for (size_t i = 0; i < (started == 0 ? count : started); ++i) {
        awaitFile("started" + std::to_string(i));
      }
    } catch (...) {
      MQT_DDSIM_QDMI_device_job_cancel(job_);
      throw;
    }
  }
  ~ControlledJob() {
    try {
      if (!released_) {
        release();
      }
    } catch (const std::exception& error) {
      ADD_FAILURE() << error.what();
      MQT_DDSIM_QDMI_device_job_cancel(job_);
    }
    std::error_code error;
    std::filesystem::remove_all(directory_, error);
  }
  ControlledJob(const ControlledJob&) = delete;
  ControlledJob& operator=(const ControlledJob&) = delete;

  void release() {
    if (!released_) {
      std::ofstream(directory_ / "release").put('1');
      for (size_t i = 0; i < count_; ++i) {
        awaitFile("finished" + std::to_string(i));
      }
      released_ = true;
    }
  }
  /// After a successful cancellation there is no worker left to release.
  void canceled() { released_ = true; }
};
} // namespace qdmi_test

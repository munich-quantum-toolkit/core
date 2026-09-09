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

#include "helpers/test_utils.hpp"
#include "mqt_ddsim_qdmi/constants.h"
#include "mqt_ddsim_qdmi/device.h"

#include <chrono>
#include <cstdint>
#include <future>
#include <stdexcept>
#include <string>

namespace qdmi_test {
struct ControlledJob;
}

/// Submit a QIR entry point that calls a barrier owned entirely by the test.
struct qdmi_test::ControlledJob {
private:
  std::promise<void> started_;
  std::promise<void> release_;
  bool released_ = false;

  static void wait(ControlledJob* self) {
    const auto ready = self->release_.get_future();
    self->started_.set_value();
    ready.wait();
  }

public:
  explicit ControlledJob(MQT_DDSIM_QDMI_Device_Job job) {
    const auto program = "define i64 @main() #0 {\n"
                         "call void inttoptr (i64 " +
                         std::to_string(reinterpret_cast<uintptr_t>(&wait)) +
                         " to ptr)(ptr inttoptr (i64 " +
                         std::to_string(reinterpret_cast<uintptr_t>(this)) +
                         " to ptr))\nret i64 0\n}\n"
                         "attributes #0 = { \"entry_point\" "
                         "\"qir_profiles\"=\"adaptive_profile\" }\n";
    if (setProgram(job, QDMI_PROGRAM_FORMAT_QIRADAPTIVESTRING, program) !=
            QDMI_SUCCESS ||
        setShots(job, 1) != QDMI_SUCCESS ||
        MQT_DDSIM_QDMI_device_job_submit(job) != QDMI_SUCCESS) {
      throw std::runtime_error("Could not submit controlled QIR job");
    }
    if (started_.get_future().wait_for(std::chrono::seconds(30)) !=
        std::future_status::ready) {
      release();
      MQT_DDSIM_QDMI_device_job_wait(job, 0);
      throw std::runtime_error("Controlled QIR job did not reach its barrier");
    }
  }
  ~ControlledJob() { release(); }
  ControlledJob(const ControlledJob&) = delete;
  ControlledJob& operator=(const ControlledJob&) = delete;

  void release() {
    if (!released_) {
      release_.set_value();
      released_ = true;
    }
  }
};

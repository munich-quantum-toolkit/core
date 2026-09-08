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

#include "mqt_ddsim_qdmi/device.h"
#include "qdmi/devices/dd/Device.hpp"

#include <future>

/// Hold a real job worker in RUNNING until the test releases it.
struct qdmi_test::ControlledJob {
private:
  std::promise<void> release_;
  bool released_ = false;

public:
  explicit ControlledJob(MQT_DDSIM_QDMI_Device_Job job) {
    std::promise<void> started;
    job->submitProgramAsync([&started, ready = release_.get_future().share()] {
      started.set_value();
      ready.wait();
      return true;
    });
    started.get_future().wait();
  }
  ~ControlledJob() { release(); }
  ControlledJob(const ControlledJob&) = delete;
  ControlledJob& operator=(const ControlledJob&) = delete;

  /// Wait for worker cleanup while keeping the job allocated.
  void finish(MQT_DDSIM_QDMI_Device_Job job) {
    release();
    job->jobHandle_.wait();
  }

  void release() {
    if (!released_) {
      release_.set_value();
      released_ = true;
    }
  }
};

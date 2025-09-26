/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "fomac/FoMaC.hpp"
#include "na/fomac/Device.hpp"

#include <utility>
#include <vector>

namespace na::zoned {
class FoMaC : public na::FoMaC {
public:
  class Device : public na::FoMaC::Device {

  public:
    explicit Device(const fomac::FoMaC::Device& device)
        : na::FoMaC::Device(device) {}
    explicit Device(na::FoMaC::Device&& device)
        : na::FoMaC::Device(std::move(device)) {}
  };
  FoMaC() = delete;
  [[nodiscard]] static auto getDevices() -> std::vector<Device>;
};
} // namespace na::zoned

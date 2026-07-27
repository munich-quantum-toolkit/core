/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "fomac/FoMaC.hpp"

#include <array>
#include <cstddef>

int main() {
  const auto submitBinaryProgram = [](const fomac::Device& device) {
    constexpr std::array<std::byte, 0> program{};
    return device.submitJob(program, QDMI_PROGRAM_FORMAT_QIRBASEMODULE, 0);
  };
  static_cast<void>(submitBinaryProgram);

  fomac::Session session;
  static_cast<void>(session.getDevices());
  return 0;
}

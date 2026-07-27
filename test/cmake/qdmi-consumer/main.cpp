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

int main(const int argc, char**) {
  fomac::Session session;
  const auto devices = session.getDevices();
  if (argc > 1 && !devices.empty()) {
    constexpr std::array program{std::byte{0}};
    static_cast<void>(devices.front().submitJob(
        program, QDMI_PROGRAM_FORMAT_QIRBASEMODULE, 0));
  }
  return 0;
}

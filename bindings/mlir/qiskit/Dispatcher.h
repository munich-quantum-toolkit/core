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

#include "QiskitAdapter.h"

#include <memory>
#include <string>

namespace mqt::bindings::qiskit {

struct InstalledVersion {
  unsigned int major = 0;
  unsigned int minor = 0;
  unsigned int patch = 0;
  std::string text;
};

[[nodiscard]] InstalledVersion inspectInstalledVersion();
[[nodiscard]] bool hasSupportedAdapter(const InstalledVersion& version);
[[nodiscard]] std::unique_ptr<Adapter> selectAdapter();

} // namespace mqt::bindings::qiskit

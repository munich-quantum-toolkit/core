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
 * Test utilities for QIR-based tests.
 */
#pragma once

#include <string>
#include <string_view>

namespace qir_test {

/// Read a QIR source file from the test circuits directory and
/// return its contents as a string.
std::string getProgram(std::string_view file);

} // namespace qir_test

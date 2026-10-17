/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#define MQT_QISKIT_VERSION_FACTORY createQiskit2_6
#define MQT_QISKIT_VERSION_EXPECTED_MAJOR 2U
#define MQT_QISKIT_VERSION_EXPECTED_MINOR 6U
#define MQT_QISKIT_VERSION_EXACT_API 0
#define MQT_QISKIT_VERSION_LABEL "2.6"
// Compile the shared implementation against this minor's private C API table.
// NOLINTNEXTLINE(bugprone-suspicious-include)
#include "Qiskit2_5.cpp"

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

#include "QiskitTranslation.h"

#include "llvm/ADT/StringRef.h"

#include <memory>

namespace mqt::bindings::qiskit {

/// Check installed package metadata without importing Qiskit.
[[nodiscard]] bool supportsVersion(llvm::StringRef version);

[[nodiscard]] std::unique_ptr<VersionedTranslation> selectTranslation();

} // namespace mqt::bindings::qiskit

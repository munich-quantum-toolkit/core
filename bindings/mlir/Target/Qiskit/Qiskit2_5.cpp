/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "QiskitAdapter.h"
#include "QiskitTranslation.h"

#include <memory>

namespace mqt::bindings::qiskit {
std::unique_ptr<VersionedTranslation> createQiskit2_5() {
  return createTranslation();
}
} // namespace mqt::bindings::qiskit

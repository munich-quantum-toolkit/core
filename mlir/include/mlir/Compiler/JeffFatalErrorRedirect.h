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

#include "mlir/Compiler/JeffDeserializerError.h"

#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/Twine.h>
#include <llvm/Support/ErrorHandling.h>

namespace llvm {

[[noreturn]] inline void throwJeffDeserializerError(const char* reason,
                                                    bool = true) {
  throw mlir::detail::JeffDeserializerError(reason);
}

[[noreturn]] inline void throwJeffDeserializerError(const StringRef reason,
                                                    bool = true) {
  throw mlir::detail::JeffDeserializerError(reason.str());
}

[[noreturn]] inline void throwJeffDeserializerError(const Twine& reason,
                                                    bool = true) {
  throw mlir::detail::JeffDeserializerError(reason.str());
}

} // namespace llvm

// jeff-mlir currently reports invalid serialized input through LLVM's fatal
// error API. Redirect only its translation target to the recoverable exception
// above; the public MQT import boundary preflights assertion-prone shapes, then
// catches and diagnoses any remaining dependency failures.
#define report_fatal_error throwJeffDeserializerError

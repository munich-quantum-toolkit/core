/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qdmi/driver/Driver.hpp"

#include "qdmi/client.h"

#include <new>

int QDMI_session_alloc(QDMI_Session* session) {
  if (session == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  *session = nullptr;
  try {
    return qdmi::Driver::get().sessionAlloc(session);
  } catch (const std::bad_alloc&) {
    return QDMI_ERROR_OUTOFMEM;
  } catch (...) {
    return QDMI_ERROR_FATAL;
  }
}

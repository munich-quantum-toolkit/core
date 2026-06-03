/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 MQSC GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "dd/Edge.hpp"
#include "dd/Node.hpp"

namespace dd {
class Package;

using VectorDD = Edge<vNode>;
using MatrixDD = Edge<mNode>;
} // namespace dd

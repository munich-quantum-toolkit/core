/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qc_programs.h"
#include "test_qc_ir.h"

#include <gtest/gtest.h>

using namespace mlir::qc;

INSTANTIATE_TEST_SUITE_P(
    QCIDOpTest, QCTest,
    testing::Values(
        QCTestCase{"Identity", MQT_NAMED_BUILDER(identity),
                   MQT_NAMED_BUILDER(identity)},
        QCTestCase{"SingleControlledIdentity",
                   MQT_NAMED_BUILDER(singleControlledIdentity),
                   MQT_NAMED_BUILDER(identity)},
        QCTestCase{"MultipleControlledIdentity",
                   MQT_NAMED_BUILDER(multipleControlledIdentity),
                   MQT_NAMED_BUILDER(identity)},
        QCTestCase{"NestedControlledIdentity",
                   MQT_NAMED_BUILDER(nestedControlledIdentity),
                   MQT_NAMED_BUILDER(identity)},
        QCTestCase{"TrivialControlledIdentity",
                   MQT_NAMED_BUILDER(trivialControlledIdentity),
                   MQT_NAMED_BUILDER(identity)},
        QCTestCase{"InverseIdentity", MQT_NAMED_BUILDER(inverseIdentity),
                   MQT_NAMED_BUILDER(identity)},
        QCTestCase{"InverseMultipleControlledIdentity",
                   MQT_NAMED_BUILDER(inverseMultipleControlledIdentity),
                   MQT_NAMED_BUILDER(identity)}));

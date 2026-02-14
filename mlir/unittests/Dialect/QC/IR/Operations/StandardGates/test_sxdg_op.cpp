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
    QCSXdgOpTest, QCTest,
    testing::Values(QCTestCase{"SXdg", MQT_NAMED_BUILDER(sxdg),
                               MQT_NAMED_BUILDER(sxdg)},
                    QCTestCase{"SingleControlledSXdg",
                               MQT_NAMED_BUILDER(singleControlledSxdg),
                               MQT_NAMED_BUILDER(singleControlledSxdg)},
                    QCTestCase{"MultipleControlledSXdg",
                               MQT_NAMED_BUILDER(multipleControlledSxdg),
                               MQT_NAMED_BUILDER(multipleControlledSxdg)},
                    QCTestCase{"NestedControlledSXdg",
                               MQT_NAMED_BUILDER(nestedControlledSxdg),
                               MQT_NAMED_BUILDER(multipleControlledSxdg)},
                    QCTestCase{"TrivialControlledSXdg",
                               MQT_NAMED_BUILDER(trivialControlledSxdg),
                               MQT_NAMED_BUILDER(sxdg)},
                    QCTestCase{"InverseSXdg", MQT_NAMED_BUILDER(inverseSxdg),
                               MQT_NAMED_BUILDER(sx)},
                    QCTestCase{"InverseMultipleControlledSXdg",
                               MQT_NAMED_BUILDER(inverseMultipleControlledSxdg),
                               MQT_NAMED_BUILDER(multipleControlledSx)}));

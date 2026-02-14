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
    QCTdgOpTest, QCTest,
    testing::Values(QCTestCase{"Tdg", MQT_NAMED_BUILDER(tdg),
                               MQT_NAMED_BUILDER(tdg)},
                    QCTestCase{"SingleControlledTdg",
                               MQT_NAMED_BUILDER(singleControlledTdg),
                               MQT_NAMED_BUILDER(singleControlledTdg)},
                    QCTestCase{"MultipleControlledTdg",
                               MQT_NAMED_BUILDER(multipleControlledTdg),
                               MQT_NAMED_BUILDER(multipleControlledTdg)},
                    QCTestCase{"NestedControlledTdg",
                               MQT_NAMED_BUILDER(nestedControlledTdg),
                               MQT_NAMED_BUILDER(multipleControlledTdg)},
                    QCTestCase{"TrivialControlledTdg",
                               MQT_NAMED_BUILDER(trivialControlledTdg),
                               MQT_NAMED_BUILDER(tdg)},
                    QCTestCase{"InverseTdg", MQT_NAMED_BUILDER(inverseTdg),
                               MQT_NAMED_BUILDER(t_)},
                    QCTestCase{"InverseMultipleControlledTdg",
                               MQT_NAMED_BUILDER(inverseMultipleControlledTdg),
                               MQT_NAMED_BUILDER(multipleControlledT)}));

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
    QCSdgOpTest, QCTest,
    testing::Values(QCTestCase{"Sdg", MQT_NAMED_BUILDER(sdg),
                               MQT_NAMED_BUILDER(sdg)},
                    QCTestCase{"SingleControlledSdg",
                               MQT_NAMED_BUILDER(singleControlledSdg),
                               MQT_NAMED_BUILDER(singleControlledSdg)},
                    QCTestCase{"MultipleControlledSdg",
                               MQT_NAMED_BUILDER(multipleControlledSdg),
                               MQT_NAMED_BUILDER(multipleControlledSdg)},
                    QCTestCase{"NestedControlledSdg",
                               MQT_NAMED_BUILDER(nestedControlledSdg),
                               MQT_NAMED_BUILDER(multipleControlledSdg)},
                    QCTestCase{"TrivialControlledSdg",
                               MQT_NAMED_BUILDER(trivialControlledSdg),
                               MQT_NAMED_BUILDER(sdg)},
                    QCTestCase{"InverseSdg", MQT_NAMED_BUILDER(inverseSdg),
                               MQT_NAMED_BUILDER(s)},
                    QCTestCase{"InverseMultipleControlledSdg",
                               MQT_NAMED_BUILDER(inverseMultipleControlledSdg),
                               MQT_NAMED_BUILDER(multipleControlledS)}));

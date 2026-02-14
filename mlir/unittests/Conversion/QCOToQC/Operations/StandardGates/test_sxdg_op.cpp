/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "TestCaseUtils.h"
#include "qc_programs.h"
#include "qco_programs.h"
#include "test_qco_to_qc.h"

#include <gtest/gtest.h>

using namespace mlir::qco;

INSTANTIATE_TEST_SUITE_P(
    QCOSXdgOpTest, QCOToQCTest,
    testing::Values(
        QCOToQCTestCase{"SXdg", MQT_NAMED_BUILDER(qco::sxdg),
                        MQT_NAMED_BUILDER(qc::sxdg)},
        QCOToQCTestCase{"SingleControlledSXdg",
                        MQT_NAMED_BUILDER(qco::singleControlledSxdg),
                        MQT_NAMED_BUILDER(qc::singleControlledSxdg)},
        QCOToQCTestCase{"MultipleControlledSXdg",
                        MQT_NAMED_BUILDER(qco::multipleControlledSxdg),
                        MQT_NAMED_BUILDER(qc::multipleControlledSxdg)}));

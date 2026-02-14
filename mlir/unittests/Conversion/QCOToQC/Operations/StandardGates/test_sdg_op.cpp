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
    QCOSdgOpTest, QCOToQCTest,
    testing::Values(QCOToQCTestCase{"Sdg", MQT_NAMED_BUILDER(qco::sdg),
                                    MQT_NAMED_BUILDER(qc::sdg)},
                    QCOToQCTestCase{"SingleControlledSdg",
                                    MQT_NAMED_BUILDER(qco::singleControlledSdg),
                                    MQT_NAMED_BUILDER(qc::singleControlledSdg)},
                    QCOToQCTestCase{
                        "MultipleControlledSdg",
                        MQT_NAMED_BUILDER(qco::multipleControlledSdg),
                        MQT_NAMED_BUILDER(qc::multipleControlledSdg)}));

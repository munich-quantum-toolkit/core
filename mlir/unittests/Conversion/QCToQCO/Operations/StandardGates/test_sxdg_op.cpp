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
#include "qco_programs.h"
#include "test_qc_to_qco.h"

#include <gtest/gtest.h>

using namespace mlir::qc;

INSTANTIATE_TEST_SUITE_P(
    QCSXdgOpTest, QCToQCOTest,
    testing::Values(
        QCToQCOTestCase{"SXdg", MQT_NAMED_BUILDER(qc::sxdg),
                        MQT_NAMED_BUILDER(qco::sxdg)},
        QCToQCOTestCase{"SingleControlledSXdg",
                        MQT_NAMED_BUILDER(qc::singleControlledSxdg),
                        MQT_NAMED_BUILDER(qco::singleControlledSxdg)},
        QCToQCOTestCase{"MultipleControlledSXdg",
                        MQT_NAMED_BUILDER(qc::multipleControlledSxdg),
                        MQT_NAMED_BUILDER(qco::multipleControlledSxdg)}));

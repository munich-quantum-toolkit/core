/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "Definitions.hpp"
#include "ir/operations/OpType.hpp"
#include "na/NAComputation.hpp"
#include "na/NAUtils.hpp"
#include "na/operations/GlobalOp.hpp"
#include "na/operations/GlobalRYOp.hpp"
#include "na/operations/LoadOp.hpp"
#include "na/operations/LocalRZOp.hpp"
#include "na/operations/MoveOp.hpp"
#include "na/operations/StoreOp.hpp"

#include <gtest/gtest.h>
#include <memory>
#include <sstream>
#include <vector>

namespace na {
TEST(NAComputation, General) {
  auto qc = NAComputation();
  const auto* const atom0 = qc.emplaceBackAtom("atom0");
  const auto* const atom1 = qc.emplaceBackAtom("atom1");
  const auto* const atom2 = qc.emplaceBackAtom("atom2");
  const auto* const globalZone = qc.emplaceBackZone("global");
  qc.emplaceBackInitialLocation(atom0, 0, 0);
  qc.emplaceBackInitialLocation(atom1, 1, 0);
  qc.emplaceBackInitialLocation(atom2, 2, 0);
  qc.emplaceBack<LocalRZOp>(qc::PI_2, atom0);
  qc.emplaceBack<GlobalRYOp>(qc::PI_2, globalZone);
  qc.emplaceBack<LoadOp>(std::vector{atom0, atom1}, std::vector{Location{0, 1}, Location{1, 1}});
  qc.emplaceBack<MoveOp>(std::vector{atom0, atom1}, std::vector{Location{4, 1}, Location{5, 1}});
  qc.emplaceBack<StoreOp>(std::vector{atom0, atom1}, std::vector{Location{4, 0}, Location{5, 0}});
  std::stringstream ss;
  ss << qc;
  EXPECT_EQ(ss.str(), "atom (0.000, 0.000) atom0\n"
                      "atom (1.000, 0.000) atom1\n"
                      "atom (2.000, 0.000) atom2\n"
                      "@+ rz 1.57080 atom0\n"
                      "@+ ry 1.57080 global\n"
                      "@+ load [\n"
                      "    (0.000, 1.000) atom0\n"
                      "    (1.000, 1.000) atom1\n"
                      "]\n"
                      "@+ move [\n"
                      "    (4.000, 1.000) atom0\n"
                      "    (5.000, 1.000) atom1\n"
                      "]\n"
                      "@+ store [\n"
                      "    (4.000, 0.000) atom0\n"
                      "    (5.000, 0.000) atom1\n"
                      "]\n");
}

TEST(NAComputation, EmptyPrint) {
  const NAComputation qc;
  std::stringstream ss;
  ss << qc;
  EXPECT_EQ(ss.str(), "");
}

TEST(NAComputation, ValidateAODConstraints) {
  auto qc = NAComputation();
  const auto* const atom0 = qc.emplaceBackAtom("atom0");
  const auto* const atom1 = qc.emplaceBackAtom("atom1");
  const auto* const atom2 = qc.emplaceBackAtom("atom2");
  const auto* const atom3 = qc.emplaceBackAtom("atom3");
  qc.emplaceBackInitialLocation(atom0, 0, 0);
  qc.emplaceBackInitialLocation(atom1, 1, 0);
  qc.emplaceBackInitialLocation(atom2, 0, 2);
  qc.emplaceBackInitialLocation(atom3, 1, 2);
  qc.emplaceBack(LoadOp({atom0, atom1}, {Location{0, 1}, Location{1, 1}}));
  EXPECT_TRUE(qc.validate());
  // atom already loaded
  qc.emplaceBack(LoadOp({atom0}, {Location{0, 1}}));
  EXPECT_FALSE(qc.validate());
  qc.clear();
  // atom not loaded
  qc.emplaceBack(MoveOp({atom0}, {Location{0, 1}}));
  EXPECT_FALSE(qc.validate());
  qc.clear();
  // two atoms identical
  qc.emplaceBack(LoadOp({atom0, atom0}, {Location{0, 1}, Location{1, 1}}));
  EXPECT_FALSE(qc.validate());
  qc.clear();
  // two end points identical
  qc.emplaceBack(LoadOp({atom0, atom1}, {Location{0, 1}, Location{0, 1}}));
  EXPECT_FALSE(qc.validate());
  qc.clear();
  // columns not preserved
  qc.emplaceBack(LoadOp({atom1, atom3}, {Location{0, 1}, Location{2, 2}}));
  EXPECT_FALSE(qc.validate());
  qc.clear();
  // rows not preserved
  qc.emplaceBack(LoadOp({atom0, atom1}, {Location{0, 1}, Location{1, -1}}));
  EXPECT_FALSE(qc.validate());
  qc.clear();
  // column order not preserved
  qc.emplaceBack(LoadOp({atom0, atom3}, {Location{1, 1}, Location{0, 1}}));
  EXPECT_FALSE(qc.validate());
  qc.clear();
  // row order not preserved
  qc.emplaceBack(LoadOp({atom0, atom3}, {Location{0, 1}, Location{2, 0}}));
  EXPECT_FALSE(qc.validate());
  qc.clear();
  // column order not preserved
  qc.emplaceBack(LoadOp({atom2, atom1}, {Location{1, 3}, Location{0, 1}}));
  EXPECT_FALSE(qc.validate());
  qc.clear();
  // row order not preserved
  qc.emplaceBack(LoadOp({atom2, atom1}, {Location{0, 1}, Location{2, 2}}));
  EXPECT_FALSE(qc.validate());
  qc.clear();
  // two atoms identical
  qc.emplaceBack(LocalRZOp(qc::PI_2, {atom0, atom0}));
  EXPECT_FALSE(qc.validate());
}
} // namespace na

/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Compiler/Programs.h"

#include "gtest/gtest.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {
TEST(ProgramInspection, InspectAllocatedRegistersAndControlFlow) {
  auto qc = mlir::QCProgram::fromOpenQASMString(R"(
    OPENQASM 3.0;
    include "stdgates.inc";
    qubit[2] a;
    qubit b;
    bit flag = measure b;
    if (flag) { x a[0]; }
  )");
  ASSERT_TRUE(qc);
  EXPECT_EQ(qc->inspect().numQubits, 3);
  EXPECT_TRUE(qc->inspect().staticQubits.empty());
  EXPECT_TRUE(qc->inspect().hasControlFlow);
  auto qco = std::move(*qc).intoQCO();
  ASSERT_TRUE(qco);
  EXPECT_EQ(qco->inspect().numQubits, 3);
  EXPECT_TRUE(qco->inspect().hasControlFlow);
}

TEST(ProgramInspection, InspectStaticSitesWithoutTreatingIDsAsCounts) {
  auto qc = mlir::QCProgram::fromOpenQASMString(R"(
    OPENQASM 3.0; include "stdgates.inc";
    x $5; h $2; z $5;
  )");
  ASSERT_TRUE(qc);
  const std::vector<uint64_t> sites{2, 5};
  EXPECT_EQ(qc->inspect().numQubits, 2);
  EXPECT_EQ(qc->inspect().staticQubits, sites);
  EXPECT_FALSE(qc->inspect().hasControlFlow);
  auto qco = std::move(*qc).intoQCO();
  ASSERT_TRUE(qco);
  EXPECT_EQ(qco->inspect().numQubits, 2);
  EXPECT_EQ(qco->inspect().staticQubits, sites);
}

TEST(ProgramInspection, UnknownAndEmptyWidthsDiffer) {
  auto empty = mlir::QCProgram::fromMLIRString(
      "module { func.func @main() attributes {mqt.entry_point} { return } }");
  auto library = mlir::QCProgram::fromMLIRString("module {}");
  auto dynamic = mlir::QCProgram::fromMLIRString(R"(
    module {
      func.func @main(%n: index) attributes {mqt.entry_point} {
        %q = memref.alloc(%n) : memref<?x!qc.qubit>
        memref.dealloc %q : memref<?x!qc.qubit>
        return
      }
    }
  )");
  ASSERT_TRUE(empty);
  ASSERT_TRUE(library);
  ASSERT_TRUE(dynamic);
  EXPECT_EQ(empty->inspect().numQubits, 0);
  EXPECT_TRUE(empty->gateCounts().empty());
  EXPECT_EQ(empty->staticDepth(), 0);
  EXPECT_FALSE(library->inspect().numQubits);
  EXPECT_TRUE(library->gateCounts().empty());
  EXPECT_FALSE(library->staticDepth());
  EXPECT_FALSE(dynamic->inspect().numQubits);
  auto qco = std::move(*dynamic).intoQCO();
  ASSERT_TRUE(qco);
  EXPECT_FALSE(qco->inspect().numQubits);
}

TEST(ProgramInspection, StaticDepthTracksPhysicalAliases) {
  auto qc = mlir::QCProgram::fromOpenQASMString(R"(
    OPENQASM 3.0; include "stdgates.inc";
    h $5; z $5; x $2; gphase(0.3);
  )");
  ASSERT_TRUE(qc);
  EXPECT_EQ(qc->staticDepth(), 2);
  EXPECT_EQ(qc->gateCounts().at("gphase"), 1);
}

TEST(ProgramInspection, StaticDepthTracksScalarAllocations) {
  auto qc = mlir::QCProgram::fromMLIRString(R"(
    module {
      func.func @main() attributes {mqt.entry_point} {
        %q = qc.alloc : !qc.qubit
        qc.h %q : !qc.qubit
        qc.x %q : !qc.qubit
        qc.dealloc %q : !qc.qubit
        return
      }
    }
  )");
  ASSERT_TRUE(qc);
  EXPECT_EQ(qc->staticDepth(), 2);
}

TEST(ProgramInspection, StoredReferencesHaveUnknownDepthNotExtraWidth) {
  auto qc = mlir::QCProgram::fromMLIRString(R"(
    module {
      func.func @main() attributes {mqt.entry_point} {
        %i = arith.constant 0 : index
        %q = qc.alloc : !qc.qubit
        %refs = memref.alloca() : memref<1x!qc.qubit>
        memref.store %q, %refs[%i] : memref<1x!qc.qubit>
        qc.h %q : !qc.qubit
        %alias = memref.load %refs[%i] : memref<1x!qc.qubit>
        qc.x %alias : !qc.qubit
        qc.dealloc %q : !qc.qubit
        return
      }
    }
  )");
  ASSERT_TRUE(qc);
  EXPECT_EQ(qc->inspect().numQubits, 1);
  EXPECT_FALSE(qc->staticDepth());
}

TEST(ProgramInspection, RegisterViewsHaveUnknownDepth) {
  auto qc = mlir::QCProgram::fromMLIRString(R"(
    module {
      func.func @main() attributes {mqt.entry_point} {
        %i = arith.constant 0 : index
        %q = memref.alloc() : memref<1x!qc.qubit>
        %view = memref.cast %q : memref<1x!qc.qubit> to memref<?x!qc.qubit>
        %a = memref.load %q[%i] : memref<1x!qc.qubit>
        qc.h %a : !qc.qubit
        %b = memref.load %view[%i] : memref<?x!qc.qubit>
        qc.x %b : !qc.qubit
        memref.dealloc %q : memref<1x!qc.qubit>
        return
      }
    }
  )");
  ASSERT_TRUE(qc);
  EXPECT_EQ(qc->inspect().numQubits, 1);
  EXPECT_FALSE(qc->staticDepth());
}

TEST(ProgramInspection, QuantumInputsAndUnstructuredControlFlowAreUnknown) {
  auto input = mlir::QCProgram::fromMLIRString(R"(
    module {
      func.func @main(%q: !qc.qubit) attributes {mqt.entry_point} {
        qc.h %q : !qc.qubit
        return
      }
    }
  )");
  ASSERT_TRUE(input);
  EXPECT_FALSE(input->inspect().numQubits);
  EXPECT_FALSE(input->staticDepth());
  auto branches = mlir::QCProgram::fromMLIRString(R"(
    module {
      func.func @main() attributes {mqt.entry_point} {
        cf.br ^next
      ^next:
        return
      }
    }
  )");
  ASSERT_TRUE(branches);
  EXPECT_TRUE(branches->inspect().hasControlFlow);
  EXPECT_FALSE(branches->staticDepth());
}

TEST(ProgramInspection, InspectionIncludesHelpersButNotNestedModules) {
  auto qc = mlir::QCProgram::fromMLIRString(R"(
    module {
      func.func @main() attributes {mqt.entry_point} { return }
      func.func @helper(%condition: i1) {
        %q = qc.static 5 : !qc.qubit
        scf.if %condition { qc.h %q : !qc.qubit }
        return
      }
      module {
        func.func @nested() {
          %q = qc.static 99 : !qc.qubit
          qc.x %q : !qc.qubit
          return
        }
      }
    }
  )");
  ASSERT_TRUE(qc);
  EXPECT_EQ(qc->inspect().staticQubits, (std::vector<uint64_t>{5}));
  EXPECT_EQ(qc->inspect().numQubits, 1);
  EXPECT_TRUE(qc->inspect().hasControlFlow);
  EXPECT_TRUE(qc->gateCounts().empty());
  EXPECT_EQ(qc->staticDepth(), 0);
}

TEST(ProgramInspection, DepthHasBoundedNesting) {
  for (const auto nesting : {127, 128}) {
    std::string source = R"(module {
      func.func @main(%flag: i1) attributes {mqt.entry_point} {
        %q = qc.static 0 : !qc.qubit
    )";
    for (int i = 0; i < nesting; ++i) {
      source += "scf.if %flag {\n";
    }
    source += "qc.h %q : !qc.qubit\n";
    source.append(static_cast<size_t>(nesting), '}');
    source += "return } }";
    auto qc = mlir::QCProgram::fromMLIRString(source);
    ASSERT_TRUE(qc);
    EXPECT_EQ(qc->staticDepth(),
              nesting == 127 ? std::optional<size_t>{1} : std::nullopt);
  }
}
} // namespace

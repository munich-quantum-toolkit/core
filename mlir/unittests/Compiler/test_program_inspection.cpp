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
#include <map>
#include <string>
#include <utility>
#include <vector>

template <class Program>
static void expectCounts(const Program& program, const size_t single,
                         const size_t two,
                         const std::map<std::string, size_t>& expected) {
  size_t total = 0;
  for (const auto& [symbol, count] : expected) {
    total += count;
  }
  const auto info = program.inspect();
  EXPECT_EQ(info.numGates, total);
  EXPECT_EQ(info.numSingleQubitGates, single);
  EXPECT_EQ(info.numTwoQubitGates, two);
  EXPECT_EQ(info.gateCounts, expected);
  EXPECT_EQ(program.numGates(), total);
  EXPECT_EQ(program.numSingleQubitGates(), single);
  EXPECT_EQ(program.numTwoQubitGates(), two);
  EXPECT_EQ(program.gateCounts(), expected);
}

namespace {
TEST(ProgramInspection, CountCallsAndNestedModifiersOnce) {
  auto qc = mlir::QCProgram::fromOpenQASMString(R"(
    OPENQASM 3.0; include "stdgates.inc";
    gate foo a { h a; x a; }
    qubit[3] q;
    bit c;
    foo q[0];
    ctrl @ inv @ x q[0], q[1];
    inv @ h q[0];
    pow(2) @ h q[1];
    gphase(0.3);
    h q[0];
    swap q[0], q[1];
    ccx q[0], q[1], q[2];
    ctrl @ swap q[0], q[1], q[2];
    inv @ cx q[0], q[1];
    barrier q[0];
    barrier q[0], q[1];
    c = measure q[0];
    reset q[1];
  )");
  ASSERT_TRUE(qc);
  const std::map<std::string, size_t> expected{
      {"foo", 1},    {"ctrl", 3}, {"inv", 2},  {"pow", 1},
      {"gphase", 1}, {"h", 1},    {"swap", 1},
  };
  expectCounts(*qc, 4, 3, expected);
  EXPECT_FALSE(qc->inspect().hasControlFlow);
  auto qco = std::move(*qc).intoQCO();
  ASSERT_TRUE(qco);
  expectCounts(*qco, 4, 3, expected);
}

TEST(ProgramInspection, QCOCountsNativeThreeQubitGates) {
  auto qco = mlir::QCOProgram::fromMLIRString(R"(
    module {
      func.func @main() attributes {mqt.entry_point} {
        %a = qco.alloc : !qco.qubit
        %b = qco.alloc : !qco.qubit
        %c = qco.alloc : !qco.qubit
        %x, %y, %z = qco.rccx %a, %b, %c : !qco.qubit, !qco.qubit, !qco.qubit -> !qco.qubit, !qco.qubit, !qco.qubit
        qco.sink %x : !qco.qubit
        qco.sink %y : !qco.qubit
        qco.sink %z : !qco.qubit
        return
      }
    }
  )");
  ASSERT_TRUE(qco);
  expectCounts(*qco, 0, 0, {{"rccx", 1}});
}

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

TEST(ProgramInspection, InspectSortedDistinctStaticSites) {
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
  EXPECT_FALSE(library->inspect().numQubits);
  EXPECT_TRUE(library->gateCounts().empty());
  EXPECT_FALSE(dynamic->inspect().numQubits);
  auto qco = std::move(*dynamic).intoQCO();
  ASSERT_TRUE(qco);
  EXPECT_FALSE(qco->inspect().numQubits);
}

TEST(ProgramInspection, StoredReferencesDoNotAddWidth) {
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
}

TEST(ProgramInspection, AllocationWidthOverflowIsUnknown) {
  for (const std::string allocations : {
           R"(
        %a = memref.alloc() : memref<9223372036854775807x!qc.qubit>
        %b = memref.alloc() : memref<9223372036854775807x!qc.qubit>
        %c = memref.alloc() : memref<2x!qc.qubit>
        %q = qc.alloc : !qc.qubit
           )",
           "%q = memref.alloc() : memref<4294967296x4294967296x!qc.qubit>",
       }) {
    SCOPED_TRACE(allocations);
    auto qc = mlir::QCProgram::fromMLIRString(
        "module { func.func @main() attributes {mqt.entry_point} { " +
        allocations + " return } }");
    ASSERT_TRUE(qc);
    EXPECT_FALSE(qc->inspect().numQubits);
  }
}

TEST(ProgramInspection, QuantumInputsHaveUnknownWidthAndBranchesAreDetected) {
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
  EXPECT_EQ(qc->inspect().numGates, 0);
  EXPECT_TRUE(qc->inspect().gateCounts.empty());
}

TEST(ProgramInspection, CountGatesWithoutEntryPoint) {
  auto qc = mlir::QCProgram::fromMLIRString(R"mlir(module {
    func.func @helper(%qubit: !qc.qubit) {
      qc.h %qubit : !qc.qubit
      return
    }
  })mlir");
  ASSERT_TRUE(qc);
  expectCounts(*qc, 0, 0, {});
  auto qco = std::move(*qc).intoQCO();
  ASSERT_TRUE(qco);
  expectCounts(*qco, 0, 0, {});
}

TEST(ProgramInspection, CountGatesInStructuredControlFlow) {
  const std::string qasm = R"(OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
bit condition = measure q[0];
int selector = 1;
if (condition) {
  for int i in [0:2] {
    x q[i];
  }
} else {
  cx q[0], q[1];
}
while (condition) {
  ctrl @ x q[0], q[1];
}
switch (selector) {
  case 1 {
    swap q[0], q[1];
  }
  default {
    z q[2];
  }
}
)";
  auto qc = mlir::QCProgram::fromOpenQASMString(qasm);
  ASSERT_TRUE(qc);
  const std::map<std::string, size_t> expectedCounts{
      {"ctrl", 2},
      {"swap", 1},
      {"x", 1},
      {"z", 1},
  };
  expectCounts(*qc, 2, 3, expectedCounts);
  auto qco = std::move(*qc).intoQCO();
  ASSERT_TRUE(qco);
  expectCounts(*qco, 2, 3, expectedCounts);
}

} // namespace

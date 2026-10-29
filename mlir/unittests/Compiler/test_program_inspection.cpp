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

#include "llvm/Support/LogicalResult.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
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
}

template <class Program>
static void expectDepth(const Program& program,
                        const std::optional<size_t> expected) {
  EXPECT_EQ(program.inspect().depth, expected);
  EXPECT_EQ(program.depth(), expected);
}

static void expectDepthInBothDialects(mlir::QCProgram qc,
                                      const std::optional<size_t> expected) {
  expectDepth(qc, expected);
  auto qco = std::move(qc).intoQCO();
  ASSERT_TRUE(mlir::succeeded(qco));
  expectDepth(*qco, expected);
}

static void expectDepthInBothDialects(const std::string_view qasm,
                                      const std::optional<size_t> expected) {
  auto qc = mlir::QCProgram::fromOpenQASMString(qasm);
  ASSERT_TRUE(mlir::succeeded(qc));
  expectDepthInBothDialects(std::move(*qc), expected);
}

namespace {
TEST(ProgramInspection, CountCallsAndNestedModifiersOnce) {
  auto qc = mlir::QCProgram::fromOpenQASMString(R"(
    OPENQASM 3.0; include "stdgates.inc";
    gate foo a { h a; x a; }
    qubit[3] q;
    bit c;
    foo q[0];
    ctrl @ foo q[0], q[1];
    ctrl @ inv @ x q[0], q[1];
    ctrl(2) @ inv @ x q[0], q[1], q[2];
    inv @ h q[0];
    pow(2) @ h q[1];
    gphase(0.3);
    h q[0];
    swap q[0], q[1];
    ccx q[0], q[1], q[2];
    ctrl @ swap q[0], q[1], q[2];
    inv @ cx q[0], q[1];
    cx q[0], q[1];
    cz q[0], q[1];
    crx(0.2) q[0], q[1];
    crx(0.4) q[0], q[1];
    barrier q[0];
    barrier q[0], q[1];
    c = measure q[0];
    reset q[1];
  )");
  ASSERT_TRUE(mlir::succeeded(qc));
  const std::map<std::string, size_t> expected{
      {"foo", 1},     {"ctrl(foo)", 1}, {"ctrl(inv(x))", 1},
      {"inv(h)", 1},  {"pow(h)", 1},    {"ctrl(2,inv(x))", 1},
      {"gphase", 1},  {"h", 1},         {"swap", 1},
      {"ccx", 1},     {"cswap", 1},     {"inv(cx)", 1},
      {"cx", 1},      {"cz", 1},        {"crx", 2},
      {"measure", 1}, {"reset", 1},
  };
  expectCounts(*qc, 6, 8, expected);
  EXPECT_EQ(qc->operationCounts().at("qc.barrier"), 2);
  EXPECT_FALSE(qc->inspect().hasControlFlow);
  auto qco = std::move(*qc).intoQCO();
  ASSERT_TRUE(mlir::succeeded(qco));
  expectCounts(*qco, 6, 8, expected);
  EXPECT_EQ(qco->operationCounts().at("qco.barrier"), 2);
}

TEST(ProgramInspection, CompositeAndUnusedTargetModifiersStayAtomic) {
  auto qc = mlir::QCProgram::fromMLIRString(R"(
    module {
      func.func @main() attributes {mqt.entry_point} {
        %a = qc.alloc : !qc.qubit
        %b = qc.alloc : !qc.qubit
        %c = qc.alloc : !qc.qubit
        %power = arith.constant 0.5 : f64
        qc.ctrl(%a) targets(%t = %b) {
          qc.h %t : !qc.qubit
          qc.x %t : !qc.qubit
          qc.yield
        } : {!qc.qubit}, {!qc.qubit}
        qc.inv (%t = %a) {
          qc.h %t : !qc.qubit
          qc.x %t : !qc.qubit
          qc.yield
        } : !qc.qubit
        qc.pow(%power) (%t = %b) {
          qc.h %t : !qc.qubit
          qc.x %t : !qc.qubit
          qc.yield
        } : !qc.qubit
        qc.ctrl(%a) targets(%x = %b, %unused = %c) {
          qc.x %x : !qc.qubit
          qc.yield
        } : {!qc.qubit}, {!qc.qubit, !qc.qubit}
        qc.dealloc %a : !qc.qubit
        qc.dealloc %b : !qc.qubit
        qc.dealloc %c : !qc.qubit
        return
      }
    }
  )");
  ASSERT_TRUE(mlir::succeeded(qc));
  const std::map<std::string, size_t> expected{
      {"ctrl", 2},
      {"inv", 1},
      {"pow", 1},
  };
  expectCounts(*qc, 2, 1, expected);
  auto qco = std::move(*qc).intoQCO();
  ASSERT_TRUE(mlir::succeeded(qco));
  expectCounts(*qco, 2, 1, expected);
}

TEST(ProgramInspection, QCOCountsNativeThreeQubitGates) {
  auto qco = mlir::QCOProgram::fromMLIRString(R"(
    module {
      func.func @main() attributes {mqt.entry_point} {
        %a = qco.alloc : !qco.qubit
        %b = qco.alloc : !qco.qubit
        %c = qco.alloc : !qco.qubit
        %x, %y, %z = qco.rccx %a, %b, %c : !qco.qubit, !qco.qubit, !qco.qubit
        qco.sink %x : !qco.qubit
        qco.sink %y : !qco.qubit
        qco.sink %z : !qco.qubit
        return
      }
    }
  )");
  ASSERT_TRUE(mlir::succeeded(qco));
  expectCounts(*qco, 0, 0, {{"rccx", 1}});
}

TEST(ProgramInspection, InspectSortedDistinctStaticSites) {
  auto qc = mlir::QCProgram::fromOpenQASMString(R"(
    OPENQASM 3.0; include "stdgates.inc";
    x $5; h $2; z $5;
  )");
  ASSERT_TRUE(mlir::succeeded(qc));
  const std::vector<uint64_t> sites{2, 5};
  EXPECT_EQ(qc->inspect().numQubits, 2);
  EXPECT_EQ(qc->inspect().staticQubits, sites);
  EXPECT_FALSE(qc->inspect().hasControlFlow);
}

TEST(ProgramInspection, UnknownAndEmptyWidthsDiffer) {
  auto empty = mlir::QCProgram::fromMLIRString(
      "module { func.func @main() attributes {mqt.entry_point} { return } }");
  auto library = mlir::QCProgram::fromMLIRString(R"(
    module {
      func.func @helper(%q: !qc.qubit) {
        qc.h %q : !qc.qubit
        return
      }
    }
  )");
  auto dynamic = mlir::QCProgram::fromMLIRString(R"(
    module {
      func.func @main(%n: index) attributes {mqt.entry_point} {
        %q = memref.alloc(%n) : memref<?x!qc.qubit>
        memref.dealloc %q : memref<?x!qc.qubit>
        return
      }
    }
  )");
  ASSERT_TRUE(mlir::succeeded(empty));
  ASSERT_TRUE(mlir::succeeded(library));
  ASSERT_TRUE(mlir::succeeded(dynamic));
  EXPECT_EQ(empty->inspect().numQubits, 0);
  EXPECT_TRUE(empty->gateCounts().empty());
  EXPECT_FALSE(library->inspect().numQubits);
  expectCounts(*library, 0, 0, {});
  EXPECT_FALSE(dynamic->inspect().numQubits);
  auto qco = std::move(*dynamic).intoQCO();
  ASSERT_TRUE(mlir::succeeded(qco));
  EXPECT_FALSE(qco->inspect().numQubits);
}

TEST(ProgramInspection, AllocationsAddWidthButStoredReferencesDoNot) {
  auto qc = mlir::QCProgram::fromMLIRString(R"(
    module {
      func.func @main() attributes {mqt.entry_point} {
        %i = arith.constant 0 : index
        %q = qc.alloc : !qc.qubit
        %register = memref.alloc() : memref<2x!qc.qubit>
        %refs = memref.alloca() : memref<1x!qc.qubit>
        memref.store %q, %refs[%i] : memref<1x!qc.qubit>
        qc.h %q : !qc.qubit
        %alias = memref.load %refs[%i] : memref<1x!qc.qubit>
        qc.x %alias : !qc.qubit
        qc.dealloc %q : !qc.qubit
        memref.dealloc %register : memref<2x!qc.qubit>
        return
      }
    }
  )");
  ASSERT_TRUE(mlir::succeeded(qc));
  EXPECT_EQ(qc->inspect().numQubits, 3);
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
    ASSERT_TRUE(mlir::succeeded(qc));
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
  ASSERT_TRUE(mlir::succeeded(input));
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
  ASSERT_TRUE(mlir::succeeded(branches));
  EXPECT_TRUE(branches->inspect().hasControlFlow);
  EXPECT_EQ(branches->controlFlowCounts(),
            (std::map<std::string, size_t>{{"cf.br", 1}}));
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
  ASSERT_TRUE(mlir::succeeded(qc));
  const auto info = qc->inspect();
  EXPECT_EQ(info.staticQubits, (std::vector<uint64_t>{5}));
  EXPECT_EQ(info.numQubits, 1);
  EXPECT_TRUE(info.hasControlFlow);
  expectCounts(*qc, 0, 0, {});
  EXPECT_TRUE(qc->controlFlowCounts().empty());
  EXPECT_EQ(info.operationCounts, qc->operationCounts());
  EXPECT_EQ(info.operationCounts,
            (std::map<std::string, size_t>{{"builtin.module", 2},
                                           {"func.func", 3},
                                           {"func.return", 3},
                                           {"qc.static", 2},
                                           {"qc.h", 1},
                                           {"qc.x", 1},
                                           {"scf.if", 1},
                                           {"scf.yield", 1}}));
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
  ASSERT_TRUE(mlir::succeeded(qc));
  const std::map<std::string, size_t> expectedCounts{
      {"cx", 2}, {"measure", 1}, {"swap", 1}, {"x", 1}, {"z", 1},
  };
  expectCounts(*qc, 3, 3, expectedCounts);
  const auto qcInfo = qc->inspect();
  EXPECT_EQ(qcInfo.numQubits, 3);
  EXPECT_TRUE(qcInfo.hasControlFlow);
  EXPECT_EQ(qcInfo.controlFlowCounts, qc->controlFlowCounts());
  EXPECT_EQ(qcInfo.controlFlowCounts,
            (std::map<std::string, size_t>{{"scf.if", 1},
                                           {"scf.for", 1},
                                           {"scf.while", 1},
                                           {"scf.index_switch", 1}}));
  auto qco = std::move(*qc).intoQCO();
  ASSERT_TRUE(mlir::succeeded(qco));
  expectCounts(*qco, 3, 3, expectedCounts);
  const auto qcoInfo = qco->inspect();
  EXPECT_EQ(qcoInfo.numQubits, 3);
  EXPECT_TRUE(qcoInfo.hasControlFlow);
  EXPECT_EQ(qcoInfo.controlFlowCounts, qco->controlFlowCounts());
  EXPECT_EQ(qcoInfo.controlFlowCounts,
            (std::map<std::string, size_t>{{"qco.if", 1},
                                           {"scf.for", 1},
                                           {"scf.while", 1},
                                           {"qco.index_switch", 1}}));
}

TEST(ProgramInspection, DepthFollowsQubitDependencies) {
  expectDepthInBothDialects(R"(
    OPENQASM 3.0; include "stdgates.inc";
    qubit[4] q;
    h q[0]; x q[1]; y q[2]; // q0, q1, q2: 1
    h q[0];                 // q0: 2
    cx q[0], q[1];          // q0, q1: 3
    cx q[2], q[3];          // q2, q3: 2
    ccx q[1], q[2], q[3];   // q1, q2, q3: 4
  )",
                            4);
}

TEST(ProgramInspection, DepthWithGateModifiers) {
  expectDepthInBothDialects(R"(
    OPENQASM 3.0; include "stdgates.inc";
    qubit[3] q;
    ctrl @ inv @ x q[0], q[1];
    pow(2) @ h q[2];
    x q[1];
  )",
                            2);
}

TEST(ProgramInspection, DepthWithFunctionCalls) {
  expectDepthInBothDialects(R"(
    OPENQASM 3.0; include "stdgates.inc";
    gate foo a { h a; x a; }
    qubit[3] q;
    foo q[0];
    h q[1];
    x q[1];
    ctrl @ foo q[1], q[2];
  )",
                            std::nullopt);
}

TEST(ProgramInspection, DepthAlignsBarriersWithoutAddingLayers) {
  expectDepthInBothDialects(R"(
    OPENQASM 3.0; include "stdgates.inc";
    qubit[3] q;
    h q[0]; h q[0];     // q0: 2
    barrier q[0], q[1]; // q0, q1: 2
    x q[1];             // q1: 3
    x q[2];             // q2: 1
  )",
                            3);

  expectDepthInBothDialects(R"(
    OPENQASM 3.0; include "stdgates.inc";
    qubit[2] q;
    barrier q[0], q[1];
  )",
                            0);
}

TEST(ProgramInspection, DepthCountsMeasureResetAndIdentityButNotGlobalPhase) {
  expectDepthInBothDialects(R"(
    OPENQASM 3.0; include "stdgates.inc";
    qubit[2] q;
    bit c;
    gphase(0.3);
    id q[0];          // q0: 1
    c = measure q[0]; // q0: 2
    reset q[0];       // q0: 3
    h q[1];           // q1: 1
  )",
                            3);
  expectDepthInBothDialects(R"(
    OPENQASM 3.0; include "stdgates.inc";
    qubit q;
    gphase(0.3);
  )",
                            0);
}

TEST(ProgramInspection, DepthFollowsClassicalFeedForward) {
  expectDepthInBothDialects(R"(
    OPENQASM 3.0; include "stdgates.inc";
    qubit[2] q;
    bit[2] c;
    h q[0];              // q0: 1
    c[0] = measure q[0]; // q0, c0: 2
    if (c[0]) x q[1];    // c0, q1: 3 (waits for the measured bit)
    c[1] = measure q[1]; // q1, c1: 4
  )",
                            4);
}

TEST(ProgramInspection, DepthOrdersMeasurementsIntoTheSameBit) {
  expectDepthInBothDialects(R"(
    OPENQASM 3.0; include "stdgates.inc";
    qubit[2] q;
    bit c;
    h q[0]; h q[0];   // q0: 2
    c = measure q[0]; // q0, c: 3
    c = measure q[1]; // q1, c: 4 (waits for c)
  )",
                            4);
  expectDepthInBothDialects(R"(
    OPENQASM 3.0; include "stdgates.inc";
    qubit[2] q;
    bit[2] c;
    h q[0]; h q[0];      // q0: 2
    c[0] = measure q[0]; // q0, c0: 3
    c[1] = measure q[1]; // q1, c1: 1
  )",
                            3);
}

TEST(ProgramInspection, DepthWithSimpleControlFlow) {
  expectDepthInBothDialects(R"(
    OPENQASM 3.0; include "stdgates.inc";
    qubit[2] q;
    bit c = measure q[0];
    if (c) {
      x q[1];
    }
  )",
                            2);
}

TEST(ProgramInspection, DepthWithBlockControlFlow) {
  expectDepthInBothDialects(R"(
    OPENQASM 3.0; include "stdgates.inc";
    qubit[2] q;
    bit c = measure q[0];
    if (c) {
      x q[1];
      y q[1];
    }
  )",
                            2);

  expectDepthInBothDialects(R"(
    OPENQASM 3.0; include "stdgates.inc";
    qubit[2] q;
    h q[0];
    bit c = measure q[0];
    if (c) {
      x q[1];
      y q[1];
    }
  )",
                            3);
}

TEST(ProgramInspection, DepthCountsBothIfBranchesAsOneLayer) {
  expectDepthInBothDialects(R"(
    OPENQASM 3.0; include "stdgates.inc";
    qubit[3] q;
    bit[2] c;
    h q[0];
    c[0] = measure q[0];
    if (c[0]) {
      x q[1];
    } else {
      x q[2];
    }
    c[1] = measure q[2];
  )",
                            4);
}

TEST(ProgramInspection, DepthWithSwitchControlFlow) {
  expectDepthInBothDialects(R"(OPENQASM 3.0;
    include "stdgates.inc";
    qubit[3] q;
    h q[0];
    int condition = 1;
    switch (condition) {
      case 1 {
        x q[1];
        y q[1];
      }
      default {
        z q[2];
      }
    }
  )",
                            std::nullopt);
}

TEST(ProgramInspection, DepthWithComplexControlFlow) {
  expectDepthInBothDialects(R"(OPENQASM 3.0;
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
    switch (selector) {
      case 1 {
        swap q[0], q[1];
      }
      default {
        z q[2];
      }
    }
  )",
                            std::nullopt);
}

TEST(ProgramInspection, DepthForDynamicLoops) {
  expectDepthInBothDialects(R"(OPENQASM 3.0;
    include "stdgates.inc";
    qubit[3] q;
    bit condition = measure q[0];
    while (condition) {
      ctrl @ x q[0], q[1];
    }
  )",
                            std::nullopt);
}

} // namespace

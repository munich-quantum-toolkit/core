/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/// @file test_qco_context_sensitive_specialization.cpp
/// Tests for the `quantum-context-sensitive-specialization` pass.

#include "mqt/Dialect/QCO/Transforms/Passes.h"

#include "IPOTestFixture.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"

#include <gtest/gtest.h>
#include <numbers>
#include <tuple>

namespace {

using QCOContextSensitiveSpecializationTest = ::mqt::test::IPOTestBase;
using namespace mlir;
using namespace mlir::qco;

// ==========================================================================
// Context-sensitive specialization for arguments in the |0> state.
// ==========================================================================

/// A gate that acts trivially on |0> is dropped from a specialized copy
/// of the callee when the caller passes a freshly allocated qubit.
TEST_F(QCOContextSensitiveSpecializationTest,
       specializeZeroArgumentDropsDiagonalGate) {
  programBuilder.initialize();
  programBuilder.createFunction("f", {getQubitType()}, [&](ValueRange args) {
    return SmallVector<Value>{programBuilder.z(args[0])};
  });

  auto q = programBuilder.allocQubit();
  auto results = callFunction(programBuilder, "f", {q});
  programBuilder.sink(results[0]);
  moduleOp = programBuilder.finalize();

  referenceBuilder.initialize();
  // ... while the call is redirected to a specialization without the gate.
  referenceBuilder.createFunction(
      "f_spec_zero_arg_0", {getQubitType()},
      [&](ValueRange specArgs) { return SmallVector<Value>{specArgs[0]}; });

  auto refQ = referenceBuilder.allocQubit();
  auto refResults = callFunction(referenceBuilder, "f_spec_zero_arg_0", {refQ});
  referenceBuilder.sink(refResults[0]);
  reference = referenceBuilder.finalize();

  expectSingleStageMatchesReference(createContextSensitiveSpecialization());
}

/// A reset applied to an argument that is already in the |0> state is
/// dropped. A reset does not implement the unitary interface, so this exercises
/// a different removal path than the gate case above.
TEST_F(QCOContextSensitiveSpecializationTest,
       specializeZeroArgumentDropsReset) {
  const auto qubitType = getQubitType();

  programBuilder.initialize();
  programBuilder.createFunction("f", {qubitType}, [&](ValueRange args) {
    return SmallVector<Value>{programBuilder.reset(args[0])};
  });

  auto q = programBuilder.allocQubit();
  auto results = callFunction(programBuilder, "f", {q});
  programBuilder.sink(results[0]);
  moduleOp = programBuilder.finalize();

  referenceBuilder.initialize();
  referenceBuilder.createFunction(
      "f_spec_zero_arg_0", {qubitType},
      [&](ValueRange specArgs) { return SmallVector<Value>{specArgs[0]}; });

  auto refQ = referenceBuilder.allocQubit();
  auto refResults = callFunction(referenceBuilder, "f_spec_zero_arg_0", {refQ});
  referenceBuilder.sink(refResults[0]);
  reference = referenceBuilder.finalize();

  expectSingleStageMatchesReference(createContextSensitiveSpecialization());
}

/// A controlled gate whose control is known to be in the |0> state is
/// dropped entirely, together with its effect on the target qubit.
TEST_F(QCOContextSensitiveSpecializationTest,
       specializeZeroArgumentDropsControlledGate) {
  const auto qubitType = getQubitType();

  programBuilder.initialize();
  programBuilder.createFunction(
      "f", {qubitType, qubitType}, [&](ValueRange args) {
        auto [control, target] = programBuilder.cx(args[0], args[1]);
        return SmallVector<Value>{control, target};
      });

  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.h(programBuilder.allocQubit());
  auto results = callFunction(programBuilder, "f", {q0, q1});
  programBuilder.sink(results[0]);
  programBuilder.sink(results[1]);
  moduleOp = programBuilder.finalize();

  referenceBuilder.initialize();

  referenceBuilder.createFunction(
      "f_spec_zero_arg_0", {qubitType, qubitType}, [&](ValueRange specArgs) {
        return SmallVector<Value>{specArgs[0], specArgs[1]};
      });

  auto refQ0 = referenceBuilder.allocQubit();
  auto refQ1 = referenceBuilder.h(referenceBuilder.allocQubit());
  auto refResults =
      callFunction(referenceBuilder, "f_spec_zero_arg_0", {refQ0, refQ1});
  referenceBuilder.sink(refResults[0]);
  referenceBuilder.sink(refResults[1]);
  reference = referenceBuilder.finalize();

  expectSingleStageMatchesReference(createContextSensitiveSpecialization());
}

/// A gate that does not act trivially on |0> must not be dropped.
TEST_F(QCOContextSensitiveSpecializationTest,
       noZeroSpecializationForNonTrivialGate) {
  programBuilder.initialize();
  programBuilder.createFunction("f", {getQubitType()}, [&](ValueRange args) {
    return SmallVector<Value>{programBuilder.x(args[0])};
  });

  auto q = programBuilder.allocQubit();
  auto results = callFunction(programBuilder, "f", {q});
  programBuilder.sink(results[0]);
  moduleOp = programBuilder.finalize();

  referenceBuilder.initialize();
  referenceBuilder.createFunction(
      "f", {getQubitType()}, [&](ValueRange refArgs) {
        return SmallVector<Value>{referenceBuilder.x(refArgs[0])};
      });

  auto refQ = referenceBuilder.allocQubit();
  auto refResults = callFunction(referenceBuilder, "f", {refQ});
  referenceBuilder.sink(refResults[0]);
  reference = referenceBuilder.finalize();

  expectSingleStageMatchesReference(createContextSensitiveSpecialization());
}

/// If the state of the argument is unknown, no specialization applies.
TEST_F(QCOContextSensitiveSpecializationTest,
       noSpecializationForUnknownArgumentState) {
  programBuilder.initialize();
  programBuilder.createFunction("f", {getQubitType()}, [&](ValueRange args) {
    return SmallVector<Value>{programBuilder.z(args[0])};
  });

  // A `y` gate leaves the qubit in a state the pass cannot reason about.
  auto q = programBuilder.y(programBuilder.allocQubit());
  auto results = callFunction(programBuilder, "f", {q});
  programBuilder.sink(results[0]);
  moduleOp = programBuilder.finalize();

  referenceBuilder.initialize();
  referenceBuilder.createFunction(
      "f", {getQubitType()}, [&](ValueRange refArgs) {
        return SmallVector<Value>{referenceBuilder.z(refArgs[0])};
      });

  auto refQ = referenceBuilder.y(referenceBuilder.allocQubit());
  auto refResults = callFunction(referenceBuilder, "f", {refQ});
  referenceBuilder.sink(refResults[0]);
  reference = referenceBuilder.finalize();

  expectSingleStageMatchesReference(createContextSensitiveSpecialization());
}

/// Two call sites that qualify for the same specialization share a single
/// specialized copy of the callee.
TEST_F(QCOContextSensitiveSpecializationTest,
       reuseZeroSpecializationAcrossCallSites) {
  programBuilder.initialize();
  programBuilder.createFunction("f", {getQubitType()}, [&](ValueRange args) {
    return SmallVector<Value>{programBuilder.s(args[0])};
  });

  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.allocQubit();
  auto results0 = callFunction(programBuilder, "f", {q0});
  auto results1 = callFunction(programBuilder, "f", {q1});
  programBuilder.sink(results0[0]);
  programBuilder.sink(results1[0]);
  moduleOp = programBuilder.finalize();

  referenceBuilder.initialize();

  referenceBuilder.createFunction(
      "f_spec_zero_arg_0", {getQubitType()},
      [&](ValueRange specArgs) { return SmallVector<Value>{specArgs[0]}; });

  auto refQ0 = referenceBuilder.allocQubit();
  auto refQ1 = referenceBuilder.allocQubit();
  auto refResults0 =
      callFunction(referenceBuilder, "f_spec_zero_arg_0", {refQ0});
  auto refResults1 =
      callFunction(referenceBuilder, "f_spec_zero_arg_0", {refQ1});
  referenceBuilder.sink(refResults0[0]);
  referenceBuilder.sink(refResults1[0]);
  reference = referenceBuilder.finalize();

  expectSingleStageMatchesReference(createContextSensitiveSpecialization());
}

/// A run of operations that all leave |0> alone is dropped in one go.
TEST_F(QCOContextSensitiveSpecializationTest,
       specializeZeroArgumentDropsNoOpRun) {
  const auto qubitType = getQubitType();

  programBuilder.initialize();
  programBuilder.createFunction("f", {qubitType}, [&](ValueRange args) {
    return SmallVector<Value>{
        programBuilder.t(programBuilder.s(programBuilder.z(args[0]))),
    };
  });

  auto q = programBuilder.allocQubit();
  auto results = callFunction(programBuilder, "f", {q});
  programBuilder.sink(results[0]);
  moduleOp = programBuilder.finalize();

  referenceBuilder.initialize();
  referenceBuilder.createFunction(
      "f_spec_zero_arg_0", {qubitType},
      [&](ValueRange specArgs) { return SmallVector<Value>{specArgs[0]}; });

  auto refQ = referenceBuilder.allocQubit();
  auto refResults = callFunction(referenceBuilder, "f_spec_zero_arg_0", {refQ});
  referenceBuilder.sink(refResults[0]);
  reference = referenceBuilder.finalize();

  expectSingleStageMatchesReference(createContextSensitiveSpecialization());
}

/// A phase gate fixes |0> for any angle and is dropped even when the
/// angle is only known at run time.
TEST_F(QCOContextSensitiveSpecializationTest,
       specializeZeroArgumentDropsPhaseGate) {
  const auto qubitType = getQubitType();
  const auto floatType = programBuilder.getF64Type();

  programBuilder.initialize();
  programBuilder.createFunction(
      "f", {qubitType, floatType}, [&](ValueRange args) {
        return SmallVector<Value>{programBuilder.p(args[1], args[0])};
      });

  auto q = programBuilder.allocQubit();
  // An angle outside the specialized set, so only the |0> specialization fires.
  auto angle = programBuilder.floatConstant(0.7);
  auto results = callFunction(programBuilder, "f", {q, angle});
  programBuilder.sink(results[0]);
  moduleOp = programBuilder.finalize();

  referenceBuilder.initialize();
  // The angle stays in the signature, it is simply no longer read.
  referenceBuilder.createFunction(
      "f_spec_zero_arg_0", {qubitType, floatType},
      [&](ValueRange specArgs) { return SmallVector<Value>{specArgs[0]}; });

  auto refQ = referenceBuilder.allocQubit();
  auto refAngle = referenceBuilder.floatConstant(0.7);
  auto refResults =
      callFunction(referenceBuilder, "f_spec_zero_arg_0", {refQ, refAngle});
  referenceBuilder.sink(refResults[0]);
  reference = referenceBuilder.finalize();

  expectSingleStageMatchesReference(createContextSensitiveSpecialization());
}

/// A z-rotation must not be dropped: it maps |0> to a phase multiple of
/// itself, so removing it would change the global phase of the program.
TEST_F(QCOContextSensitiveSpecializationTest,
       noZeroSpecializationForZRotation) {
  const auto qubitType = getQubitType();
  const auto floatType = programBuilder.getF64Type();

  const auto buildProgram = [&](QCOProgramBuilder& b) {
    b.initialize();
    b.createFunction("f", {qubitType, floatType}, [&](ValueRange args) {
      return SmallVector<Value>{b.rz(args[1], args[0])};
    });

    auto q = b.allocQubit();
    auto results = callFunction(b, "f", {q, b.floatConstant(0.7)});
    b.sink(results[0]);
  };

  buildProgram(programBuilder);
  moduleOp = programBuilder.finalize();
  buildProgram(referenceBuilder);
  reference = referenceBuilder.finalize();

  expectSingleStageMatchesReference(createContextSensitiveSpecialization());
}

/// A specialization of a public callee is private.
///
/// Cloning carries visibility over, so this used to export the generated symbol
/// and, since orphan cleanup skips public functions, never reclaim it.
TEST_F(QCOContextSensitiveSpecializationTest,
       specializationOfPublicCalleeIsPrivate) {
  auto module = parseModule(R"mlir(
func.func @callee(%q: !qco.qubit) -> !qco.qubit {
  %0 = qco.z %q : !qco.qubit -> !qco.qubit
  return %0 : !qco.qubit
}
func.func @main() {
  %q = qco.alloc : !qco.qubit
  %r = func.call @callee(%q) : (!qco.qubit) -> !qco.qubit
  qco.sink %r : !qco.qubit
  return
}
)mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(runStage(module.get(), createContextSensitiveSpecialization())
                  .succeeded());

  auto specializations = 0;
  module->walk([&](func::FuncOp func) {
    if (func.getName() == "callee" || func.getName() == "main") {
      return;
    }
    ++specializations;
    EXPECT_TRUE(func.isPrivate())
        << "specialization " << func.getName().str() << " must not be exported";
  });
  EXPECT_EQ(specializations, 1) << "the public callee should be specialized";
}

// ==========================================================================
// Context-sensitive specialization for arguments in the |+> state.
// ==========================================================================

/// An `x` gate acting on a qubit known to be in the |+> state is dropped
/// from a specialized copy of the callee.
TEST_F(QCOContextSensitiveSpecializationTest,
       specializePlusArgumentDropsXGate) {
  programBuilder.initialize();
  programBuilder.createFunction("f", {getQubitType()}, [&](ValueRange args) {
    return SmallVector<Value>{programBuilder.x(args[0])};
  });

  auto q = programBuilder.h(programBuilder.allocQubit());
  auto results = callFunction(programBuilder, "f", {q});
  programBuilder.sink(results[0]);
  moduleOp = programBuilder.finalize();

  referenceBuilder.initialize();

  referenceBuilder.createFunction(
      "f_spec_plus_arg_0", {getQubitType()},
      [&](ValueRange specArgs) { return SmallVector<Value>{specArgs[0]}; });

  auto refQ = referenceBuilder.h(referenceBuilder.allocQubit());
  auto refResults = callFunction(referenceBuilder, "f_spec_plus_arg_0", {refQ});
  referenceBuilder.sink(refResults[0]);
  reference = referenceBuilder.finalize();

  expectSingleStageMatchesReference(createContextSensitiveSpecialization());
}

/// A gate that does not act trivially on |+> must not be dropped.
TEST_F(QCOContextSensitiveSpecializationTest, noPlusSpecializationForNonXGate) {
  programBuilder.initialize();
  programBuilder.createFunction("f", {getQubitType()}, [&](ValueRange args) {
    return SmallVector<Value>{programBuilder.z(args[0])};
  });

  auto q = programBuilder.h(programBuilder.allocQubit());
  auto results = callFunction(programBuilder, "f", {q});
  programBuilder.sink(results[0]);
  moduleOp = programBuilder.finalize();

  referenceBuilder.initialize();
  referenceBuilder.createFunction(
      "f", {getQubitType()}, [&](ValueRange refArgs) {
        return SmallVector<Value>{referenceBuilder.z(refArgs[0])};
      });

  auto refQ = referenceBuilder.h(referenceBuilder.allocQubit());
  auto refResults = callFunction(referenceBuilder, "f", {refQ});
  referenceBuilder.sink(refResults[0]);
  reference = referenceBuilder.finalize();

  expectSingleStageMatchesReference(createContextSensitiveSpecialization());
}

// ==========================================================================
// Context-sensitive specialization for constant rotation angles.
// ==========================================================================

/// A rotation angle of pi passed at the call site is baked into a
/// specialized copy of the callee.
TEST_F(QCOContextSensitiveSpecializationTest, specializeConstantRotationAngle) {
  const auto qubitType = getQubitType();
  const auto floatType = programBuilder.getF64Type();

  programBuilder.initialize();
  programBuilder.createFunction(
      "f", {qubitType, floatType}, [&](ValueRange args) {
        return SmallVector<Value>{programBuilder.rz(args[1], args[0])};
      });

  auto q = programBuilder.allocQubit();
  auto angle = programBuilder.floatConstant(std::numbers::pi);
  auto results = callFunction(programBuilder, "f", {q, angle});
  programBuilder.sink(results[0]);
  moduleOp = programBuilder.finalize();

  referenceBuilder.initialize();

  // The specialized copy keeps the parameter in its signature but no longer
  // reads it; the angle becomes a constant in the body.
  referenceBuilder.createFunction(
      "f_spec_fixed_angle_1", {qubitType, floatType}, [&](ValueRange specArgs) {
        return SmallVector<Value>{
            referenceBuilder.rz(std::numbers::pi, specArgs[0]),
        };
      });

  auto refQ = referenceBuilder.allocQubit();
  auto refAngle = referenceBuilder.floatConstant(std::numbers::pi);
  auto refResults =
      callFunction(referenceBuilder, "f_spec_fixed_angle_1", {refQ, refAngle});
  referenceBuilder.sink(refResults[0]);
  reference = referenceBuilder.finalize();

  expectSingleStageMatchesReference(createContextSensitiveSpecialization());
}

/// A rotation angle of pi/2 is likewise specialized.
TEST_F(QCOContextSensitiveSpecializationTest, specializeHalfPiRotationAngle) {
  const auto qubitType = getQubitType();
  const auto floatType = programBuilder.getF64Type();

  programBuilder.initialize();
  programBuilder.createFunction(
      "f", {qubitType, floatType}, [&](ValueRange args) {
        return SmallVector<Value>{programBuilder.rx(args[1], args[0])};
      });

  auto q = programBuilder.allocQubit();
  auto angle = programBuilder.floatConstant(std::numbers::pi / 2);
  auto results = callFunction(programBuilder, "f", {q, angle});
  programBuilder.sink(results[0]);
  moduleOp = programBuilder.finalize();

  referenceBuilder.initialize();

  referenceBuilder.createFunction(
      "f_spec_fixed_angle_1", {qubitType, floatType}, [&](ValueRange specArgs) {
        return SmallVector<Value>{
            referenceBuilder.rx(std::numbers::pi / 2, specArgs[0]),
        };
      });

  auto refQ = referenceBuilder.allocQubit();
  auto refAngle = referenceBuilder.floatConstant(std::numbers::pi / 2);
  auto refResults =
      callFunction(referenceBuilder, "f_spec_fixed_angle_1", {refQ, refAngle});
  referenceBuilder.sink(refResults[0]);
  reference = referenceBuilder.finalize();

  expectSingleStageMatchesReference(createContextSensitiveSpecialization());
}

/// Two call sites passing different constant angles to the same callee
/// each get their own specialization.
TEST_F(QCOContextSensitiveSpecializationTest,
       separateRotationSpecializationPerAngle) {
  const auto qubitType = getQubitType();
  const auto floatType = programBuilder.getF64Type();

  programBuilder.initialize();
  programBuilder.createFunction(
      "f", {qubitType, floatType}, [&](ValueRange args) {
        return SmallVector<Value>{programBuilder.rz(args[1], args[0])};
      });

  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.allocQubit();
  auto results0 =
      callFunction(programBuilder, "f",
                   {q0, programBuilder.floatConstant(std::numbers::pi)});
  auto results1 =
      callFunction(programBuilder, "f",
                   {q1, programBuilder.floatConstant(std::numbers::pi / 2)});
  programBuilder.sink(results0[0]);
  programBuilder.sink(results1[0]);
  moduleOp = programBuilder.finalize();

  referenceBuilder.initialize();

  // The rewriter reaches the pi/2 call first, so that specialization keeps the
  // plain name and the pi one gets the uniqued name.
  referenceBuilder.createFunction(
      "f_spec_fixed_angle_1", {qubitType, floatType},
      [&](ValueRange specHalfPiArgs) {
        return SmallVector<Value>{
            referenceBuilder.rz(std::numbers::pi / 2, specHalfPiArgs[0]),
        };
      });

  referenceBuilder.createFunction(
      "f_spec_fixed_angle_1_0", {qubitType, floatType},
      [&](ValueRange specPiArgs) {
        return SmallVector<Value>{
            referenceBuilder.rz(std::numbers::pi, specPiArgs[0]),
        };
      });

  auto refQ0 = referenceBuilder.allocQubit();
  auto refQ1 = referenceBuilder.allocQubit();
  auto refResults0 =
      callFunction(referenceBuilder, "f_spec_fixed_angle_1_0",
                   {refQ0, referenceBuilder.floatConstant(std::numbers::pi)});
  auto refResults1 = callFunction(
      referenceBuilder, "f_spec_fixed_angle_1",
      {refQ1, referenceBuilder.floatConstant(std::numbers::pi / 2)});
  referenceBuilder.sink(refResults0[0]);
  referenceBuilder.sink(refResults1[0]);
  reference = referenceBuilder.finalize();

  expectSingleStageMatchesReference(createContextSensitiveSpecialization());
}

/// An angle the callee never reads is not specialized, because the copy
/// would be identical to the callee it was cloned from.
TEST_F(QCOContextSensitiveSpecializationTest,
       noSpecializationForUnusedRotationAngle) {
  const auto qubitType = getQubitType();
  const auto floatType = programBuilder.getF64Type();

  const auto buildProgram = [&](QCOProgramBuilder& b) {
    b.initialize();
    // The angle is part of the signature but nothing in the body uses it.
    b.createFunction("f", {qubitType, floatType}, [&](ValueRange args) {
      return SmallVector<Value>{b.h(args[0])};
    });

    auto q = b.allocQubit();
    auto results = callFunction(b, "f", {q, b.floatConstant(std::numbers::pi)});
    b.sink(results[0]);
  };

  buildProgram(programBuilder);
  moduleOp = programBuilder.finalize();
  buildProgram(referenceBuilder);
  reference = referenceBuilder.finalize();

  expectSingleStageMatchesReference(createContextSensitiveSpecialization());
}

/// An angle outside the set of specialized angles leaves the callee
/// untouched.
TEST_F(QCOContextSensitiveSpecializationTest,
       noSpecializationForArbitraryRotationAngle) {
  const auto qubitType = getQubitType();
  const auto floatType = programBuilder.getF64Type();

  programBuilder.initialize();
  programBuilder.createFunction(
      "f", {qubitType, floatType}, [&](ValueRange args) {
        return SmallVector<Value>{programBuilder.rz(args[1], args[0])};
      });

  auto q = programBuilder.allocQubit();
  auto angle = programBuilder.floatConstant(0.7);
  auto results = callFunction(programBuilder, "f", {q, angle});
  programBuilder.sink(results[0]);
  moduleOp = programBuilder.finalize();

  referenceBuilder.initialize();
  referenceBuilder.createFunction(
      "f", {qubitType, floatType}, [&](ValueRange refArgs) {
        return SmallVector<Value>{referenceBuilder.rz(refArgs[1], refArgs[0])};
      });

  auto refQ = referenceBuilder.allocQubit();
  auto refAngle = referenceBuilder.floatConstant(0.7);
  auto refResults = callFunction(referenceBuilder, "f", {refQ, refAngle});
  referenceBuilder.sink(refResults[0]);
  reference = referenceBuilder.finalize();

  expectSingleStageMatchesReference(createContextSensitiveSpecialization());
}

/// Two call sites that qualify for the same |+> specialization share a
/// single specialized copy of the callee.
TEST_F(QCOContextSensitiveSpecializationTest,
       reusePlusSpecializationAcrossCallSites) {
  const auto qubitType = getQubitType();

  programBuilder.initialize();
  programBuilder.createFunction("f", {qubitType}, [&](ValueRange args) {
    return SmallVector<Value>{programBuilder.x(args[0])};
  });

  auto q0 = programBuilder.h(programBuilder.allocQubit());
  auto q1 = programBuilder.h(programBuilder.allocQubit());
  auto results0 = callFunction(programBuilder, "f", {q0});
  auto results1 = callFunction(programBuilder, "f", {q1});
  programBuilder.sink(results0[0]);
  programBuilder.sink(results1[0]);
  moduleOp = programBuilder.finalize();

  referenceBuilder.initialize();
  referenceBuilder.createFunction(
      "f_spec_plus_arg_0", {qubitType},
      [&](ValueRange specArgs) { return SmallVector<Value>{specArgs[0]}; });

  auto refQ0 = referenceBuilder.h(referenceBuilder.allocQubit());
  auto refQ1 = referenceBuilder.h(referenceBuilder.allocQubit());
  auto refResults0 =
      callFunction(referenceBuilder, "f_spec_plus_arg_0", {refQ0});
  auto refResults1 =
      callFunction(referenceBuilder, "f_spec_plus_arg_0", {refQ1});
  referenceBuilder.sink(refResults0[0]);
  referenceBuilder.sink(refResults1[0]);
  reference = referenceBuilder.finalize();

  expectSingleStageMatchesReference(createContextSensitiveSpecialization());
}

/// Two call sites passing the same constant angle share a single
/// specialized copy of the callee.
TEST_F(QCOContextSensitiveSpecializationTest,
       reuseRotationSpecializationAcrossCallSites) {
  const auto qubitType = getQubitType();
  const auto floatType = programBuilder.getF64Type();

  programBuilder.initialize();
  programBuilder.createFunction(
      "f", {qubitType, floatType}, [&](ValueRange args) {
        return SmallVector<Value>{programBuilder.rz(args[1], args[0])};
      });

  auto q0 = programBuilder.allocQubit();
  auto q1 = programBuilder.allocQubit();
  auto angle = programBuilder.floatConstant(std::numbers::pi);
  auto results0 = callFunction(programBuilder, "f", {q0, angle});
  auto results1 = callFunction(programBuilder, "f", {q1, angle});
  programBuilder.sink(results0[0]);
  programBuilder.sink(results1[0]);
  moduleOp = programBuilder.finalize();

  referenceBuilder.initialize();
  referenceBuilder.createFunction(
      "f_spec_fixed_angle_1", {qubitType, floatType}, [&](ValueRange specArgs) {
        return SmallVector<Value>{
            referenceBuilder.rz(std::numbers::pi, specArgs[0]),
        };
      });

  auto refQ0 = referenceBuilder.allocQubit();
  auto refQ1 = referenceBuilder.allocQubit();
  auto refAngle = referenceBuilder.floatConstant(std::numbers::pi);
  auto refResults0 =
      callFunction(referenceBuilder, "f_spec_fixed_angle_1", {refQ0, refAngle});
  auto refResults1 =
      callFunction(referenceBuilder, "f_spec_fixed_angle_1", {refQ1, refAngle});
  referenceBuilder.sink(refResults0[0]);
  referenceBuilder.sink(refResults1[0]);
  reference = referenceBuilder.finalize();

  expectSingleStageMatchesReference(createContextSensitiveSpecialization());
}

// ==========================================================================

} // namespace

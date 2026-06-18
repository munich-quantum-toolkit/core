/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

// Custom native-gate menus, randomized equivalence, and IBM-fractional stress
// circuits for the native-gate synthesis pass.

#include "native_synthesis_pass_test_fixture.h"
#include "native_synthesis_test_helpers.h"
#include "qc_programs.h"

#include <gtest/gtest.h>
#include <llvm/Support/Casting.h>
#include <mlir/Conversion/QCToQCO/QCToQCO.h>
#include <mlir/Dialect/QC/Builder/QCProgramBuilder.h>
#include <mlir/Dialect/QCO/IR/QCOInterfaces.h>
#include <mlir/Dialect/QCO/IR/QCOOps.h>
#include <mlir/Dialect/QCO/Transforms/Passes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>

#include <cctype>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <random>
#include <string>
#include <utility>
#include <vector>

using namespace mlir;
using namespace mlir::qco;
using namespace mlir::qco::native_synth_test;

namespace {

struct CustomMenuSpec {
  std::string menuCsv;
  bool allowCx = false;
  bool allowCz = false;
  bool allowU = false;
  bool allowX = false;
  bool allowSX = false;
  bool allowRZ = false;
  bool allowRX = false;
  bool allowRY = false;
  bool allowR = false;
  bool allowRzz = false;
};

} // namespace

static std::vector<std::string> splitCSV(const std::string& s) {
  std::vector<std::string> out;
  std::size_t tokenStart = 0;
  while (tokenStart <= s.size()) {
    const auto tokenEnd = s.find(',', tokenStart);
    const auto end = (tokenEnd == std::string::npos) ? s.size() : tokenEnd;
    std::size_t left = tokenStart;
    while (left < end &&
           std::isspace(static_cast<unsigned char>(s[left])) != 0) {
      ++left;
    }
    std::size_t right = end;
    while (right > left &&
           std::isspace(static_cast<unsigned char>(s[right - 1])) != 0) {
      --right;
    }
    if (left < right) {
      std::string token = s.substr(left, right - left);
      for (char& ch : token) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
      }
      out.push_back(std::move(token));
    }
    if (tokenEnd == std::string::npos) {
      break;
    }
    tokenStart = tokenEnd + 1;
  }
  return out;
}

static CustomMenuSpec parseCustomMenu(const std::string& csv) {
  CustomMenuSpec spec;
  spec.menuCsv = csv;
  for (const auto& tok : splitCSV(csv)) {
    if (tok == "u") {
      spec.allowU = true;
    } else if (tok == "x") {
      spec.allowX = true;
    } else if (tok == "sx") {
      spec.allowSX = true;
    } else if (tok == "rz" || tok == "p") {
      // ``p`` is an alias for Z-axis rotation in ``native-gates`` (see pass
      // docs).
      spec.allowRZ = true;
    } else if (tok == "rx") {
      spec.allowRX = true;
    } else if (tok == "ry") {
      spec.allowRY = true;
    } else if (tok == "r") {
      spec.allowR = true;
    } else if (tok == "cx") {
      spec.allowCx = true;
    } else if (tok == "cz") {
      spec.allowCz = true;
    } else if (tok == "rzz") {
      spec.allowRzz = true;
    }
  }
  return spec;
}

static bool onlyAllowsMenuNativeOps(ModuleOp moduleOp,
                                    const CustomMenuSpec& spec) {
  bool ok = true;
  moduleOp.walk([&](Operation* op) {
    if (!ok) {
      return;
    }
    if (!llvm::isa<qco::UnitaryOpInterface>(op)) {
      return;
    }
    // Non-synthesized helper ops are allowed to remain.
    if (llvm::isa<qco::BarrierOp, qco::GPhaseOp>(op)) {
      return;
    }
    if (llvm::isa<qco::IdOp>(op)) {
      return;
    }

    // Treat `p` as a phase/Z-rotation alias when `rz` is allowed.
    if (llvm::isa<qco::POp>(op)) {
      ok = spec.allowRZ;
      return;
    }

    if (llvm::isa<qco::UOp>(op)) {
      ok = spec.allowU;
      return;
    }
    if (llvm::isa<qco::XOp>(op)) {
      // `cx` is represented as a `qco.ctrl` with a `qco.x` in the body region.
      if (llvm::isa_and_present<qco::CtrlOp>(op->getParentOp())) {
        ok = spec.allowCx;
      } else {
        ok = spec.allowX;
      }
      return;
    }
    if (llvm::isa<qco::SXOp>(op)) {
      ok = spec.allowSX;
      return;
    }
    if (llvm::isa<qco::RZOp>(op)) {
      ok = spec.allowRZ;
      return;
    }
    if (llvm::isa<qco::RXOp>(op)) {
      if (spec.allowRX) {
        ok = true;
        return;
      }
      // If `rx` is not native, only the `rx(±pi)` case is accepted as an
      // X-equivalent under the IBM-basic family fallback.
      if (!(spec.allowX && spec.allowSX && spec.allowRZ)) {
        ok = false;
        return;
      }
      auto rx = llvm::cast<qco::RXOp>(op);
      const auto theta = evaluateConstF64(rx.getTheta());
      if (!theta.has_value()) {
        ok = false;
        return;
      }
      const double rem = std::remainder(*theta, 2.0 * std::numbers::pi);
      ok = std::abs(std::abs(rem) - std::numbers::pi) <= 1e-10;
      return;
    }
    if (llvm::isa<qco::RYOp>(op)) {
      ok = spec.allowRY;
      return;
    }
    if (llvm::isa<qco::ZOp>(op)) {
      // `cz` is represented as a `qco.ctrl` with a `qco.z` in the body region.
      if (llvm::isa_and_present<qco::CtrlOp>(op->getParentOp())) {
        ok = spec.allowCz;
      } else {
        ok = false;
      }
      return;
    }
    if (llvm::isa<qco::ROp>(op)) {
      ok = spec.allowR;
      return;
    }
    if (llvm::isa<qco::RZZOp>(op)) {
      ok = spec.allowRzz;
      return;
    }
    if (auto ctrl = llvm::dyn_cast<qco::CtrlOp>(op)) {
      if (ctrl.getNumControls() != 1 || ctrl.getNumTargets() != 1) {
        ok = false;
        return;
      }
      Operation* body = ctrl.getBodyUnitary(0).getOperation();
      if (llvm::isa<qco::XOp>(body)) {
        ok = spec.allowCx;
        return;
      }
      if (llvm::isa<qco::ZOp>(body)) {
        ok = spec.allowCz;
        return;
      }
      ok = false;
      return;
    }
    ok = false;
  });
  return ok;
}

TEST_F(NativeSynthesisPassTest, RandomizedCustomMenusAndCircuitsAreEquivalent) {
  // Sample many valid custom menus and generate matching random input circuits.
  // For each case, we assert that native synthesis (a) succeeds, (b) emits only
  // ops allowed by the menu, and (c) preserves the 2-qubit unitary up to global
  // phase.
  std::mt19937 rng(0xC0FFEE);
  std::uniform_real_distribution<double> angle(-1.0, 1.0);
  std::uniform_int_distribution<int> stepsDist(4, 14);
  std::uniform_int_distribution<int> gateDist(0, 9);
  std::uniform_int_distribution<int> whichQubit(0, 1);

  // Menus are chosen from known-valid families that the pass supports.
  const std::vector<std::string> menuPool = {
      "u,cx",     "u,cz", "x,sx,rz,rx,cx", "rx,rz,cx",   "rx,ry,cx",
      "ry,rz,cz", "r,cz", "u,rx,rz,cx,cz", "u,rx,rz,cx",
  };
  std::uniform_int_distribution<std::size_t> menuDist(0, menuPool.size() - 1);

  constexpr int numCases = 18;
  for (int caseIdx = 0; caseIdx < numCases; ++caseIdx) {
    const std::string& menuCsv = menuPool[menuDist(rng)];
    const auto menuSpec = parseCustomMenu(menuCsv);

    // Build an input circuit that uses only two qubits and (if present) only
    // the entangler types allowed by the menu. Use a mix of operations that the
    // pass is expected to rewrite into the menu.
    auto buildCircuit = [&]() {
      mlir::qc::QCProgramBuilder builder(context.get());
      builder.initialize();
      const auto q0 = builder.allocQubit();
      const auto q1 = builder.allocQubit();

      const int steps = stepsDist(rng);
      for (int i = 0; i < steps; ++i) {
        const auto q = (whichQubit(rng) == 0) ? q0 : q1;

        // Choose operations based on the menu family to avoid generating inputs
        // that are not exactly synthesizable with the configured gateset.
        if (menuSpec.allowU) {
          // Keep input gates within the robust unitary evaluator set.
          switch (gateDist(rng) % 5) {
          case 0:
            builder.rz(angle(rng), q);
            break;
          case 1:
            builder.rx(angle(rng), q);
            break;
          case 2:
            builder.ry(angle(rng), q);
            break;
          case 3:
            builder.p(angle(rng), q);
            break;
          case 4:
            if (menuSpec.allowCz) {
              builder.cz(q0, q1);
            } else if (menuSpec.allowCx) {
              builder.cx(q0, q1);
            } else {
              builder.rz(angle(rng), q);
            }
            break;
          default:
            break;
          }
        } else if (menuSpec.allowR && menuSpec.allowCz && !menuSpec.allowCx) {
          // Minimal r/cz menu: generate only operations directly expressible in
          // that gateset so synthesis is required to succeed.
          switch (gateDist(rng) % 4) {
          case 0:
            builder.r(angle(rng), angle(rng), q);
            break;
          case 1:
            // X/Y-like rotations expressed via r(theta, phi).
            builder.r(std::numbers::pi, angle(rng), q);
            break;
          case 2:
            builder.r(angle(rng), angle(rng), q);
            break;
          case 3:
            builder.cz(q0, q1);
            break;
          default:
            break;
          }
        } else if (menuSpec.allowRX && menuSpec.allowRY && menuSpec.allowCx &&
                   !menuSpec.allowRZ) {
          // Axis-pair RX/RY with CX: avoid Z-axis primitives.
          switch (gateDist(rng) % 6) {
          case 0:
            builder.rx(angle(rng), q);
            break;
          case 1:
            builder.ry(angle(rng), q);
            break;
          case 2:
            builder.rx(std::numbers::pi, q);
            break;
          case 3:
            builder.ry(std::numbers::pi, q);
            break;
          case 4:
            builder.ry(angle(rng), q);
            break;
          case 5:
            builder.cx(q0, q1);
            break;
          default:
            break;
          }
        } else if (menuSpec.allowRX && menuSpec.allowRZ && menuSpec.allowCx) {
          // Axis-pair RX/RZ with CX.
          switch (gateDist(rng) % 6) {
          case 0:
            builder.rx(angle(rng), q);
            break;
          case 1:
            builder.rz(angle(rng), q);
            break;
          case 2:
            builder.rx(std::numbers::pi, q);
            break;
          case 3:
            builder.rz(std::numbers::pi, q);
            break;
          case 4:
            builder.rz(angle(rng), q);
            break;
          case 5:
            builder.cx(q0, q1);
            break;
          default:
            break;
          }
        } else if (menuSpec.allowRY && menuSpec.allowRZ && menuSpec.allowCz) {
          // Axis-pair RY/RZ with CZ.
          switch (gateDist(rng) % 6) {
          case 0:
            builder.ry(angle(rng), q);
            break;
          case 1:
            builder.rz(angle(rng), q);
            break;
          case 2:
            builder.ry(std::numbers::pi, q);
            break;
          case 3:
            builder.rz(std::numbers::pi, q);
            break;
          case 4:
            builder.rz(angle(rng), q);
            break;
          case 5:
            builder.cz(q0, q1);
            break;
          default:
            break;
          }
        } else {
          // IBM-basic-ish menus (x,sx,rz[,rx],cx): use Z/SX patterns + CX.
          switch (gateDist(rng) % 7) {
          case 0:
            builder.rz(angle(rng), q);
            break;
          case 1:
            builder.p(angle(rng), q);
            break;
          case 2:
            builder.rz(angle(rng), q);
            break;
          case 3:
            builder.rx(menuSpec.allowRX ? angle(rng) : std::numbers::pi, q);
            break;
          case 4:
            builder.rz(angle(rng), q);
            break;
          case 5:
            if (menuSpec.allowRX) {
              builder.rx(angle(rng), q);
            } else {
              builder.p(angle(rng), q);
            }
            break;
          case 6:
            if (menuSpec.allowCx) {
              builder.cx(q0, q1);
            } else {
              builder.rz(angle(rng), q);
            }
            break;
          default:
            break;
          }
        }
      }

      builder.dealloc(q0);
      builder.dealloc(q1);
      return builder.finalize();
    };

    // Build the random circuit exactly once, then clone it for the expected and
    // synthesized paths so the unitary comparison is meaningful.
    auto input = buildCircuit();
    const auto inputText = moduleToString(input);

    auto expected =
        mlir::parseSourceString<mlir::ModuleOp>(inputText, context.get());
    ASSERT_TRUE(expected) << "case=" << caseIdx;
    runQcToQco(expected);
    const auto expectedUnitary = computeTwoQubitUnitaryFromModule(expected);
    if (!expectedUnitary.has_value()) {
      ADD_FAILURE() << "Failed to reconstruct expected unitary for case="
                    << caseIdx << " menu=" << menuCsv << "\nIR:\n"
                    << moduleToString(expected);
      continue;
    }

    auto synthesized =
        mlir::parseSourceString<mlir::ModuleOp>(inputText, context.get());
    ASSERT_TRUE(synthesized) << "case=" << caseIdx;
    {
      PassManager pm(synthesized->getContext());
      pm.addPass(createQCToQCO());
      pm.addPass(
          qco::createNativeGateSynthesisPass(qco::NativeGateSynthesisOptions{
              .nativeGates = menuCsv,
          }));
      if (failed(pm.run(*synthesized))) {
        ADD_FAILURE() << "Native synthesis failed for menu=" << menuCsv
                      << " case=" << caseIdx << "\nQC/QCO IR:\n"
                      << moduleToString(synthesized);
        continue;
      }
    }

    EXPECT_TRUE(onlyAllowsMenuNativeOps(synthesized.get(), menuSpec))
        << "menu=" << menuCsv << "\nIR:\n"
        << moduleToString(synthesized);

    const auto synthesizedUnitary =
        computeTwoQubitUnitaryFromModule(synthesized);
    ASSERT_TRUE(synthesizedUnitary.has_value()) << "case=" << caseIdx;
    EXPECT_TRUE(
        isEquivalentUpToGlobalPhase(*expectedUnitary, *synthesizedUnitary))
        << "menu=" << menuCsv << " case=" << caseIdx;
  }
}

TEST_F(NativeSynthesisPassTest,
       LargeCircuitEquivalentAndNativeGatesIbmFractional) {
  auto buildStressCircuit = [&](MLIRContext* ctx) {
    return mlir::qc::QCProgramBuilder::build(
        ctx, mlir::qc::nativeSynthCustomMenusIbmFractionalTwoQStress);
  };
  expectEquivalentAndNativeAfterSynthesis(
      [&] { return buildStressCircuit(context.get()); }, "x,sx,rz,rx,rzz,cz",
      &NativeSynthesisPassTest::onlyIbmFractionalOps,
      computeTwoQubitUnitaryFromModule);
}

TEST_F(NativeSynthesisPassTest,
       AllGateFamiliesEquivalentAndNativeIbmFractional) {
  expectEquivalentAndNativeAfterSynthesis(
      [&] { return buildIbmFractionalAllGateFamiliesCircuit(); },
      "x,sx,rz,rx,rzz,cz", &NativeSynthesisPassTest::onlyIbmFractionalOps,
      computeTwoQubitUnitaryFromModule);
}

TEST_F(NativeSynthesisPassTest, XXPlusMinusYYEquivalentAndNativeIbmFractional) {
  constexpr const char* kIbmFrac = "x,sx,rz,rx,rzz,cz";
  expectEquivalentAndNativeAfterSynthesis(
      [&] {
        return mlir::qc::QCProgramBuilder::build(
            context.get(), mlir::qc::nativeSynthCustomMenusXxPlusYyChain);
      },
      kIbmFrac, &NativeSynthesisPassTest::onlyIbmFractionalOps,
      computeTwoQubitUnitaryFromModule);
  expectEquivalentAndNativeAfterSynthesis(
      [&] {
        return mlir::qc::QCProgramBuilder::build(
            context.get(), mlir::qc::nativeSynthCustomMenusXxMinusYyOnly);
      },
      kIbmFrac, &NativeSynthesisPassTest::onlyIbmFractionalOps,
      computeTwoQubitUnitaryFromModule);
}

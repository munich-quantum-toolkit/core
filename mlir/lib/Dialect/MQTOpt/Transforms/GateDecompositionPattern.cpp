/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "Helpers.h"
#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <eigen3/unsupported/Eigen/MatrixFunctions>
#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <numbers>
#include <string>
#include <utility>

namespace mqt::ir::opt {

/**
 * @brief This pattern attempts to cancel consecutive self-inverse operations.
 */
struct GateDecompositionPattern final
    : mlir::OpInterfaceRewritePattern<UnitaryInterface> {
  enum class EulerBasis : std::uint8_t {
    U3 = 0,
    U321 = 1,
    U = 2,
    PSX = 3,
    U1X = 4,
    RR = 5,
    ZYZ = 6,
    ZXZ = 7,
    XZX = 8,
    XYX = 9,
    ZSXX = 10,
    ZSX = 11,
  };

  struct QubitGateSequence {
    struct Gate {
      qc::OpType type{qc::I};
      llvm::SmallVector<fp, 3> parameter;
      llvm::SmallVector<std::size_t, 2> qubitId = {0};
    };
    std::vector<Gate> gates;
    [[nodiscard]] std::size_t complexity() const {
      std::size_t c{};
      for (auto&& gate : gates) {
        c += getComplexity(gate.type, gate.qubitId.size());
      }
      if (hasGlobalPhase()) {
        // need to add two phase gates if a global phase needs to be applied
        c += 2 * getComplexity(qc::P, 1);
      }
      return c;
    }

    fp globalPhase{};
    [[nodiscard]] bool hasGlobalPhase() const {
      return std::abs(globalPhase) > std::numeric_limits<fp>::epsilon();
    }

    [[nodiscard]] matrix4x4 getUnitaryMatrix() const {
      matrix4x4 unitaryMatrix =
          helpers::kroneckerProduct(IDENTITY_GATE, IDENTITY_GATE);
      for (auto&& gate : gates) {
        auto gateMatrix = getTwoQubitMatrix(gate);
        unitaryMatrix = gateMatrix * unitaryMatrix;
      }
      unitaryMatrix *= std::exp(IM * globalPhase);
      assert(helpers::isUnitaryMatrix(unitaryMatrix));
      return unitaryMatrix;
    }
  };
  using OneQubitGateSequence = QubitGateSequence;
  using TwoQubitGateSequence = QubitGateSequence;

  explicit GateDecompositionPattern(
      mlir::MLIRContext* context,
      llvm::SmallVector<QubitGateSequence::Gate> basisGate,
      llvm::SmallVector<EulerBasis> eulerBasis)
      : OpInterfaceRewritePattern(context),
        decomposerBasisGate{std::move(basisGate)},
        decomposerEulerBasis{std::move(eulerBasis)} {}

  mlir::LogicalResult
  matchAndRewrite(UnitaryInterface op,
                  mlir::PatternRewriter& rewriter) const override {
    auto series = TwoQubitSeries::getTwoQubitSeries(op);
    llvm::errs() << "SERIES SIZE: " << series.gates.size() << '\n';
    for (auto&& gate : series.gates) {
      std::cerr << gate.op->getName().stripDialect().str() << ", ";
    }
    std::cerr << '\n';

    if (series.gates.size() < 3) {
      // too short
      return mlir::failure();
    }
    if (llvm::is_contained(series.inQubits, mlir::Value{}) ||
        llvm::is_contained(series.outQubits, mlir::Value{})) {
      // only a single-qubit series
      return mlir::failure();
    }

    matrix4x4 unitaryMatrix = series.getUnitaryMatrix();
    helpers::print(unitaryMatrix, "UNITARY MATRIX", true);

    auto decomposer = TwoQubitBasisDecomposer::newInner(
        decomposerBasisGate[0], 1.0, decomposerEulerBasis[0]);
    auto sequence = decomposer.twoQubitDecompose(
        unitaryMatrix, DEFAULT_FIDELITY, false, std::nullopt);
    if (!sequence) {
      llvm::errs() << "NO SEQUENCE GENERATED!\n";
      return mlir::failure();
    }
    // only accept new sequence if it shortens existing series by more than two
    // gates; this prevents an oscillation with phase gates
    if (sequence->complexity() + 2 >= series.complexity) {
      // TODO: add more sophisticated metric to determine complexity of
      // series/sequence
      llvm::errs() << "SEQUENCE LONGER THAN INPUT (" << sequence->gates.size()
                   << ")\n";
      return mlir::failure();
    }

    applySeries(rewriter, series, *sequence);

    return mlir::success();
  }

protected:
  static constexpr fp SANITY_CHECK_PRECISION = 1e-12;
  [[nodiscard]] static std::size_t getComplexity(qc::OpType /*type*/,
                                                 std::size_t numOfQubits) {
    if (numOfQubits > 1) {
      constexpr std::size_t multiQubitFactor = 10;
      return (numOfQubits - 1) * multiQubitFactor;
    }
    return 1;
  }

  struct TwoQubitSeries {
    std::size_t complexity{0};
    std::array<mlir::Value, 2> inQubits;
    std::array<mlir::Value, 2> outQubits;
    fp globalPhase{};

    struct Gate {
      UnitaryInterface op;
      llvm::SmallVector<std::size_t, 2> qubitIds;
    };
    llvm::SmallVector<Gate, 8> gates;

    [[nodiscard]] static TwoQubitSeries getTwoQubitSeries(UnitaryInterface op) {
      TwoQubitSeries result(op);

      auto getUser = [](mlir::Value qubit,
                        auto&& filter) -> std::optional<UnitaryInterface> {
        if (qubit) {
          assert(qubit.hasOneUse());
          auto user =
              mlir::dyn_cast<UnitaryInterface>(*qubit.getUsers().begin());
          if (user && filter(user)) {
            return user;
          }
        }
        return std::nullopt;
      };

      bool foundGate = true;
      while (foundGate) {
        foundGate = false;
        // collect all available single-qubit operations
        for (std::size_t i = 0; i < result.outQubits.size(); ++i) {
          while (auto user = getUser(result.outQubits[i],
                                     &helpers::isSingleQubitOperation)) {
            foundGate = result.appendSingleQubitGate(*user);
            assert(foundGate); // appending a single-qubit gate should not fail
          }
        }

        for (std::size_t i = 0; i < result.outQubits.size(); ++i) {
          if (auto user =
                  getUser(result.outQubits[i], &helpers::isTwoQubitOperation)) {
            foundGate = result.appendTwoQubitGate(*user);
            break; // go back to single-qubit collection
          }
        }
      }
      return result;
    }

    matrix4x4 getUnitaryMatrix() {
      matrix4x4 unitaryMatrix =
          helpers::kroneckerProduct(IDENTITY_GATE, IDENTITY_GATE);
      for (auto&& gate : gates) {
        auto gateMatrix =
            getTwoQubitMatrix({.type = helpers::getQcType(gate.op),
                               .parameter = helpers::getParameters(gate.op),
                               .qubitId = gate.qubitIds});
        unitaryMatrix = gateMatrix * unitaryMatrix;
      }
      unitaryMatrix *= std::exp(IM * globalPhase);

      assert(helpers::isUnitaryMatrix(unitaryMatrix));
      return unitaryMatrix;
    }

  private:
    explicit TwoQubitSeries(UnitaryInterface initialOperation) {
      auto&& in = initialOperation.getAllInQubits();
      auto&& out = initialOperation->getResults();
      if (helpers::isSingleQubitOperation(initialOperation)) {
        inQubits = {in[0], mlir::Value{}};
        outQubits = {out[0], mlir::Value{}};
        gates.push_back({.op = initialOperation, .qubitIds = {0}});
      } else if (helpers::isTwoQubitOperation(initialOperation)) {
        inQubits = {in[0], in[1]};
        outQubits = {out[0], out[1]};
        gates.push_back({.op = initialOperation, .qubitIds = {0, 1}});
      }
      complexity +=
          getComplexity(helpers::getQcType(initialOperation), in.size());

      // TODO: necessary when using non-global phase gates?
      for (auto&& globalPhaseOp :
           initialOperation->getBlock()->getOps<GPhaseOp>()) {
        globalPhase += helpers::getParameters(globalPhaseOp)[0];
      }
    }

    /**
     * @return true if series continues, otherwise false
     *         (will always return true)
     */
    bool appendSingleQubitGate(UnitaryInterface nextGate) {
      auto operand = nextGate.getAllInQubits()[0];
      auto* it = llvm::find(outQubits, operand);
      if (it == outQubits.end()) {
        throw std::logic_error{"Operand of single-qubit op and user of "
                               "qubit is not in current outQubits"};
      }
      std::size_t qubitId = std::distance(outQubits.begin(), it);
      *it = nextGate->getResult(0);

      gates.push_back({.op = nextGate, .qubitIds = {qubitId}});
      complexity += getComplexity(helpers::getQcType(nextGate), 1);
      return true;
    }

    /**
     * @return true if series continues, otherwise false
     */
    bool appendTwoQubitGate(UnitaryInterface nextGate) {
      auto opInQubits = nextGate.getAllInQubits();
      auto&& firstOperand = opInQubits[0];
      auto&& secondOperand = opInQubits[1];
      auto* firstQubitIt = llvm::find(outQubits, firstOperand);
      auto* secondQubitIt = llvm::find(outQubits, secondOperand);
      if (firstQubitIt == outQubits.end() || secondQubitIt == outQubits.end()) {
        // another qubit is involved, series is finished (except there only
        // has been one qubit so far)
        auto* it = llvm::find(outQubits, mlir::Value{});
        if (it == outQubits.end()) {
          return false;
        }
        auto it2 = llvm::find(opInQubits, firstQubitIt != outQubits.end()
                                              ? *firstQubitIt
                                              : *secondQubitIt);
        inQubits[std::distance(outQubits.begin(), it)] =
            opInQubits[1 - std::distance(opInQubits.begin(), it2)];
        firstQubitIt = (firstQubitIt != outQubits.end()) ? firstQubitIt : it;
        secondQubitIt = (secondQubitIt != outQubits.end()) ? secondQubitIt : it;
      }
      std::size_t firstQubitId = std::distance(outQubits.begin(), firstQubitIt);
      std::size_t secondQubitId =
          std::distance(outQubits.begin(), secondQubitIt);
      *firstQubitIt = nextGate->getResult(0);
      *secondQubitIt = nextGate->getResult(1);

      gates.push_back(
          {.op = nextGate, .qubitIds = {firstQubitId, secondQubitId}});
      complexity += getComplexity(helpers::getQcType(nextGate), 2);
      return true;
    }
  };

  /**
   * @brief Creates a new rotation gate with no controls.
   *
   * @tparam OpType The type of the operation to be created.
   * @param op The first instance of the rotation gate.
   * @param rewriter The pattern rewriter.
   * @return A new rotation gate.
   */
  template <typename OpType>
  static OpType createOneParameterGate(mlir::PatternRewriter& rewriter,
                                       mlir::Location location, fp parameter,
                                       mlir::ValueRange inQubits) {
    auto parameterValue = rewriter.create<mlir::arith::ConstantOp>(
        location, rewriter.getF64Type(), rewriter.getF64FloatAttr(parameter));

    return rewriter.create<OpType>(
        location, inQubits.getType(), mlir::TypeRange{}, mlir::TypeRange{},
        mlir::DenseF64ArrayAttr{}, mlir::DenseBoolArrayAttr{},
        mlir::ValueRange{parameterValue}, inQubits, mlir::ValueRange{},
        mlir::ValueRange{});
  }

  template <typename OpType>
  static OpType createGate(mlir::PatternRewriter& rewriter,
                           mlir::Location location, mlir::ValueRange inQubits,
                           mlir::ValueRange ctrlQubits,
                           const llvm::SmallVector<fp, 3>& parameters) {
    mlir::SmallVector<mlir::Value, 2> parameterValues;
    for (auto&& parameter : parameters) {
      auto parameterValue = rewriter.create<mlir::arith::ConstantOp>(
          location, rewriter.getF64Type(), rewriter.getF64FloatAttr(parameter));
      parameterValues.push_back(parameterValue);
    }

    return rewriter.create<OpType>(
        location, inQubits.getType(), ctrlQubits.getType(), mlir::TypeRange{},
        mlir::DenseF64ArrayAttr{}, mlir::DenseBoolArrayAttr{}, parameterValues,
        inQubits, ctrlQubits, mlir::ValueRange{});
  }

  static void applySeries(mlir::PatternRewriter& rewriter,
                          TwoQubitSeries& series,
                          const TwoQubitGateSequence& sequence) {
    auto& lastSeriesOp = series.gates.back().op;
    auto location = lastSeriesOp->getLoc();
    rewriter.setInsertionPointAfter(lastSeriesOp);

    auto inQubits = series.inQubits;
    auto updateInQubits =
        [&inQubits](const llvm::SmallVector<std::size_t, 2>& qubitIds,
                    auto&& newGate) {
          // TODO: need to handle controls differently?
          auto results = newGate.getAllOutQubits();
          if (qubitIds.size() == 2) {
            inQubits[qubitIds[0]] = results[0];
            inQubits[qubitIds[1]] = results[1];
          } else if (qubitIds.size() == 1) {
            inQubits[qubitIds[0]] = results[0];
          } else {
            throw std::logic_error{"Invalid number of qubit IDs!"};
          }
        };

    if (sequence.hasGlobalPhase()) {
      createOneParameterGate<GPhaseOp>(rewriter, location, sequence.globalPhase,
                                       {});
    }

    std::cerr << "SERIES: ";
    for (auto&& gate : series.gates) {
      auto name = gate.op->getName().stripDialect().str();
      if (name == "x" && gate.qubitIds.size() == 2) {
        // controls come first
        std::cerr << std::format("cx() q[{}], q[{}];", gate.qubitIds[1],
                                 gate.qubitIds[0]);
      } else if (name == "i") {
      } else if (gate.op.getParams().empty()) {
        std::cerr << std::format(
            "{}() q[{}] {};", name, gate.qubitIds[0],
            (gate.qubitIds.size() > 1
                 ? (", q[" + std::to_string(gate.qubitIds[1]) + "]")
                 : std::string{}));
      } else {
        auto parameter = helpers::getParameters(gate.op)[0];
        std::cerr << std::format(
            "{}({}*pi) q[{}] {};", name, parameter / qc::PI, gate.qubitIds[0],
            (gate.qubitIds.size() > 1
                 ? (", q[" + std::to_string(gate.qubitIds[1]) + "]")
                 : std::string{}));
      }
    }
    std::cerr << '\n';
    std::cerr << "GATE SEQUENCE!: gphase(" << sequence.globalPhase / qc::PI
              << "*pi); \n";
    matrix4x4 unitaryMatrix =
        helpers::kroneckerProduct(IDENTITY_GATE, IDENTITY_GATE);
    for (auto&& gate : sequence.gates) {
      auto gateMatrix = getTwoQubitMatrix(gate);
      unitaryMatrix = gateMatrix * unitaryMatrix;

      if (gate.type == qc::X && gate.qubitId.size() == 2) {
        // controls come first
        std::cerr << std::format("cx() q[{}], q[{}];", gate.qubitId[1],
                                 gate.qubitId[0]);
      } else if (gate.parameter.empty()) {
        std::cerr << std::format(
            "{}() q[{}] {};", qc::toString(gate.type), gate.qubitId[0],
            (gate.qubitId.size() > 1
                 ? (", q[" + std::to_string(gate.qubitId[1]) + "]")
                 : std::string{}));
      } else {
        std::cerr << std::format(
            "{}({}*pi) q[{}] {};", qc::toString(gate.type),
            gate.parameter[0] / qc::PI, gate.qubitId[0],
            (gate.qubitId.size() > 1
                 ? (", q[" + std::to_string(gate.qubitId[1]) + "]")
                 : std::string{}));
      }
      std::cerr << '\n';
      if (gate.type == qc::X) {
        mlir::SmallVector<mlir::Value, 1> inCtrlQubits;
        if (gate.qubitId.size() > 1) {
          // controls come first
          inCtrlQubits.push_back(inQubits[gate.qubitId[1]]);
        }
        auto newGate =
            createGate<XOp>(rewriter, location, {inQubits[gate.qubitId[0]]},
                            inCtrlQubits, gate.parameter);
        updateInQubits(gate.qubitId, newGate);
      } else if (gate.type == qc::RX) {
        mlir::SmallVector<mlir::Value, 2> qubits;
        for (auto&& x : gate.qubitId) {
          qubits.push_back(inQubits[x]);
        }
        auto newGate =
            createGate<RXOp>(rewriter, location, qubits, {}, gate.parameter);
        updateInQubits(gate.qubitId, newGate);
      } else if (gate.type == qc::RY) {
        mlir::SmallVector<mlir::Value, 2> qubits;
        for (auto&& x : gate.qubitId) {
          qubits.push_back(inQubits[x]);
        }
        auto newGate =
            createGate<RYOp>(rewriter, location, qubits, {}, gate.parameter);
        updateInQubits(gate.qubitId, newGate);
      } else if (gate.type == qc::RZ) {
        mlir::SmallVector<mlir::Value, 2> qubits;
        for (auto&& x : gate.qubitId) {
          qubits.push_back(inQubits[x]);
        }
        auto newGate =
            createGate<RZOp>(rewriter, location, qubits, {}, gate.parameter);
        updateInQubits(gate.qubitId, newGate);
      } else {
        throw std::runtime_error{"Unknown gate type!"};
      }
    }
    std::cerr << '\n';
    helpers::print(series.getUnitaryMatrix(), "ORIGINAL UNITARY", true);
    helpers::print((unitaryMatrix * std::exp(IM * sequence.globalPhase)).eval(),
                   "RESULT UNITARY MATRIX", true);
    assert((unitaryMatrix * std::exp(IM * sequence.globalPhase))
               .isApprox(series.getUnitaryMatrix(), SANITY_CHECK_PRECISION));

    rewriter.replaceAllUsesWith(series.outQubits, inQubits);
    for (auto&& gate : llvm::reverse(series.gates)) {
      rewriter.eraseOp(gate.op);
    }
  }

  enum class Specialization : std::uint8_t {
    General,
    IdEquiv,
    SWAPEquiv,
    PartialSWAPEquiv,
    PartialSWAPFlipEquiv,
    ControlledEquiv,
    MirrorControlledEquiv,
    // These next 3 gates use the definition of fSim from eq (1) in:
    // https://arxiv.org/pdf/2001.08343.pdf
    FSimaabEquiv,
    FSimabbEquiv,
    FSimabmbEquiv,
  };

  enum class MagicBasisTransform : std::uint8_t {
    Into,
    OutOf,
  };

  static constexpr auto SQRT2 = std::numbers::sqrt2_v<fp>;
  static constexpr auto FRAC1_SQRT2 =
      static_cast<fp>(0.707106781186547524400844362104849039);
  static const matrix2x2 IDENTITY_GATE;
  static const matrix2x2 H_GATE;

  static fp remEuclid(fp a, fp b) {
    auto r = std::fmod(a, b);
    return (r < 0.0) ? r + std::abs(b) : r;
  }

  // Wrap angle into interval [-π,π). If within atol of the endpoint, clamp
  // to -π
  static fp mod2pi(fp angle, fp angleZeroEpsilon = 1e-13) {
    // remEuclid() isn't exactly the same as Python's % operator, but
    // because the RHS here is a constant and positive it is effectively
    // equivalent for this case
    auto wrapped = remEuclid(angle + qc::PI, 2. * qc::PI) - qc::PI;
    if (std::abs(wrapped - qc::PI) < angleZeroEpsilon) {
      return -qc::PI;
    }
    return wrapped;
  }

  // https://docs.rs/faer/latest/faer/mat/generic/struct.Mat.html#method.self_adjoint_eigen
  template <typename T> static auto selfAdjointEigenLower(T&& a) {
    // rdiagonal4x4 S;
    // auto U = self_adjoint_evd(A, S);

    // rmatrix4x4 U;
    // jacobi_eigen_decomposition(A, U, S);

    auto [U, S] = helpers::selfAdjointEvd(std::forward<T>(a));

    // TODO: not in original code
    if (std::real(U.determinant()) < 0.0) {
      std::cerr << "CORRECTION!\n";
      // if determinant of eigenvector matrix is -1.0, multiply first
      // eigenvector by -1.0
      U.col(0) *= -1.0;
      // U.col(U.cols() - 1) *= -1.0;
      // U *= -1.0;
      // U += std::remove_cvref_t<T>::Constant(0.0); // ensure no -0.0 exists
    }

    return std::make_pair(U, S);
  }

  static std::tuple<matrix2x2, matrix2x2, fp>
  decomposeTwoQubitProductGate(matrix4x4 specialUnitary) {
    // see pennylane math.decomposition.su2su2_to_tensor_products
    // or "7.1 Kronecker decomposition" in on_gates.pdf
    // or quantumflow.kronecker_decomposition

    helpers::print(specialUnitary, "SPECIAL_UNITARY");
    // first quadrant
    matrix2x2 r{{specialUnitary(0, 0), specialUnitary(0, 1)},
                {specialUnitary(1, 0), specialUnitary(1, 1)}};
    auto detR = r.determinant();
    std::cerr << "DET_R: " << detR << '\n';
    if (std::abs(detR) < 0.1) {
      // third quadrant
      r = matrix2x2{{specialUnitary(2, 0), specialUnitary(2, 1)},
                    {specialUnitary(3, 0), specialUnitary(3, 1)}};
      detR = r.determinant();
      std::cerr << "DET_R CORRECTION: " << detR << '\n';
    }
    if (std::abs(detR) < 0.1) {
      throw std::runtime_error{
          "decompose_two_qubit_product_gate: unable to decompose: det_r < 0.1"};
    }
    r /= std::sqrt(detR);
    helpers::print(r, "R");
    // transpose with complex conjugate of each element
    matrix2x2 rTConj = r.transpose().conjugate();

    auto temp = helpers::kroneckerProduct(IDENTITY_GATE, rTConj);
    helpers::print(temp, "TEMP (decompose_two_qubit_product_gate, 1)");
    temp = specialUnitary * temp;
    helpers::print(temp, "TEMP (decompose_two_qubit_product_gate, 2)");

    // [[a, b, c, d],
    //  [e, f, g, h], => [[a, c],
    //  [i, j, k, l],     [i, k]]
    //  [m, n, o, p]]
    matrix2x2 l{{temp(0, 0), temp(0, 2)}, {temp(2, 0), temp(2, 2)}};
    auto detL = l.determinant();
    if (std::abs(detL) < 0.9) {
      throw std::runtime_error{
          "decompose_two_qubit_product_gate: unable to decompose: detL < 0.9"};
    }
    l /= std::sqrt(detL);
    auto phase = std::arg(detL) / 2.;

    return {l, r, phase};
  }

  static matrix4x4 magicBasisTransform(const matrix4x4& unitary,
                                       MagicBasisTransform direction) {
    const matrix4x4 bNonNormalized{
        {C_ONE, IM, C_ZERO, C_ZERO},
        {C_ZERO, C_ZERO, IM, C_ONE},
        {C_ZERO, C_ZERO, IM, C_M_ONE},
        {C_ONE, M_IM, C_ZERO, C_ZERO},
    };
    const matrix4x4 b = FRAC1_SQRT2 * matrix4x4{{C_ONE, C_ZERO, C_ZERO, IM},
                                                {C_ZERO, IM, C_ONE, C_ZERO},
                                                {C_ZERO, IM, -C_ONE, C_ZERO},
                                                {C_ONE, C_ZERO, C_ZERO, -IM}};

    const matrix4x4 bNonNormalizedDagger{
        {qfp(0.5, 0.), C_ZERO, C_ZERO, qfp(0.5, 0.)},
        {qfp(0., -0.5), C_ZERO, C_ZERO, qfp(0., 0.5)},
        {C_ZERO, qfp(0., -0.5), qfp(0., -0.5), C_ZERO},
        {C_ZERO, qfp(0.5, 0.), qfp(-0.5, 0.), C_ZERO},
    };
    const matrix4x4 bDagger = b.conjugate().transpose();
    helpers::print(unitary, "UNITARY in MAGIC BASIS TRANSFORM");
    if (direction == MagicBasisTransform::OutOf) {
      // return bDagger * unitary * b; // TODO: same result?
      return bNonNormalizedDagger * unitary * bNonNormalized;
    }
    if (direction == MagicBasisTransform::Into) {
      // return b * unitary * bDagger;
      return bNonNormalized * unitary * bNonNormalizedDagger;
    }
    throw std::logic_error{"Unknown MagicBasisTransform direction!"};
  }

  static fp traceToFid(const qfp& x) {
    auto xAbs = std::abs(x);
    return (4.0 + xAbs * xAbs) / 20.0;
  }

  static fp closestPartialSwap(fp a, fp b, fp c) {
    auto m = (a + b + c) / 3.;
    auto [am, bm, cm] = std::array{a - m, b - m, c - m};
    auto [ab, bc, ca] = std::array{a - b, b - c, c - a};
    return m + (am * bm * cm * (6. + ab * ab + bc * bc + ca * ca) / 18.);
  }

  static matrix2x2 rxMatrix(fp theta) {
    auto halfTheta = theta / 2.;
    auto cos = qfp(std::cos(halfTheta), 0.);
    auto isin = qfp(0., -std::sin(halfTheta));
    return matrix2x2{{cos, isin}, {isin, cos}};
  }

  static matrix2x2 ryMatrix(fp theta) {
    auto halfTheta = theta / 2.;
    auto cos = qfp(std::cos(halfTheta), 0.);
    auto sin = qfp(std::sin(halfTheta), 0.);
    return matrix2x2{{cos, -sin}, {sin, cos}};
  }

  static matrix2x2 rzMatrix(fp theta) {
    return matrix2x2{{qfp{std::cos(theta / 2.), -std::sin(theta / 2.)}, 0},
                     {0, qfp{std::cos(theta / 2.), std::sin(theta / 2.)}}};
    // auto ilam2 = qfp(0., 0.5 * theta);
    // return {std::exp(-ilam2), C_ZERO, C_ZERO, std::exp(ilam2)};
  }

  static matrix4x4 rxxMatrix(const fp theta) {
    const auto cosTheta = std::cos(theta / 2.);
    const auto sinTheta = std::sin(theta / 2.);

    return matrix4x4{{cosTheta, C_ZERO, C_ZERO, {0., -sinTheta}},
                     {C_ZERO, cosTheta, {0., -sinTheta}, C_ZERO},
                     {C_ZERO, {0., -sinTheta}, cosTheta, C_ZERO},
                     {{0., -sinTheta}, C_ZERO, C_ZERO, cosTheta}};
  }

  static matrix4x4 ryyMatrix(const fp theta) {
    const auto cosTheta = std::cos(theta / 2.);
    const auto sinTheta = std::sin(theta / 2.);

    return matrix4x4{{{cosTheta, 0, 0, {0., sinTheta}},
                      {0, cosTheta, {0., -sinTheta}, 0},
                      {0, {0., -sinTheta}, cosTheta, 0},
                      {std::complex{0., sinTheta}, 0, 0, cosTheta}}};
  }

  static matrix4x4 rzzMatrix(const fp theta) {
    const auto cosTheta = std::cos(theta / 2.);
    const auto sinTheta = std::sin(theta / 2.);

    return matrix4x4{{qfp{cosTheta, -sinTheta}, C_ZERO, C_ZERO, C_ZERO},
                     {C_ZERO, {cosTheta, sinTheta}, C_ZERO, C_ZERO},
                     {C_ZERO, C_ZERO, {cosTheta, sinTheta}, C_ZERO},
                     {C_ZERO, C_ZERO, C_ZERO, {cosTheta, -sinTheta}}};
  }

  static matrix2x2 pMatrix(const fp lambda) {
    return matrix2x2{{1, 0}, {0, {std::cos(lambda), std::sin(lambda)}}};
  }

  static std::array<fp, 4> anglesFromUnitary(const matrix2x2& matrix,
                                             EulerBasis basis) {
    if (basis == EulerBasis::XYX) {
      return paramsXyxInner(matrix);
    }
    if (basis == EulerBasis::ZYZ) {
      return paramsZyzInner(matrix);
    }
    if (basis == EulerBasis::ZXZ) {
      return paramsZxzInner(matrix);
    }
    throw std::invalid_argument{"Unknown EulerBasis for angles_from_unitary"};
  }

  static std::array<fp, 4> paramsZyzInner(const matrix2x2& matrix) {
    auto detArg = std::arg(matrix.determinant());
    auto phase = 0.5 * detArg;
    auto theta =
        2. * std::atan2(std::abs(matrix(1, 0)), std::abs(matrix(0, 0)));
    auto ang1 = std::arg(matrix(1, 1));
    auto ang2 = std::arg(matrix(1, 0));
    auto phi = ang1 + ang2 - detArg;
    auto lam = ang1 - ang2;
    return {theta, phi, lam, phase};
  }

  static std::array<fp, 4> paramsZxzInner(const matrix2x2& matrix) {
    auto [theta, phi, lam, phase] = paramsZyzInner(matrix);
    return {theta, phi + (qc::PI / 2.), lam - (qc::PI / 2.), phase};
  }

  static std::array<fp, 4> paramsXyxInner(const matrix2x2& matrix) {
    matrix2x2 matZyz{
        {static_cast<fp>(0.5) *
             (matrix(0, 0) + matrix(0, 1) + matrix(1, 0) + matrix(1, 1)),
         static_cast<fp>(0.5) *
             (matrix(0, 0) - matrix(0, 1) + matrix(1, 0) - matrix(1, 1))},
        {static_cast<fp>(0.5) *
             (matrix(0, 0) + matrix(0, 1) - matrix(1, 0) - matrix(1, 1)),
         static_cast<fp>(0.5) *
             (matrix(0, 0) - matrix(0, 1) - matrix(1, 0) + matrix(1, 1))},
    };
    auto [theta, phi, lam, phase] = paramsZyzInner(matZyz);
    auto newPhi = mod2pi(phi + qc::PI, 0.);
    auto newLam = mod2pi(lam + qc::PI, 0.);
    return {
        theta,
        newPhi,
        newLam,
        phase + ((newPhi + newLam - phi - lam) / 2.),
    };
  }

  static const matrix2x2 IPZ;
  static const matrix2x2 IPY;
  static const matrix2x2 IPX;

  static matrix2x2 getSingleQubitMatrix(const QubitGateSequence::Gate& gate) {
    if (gate.type == qc::SX) {
      return matrix2x2{{qfp{0.5, 0.5}, qfp{0.5, -0.5}},
                       {qfp{0.5, -0.5}, qfp{0.5, 0.5}}};
    }
    if (gate.type == qc::RX) {
      return rxMatrix(gate.parameter[0]);
    }
    if (gate.type == qc::RY) {
      return ryMatrix(gate.parameter[0]);
    }
    if (gate.type == qc::RZ) {
      return rzMatrix(gate.parameter[0]);
    }
    if (gate.type == qc::X) {
      return matrix2x2{{0, 1}, {1, 0}};
    }
    if (gate.type == qc::I) {
      return IDENTITY_GATE;
    }
    if (gate.type == qc::P) {
      return pMatrix(gate.parameter[0]);
    }
    if (gate.type == qc::H) {
      static constexpr fp SQRT2_2 = static_cast<fp>(
          0.707106781186547524400844362104849039284835937688474036588L);
      return matrix2x2{{SQRT2_2, SQRT2_2}, {SQRT2_2, -SQRT2_2}};
    }
    throw std::invalid_argument{
        "unsupported gate type for single qubit matrix (" +
        qc::toString(gate.type) + ")"};
  }

  static matrix4x4 getTwoQubitMatrix(const QubitGateSequence::Gate& gate) {
    using helpers::kroneckerProduct;

    if (gate.qubitId.empty()) {
      return kroneckerProduct(IDENTITY_GATE, IDENTITY_GATE);
    }
    if (gate.qubitId.size() == 1) {
      if (gate.qubitId[0] == 0) {
        return kroneckerProduct(IDENTITY_GATE, getSingleQubitMatrix(gate));
      }
      if (gate.qubitId[0] == 1) {
        return kroneckerProduct(getSingleQubitMatrix(gate), IDENTITY_GATE);
      }
      throw std::logic_error{"Invalid qubit ID in getTwoQubitMatrix"};
    }
    if (gate.qubitId.size() == 2) {
      if (gate.type == qc::X) {
        // controlled X (CX)
        if (gate.qubitId == llvm::SmallVector<std::size_t, 2>{0, 1}) {
          return matrix4x4{
              {1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}};
        }
        if (gate.qubitId == llvm::SmallVector<std::size_t, 2>{1, 0}) {
          return matrix4x4{
              {1, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}, {0, 1, 0, 0}};
        }
      }
      if (gate.type == qc::RXX) {
        // TODO: check qubit order
        return rxxMatrix(gate.parameter[0]);
      }
      if (gate.type == qc::RYY) {
        // TODO: check qubit order
        return ryyMatrix(gate.parameter[0]);
      }
      if (gate.type == qc::RZZ) {
        // TODO: check qubit order
        return rzzMatrix(gate.parameter[0]);
      }
      if (gate.type == qc::I) {
        return kroneckerProduct(IDENTITY_GATE, IDENTITY_GATE);
      }
      throw std::invalid_argument{
          "unsupported gate type for two qubit matrix "};
    }
    throw std::logic_error{"Invalid number of qubit IDs in compute_unitary"};
  }

  struct TwoQubitWeylDecomposition {
    fp a;
    fp b;
    fp c;
    fp globalPhase;
    /**
     * q1 - k2r - C - k1r -
     *            A
     * q0 - k2l - N - k1l -
     */
    matrix2x2 k1l;
    matrix2x2 k2l;
    matrix2x2 k1r;
    matrix2x2 k2r;
    Specialization specialization;
    EulerBasis defaultEulerBasis;
    std::optional<fp> requestedFidelity;
    fp calculatedFidelity;
    matrix4x4 unitaryMatrix;

    static TwoQubitWeylDecomposition
    newInner(matrix4x4 unitaryMatrix, std::optional<fp> fidelity,
             std::optional<Specialization> specialization) {
      auto u = unitaryMatrix;
      auto detU = u.determinant();
      std::cerr << "DET_U: " << detU << '\n';
      auto detPow = std::pow(detU, static_cast<fp>(-0.25));
      u *= detPow;
      helpers::print(u, "U", true);
      auto globalPhase = std::arg(detU) / 4.;
      auto uP = magicBasisTransform(u, MagicBasisTransform::OutOf);
      helpers::print(uP, "U_P", true);
      matrix4x4 m2 = uP.transpose() * uP;
      auto defaultEulerBasis = EulerBasis::ZYZ;
      helpers::print(m2, "M2", true);

      std::cerr << "DET_U after division: " << u.determinant() << '\n';

      // M2 is a symmetric complex matrix. We need to decompose it as M2 = P D
      // P^T where P ∈ SO(4), D is diagonal with unit-magnitude elements.
      //
      // We can't use raw `eig` directly because it isn't guaranteed to give
      // us real or orthogonal eigenvectors. Instead, since `M2` is
      // complex-symmetric,
      //   M2 = A + iB
      // for real-symmetric `A` and `B`, and as
      //   M2^+ @ M2 = A^2 + B^2 + i [A, B] = 1
      // we must have `A` and `B` commute, and consequently they are
      // simultaneously diagonalizable. Mixing them together _should_ account
      // for any degeneracy problems, but it's not guaranteed, so we repeat it
      // a little bit.  The fixed seed is to make failures deterministic; the
      // value is not important.
      auto state = std::mt19937{2023};
      std::normal_distribution<fp> dist;
      auto found = false;
      diagonal4x4 d = diagonal4x4::Zero();
      matrix4x4 p = matrix4x4::Zero();

      for (int i = 0; i < 100; ++i) {
        fp randA{};
        fp randB{};
        // For debugging the algorithm use the same RNG values from the
        // previous Python implementation for the first random trial.
        // In most cases this loop only executes a single iteration and
        // using the same rng values rules out possible RNG differences
        // as the root cause of a test failure
        if (i == 0) {
          randA = 1.2602066112249388;
          randB = 0.22317849046722027;
        } else {
          randA = dist(state);
          randB = dist(state);
        }
        rmatrix4x4 m2Real = randA * m2.real() + randB * m2.imag();
        rmatrix4x4 pInnerReal = selfAdjointEigenLower(m2Real).first;
        matrix4x4 pInner = pInnerReal;
        diagonal4x4 dInner = (pInner.transpose() * m2 * pInner).diagonal();

        helpers::print(dInner, "D_INNER", true);
        helpers::print(pInner, "P_INNER", true);
        matrix4x4 diagD = dInner.asDiagonal();

        matrix4x4 compare = pInner * diagD * pInner.transpose();
        helpers::print(compare, "COMPARE");
        found = compare.isApprox(m2, 1e-13);
        if (found) {
          // p are the eigenvectors which are decomposed into the
          // single-qubit gates surrounding the canonical gate
          p = pInner;
          // d is the sqrt of the eigenvalues that are used to determine the
          // weyl coordinates and thus the parameters of the canonical gate
          d = dInner;
          break;
        }
      }
      if (!found) {
        throw std::runtime_error{
            "TwoQubitWeylDecomposition: failed to diagonalize M2."};
      }
      // check that p is in SO(4)
      assert((p.transpose() * p).isIdentity(SANITY_CHECK_PRECISION));
      // make sure determinant of sqrt(eigenvalues) is 1.0
      assert(std::abs(matrix4x4{d.asDiagonal()}.determinant() - 1.0) <
             SANITY_CHECK_PRECISION);

      // see
      // https://github.com/mpham26uchicago/laughing-umbrella/blob/main/background/Full%20Two%20Qubit%20KAK%20Implementation.ipynb,
      // Step 7
      Eigen::Vector<fp, 3> cs;
      rdiagonal4x4 dReal = -1.0 * d.cwiseArg() / 2.0;
      helpers::print(dReal, "D_REAL", true);
      dReal(3) = -dReal(0) - dReal(1) - dReal(2);
      for (int i = 0; i < static_cast<int>(cs.size()); ++i) {
        assert(i < dReal.size());
        cs[i] = remEuclid((dReal(i) + dReal(3)) / 2.0, qc::TAU);
      }
      helpers::print(cs, "CS (1)", true);

      decltype(cs) cstemp;
      llvm::transform(cs, cstemp.begin(), [](auto&& x) {
        auto tmp = remEuclid(x, qc::PI_2);
        return std::min(tmp, qc::PI_2 - tmp);
      });
      std::array<int, cstemp.size()> order{
          0, 1, 2}; // TODO: needs to be adjusted depending on eigenvector
                    // order in eigen decomposition algorithm?
      llvm::stable_sort(order,
                        [&](auto a, auto b) { return cstemp[a] < cstemp[b]; });
      // llvm::stable_sort(order, [&](fp a, fp b) {
      //   auto tmp1 = remEuclid(cs[a], qc::PI_2);
      //   tmp1 = std::min(tmp1, qc::PI_2 - tmp1);
      //   auto tmp2 = remEuclid(cs[b], qc::PI_2);
      //   tmp2 = std::min(tmp2, qc::PI_2 - tmp2);
      //   return tmp1 < tmp2;
      // });
      std::tie(order[0], order[1], order[2]) =
          std::tuple{order[1], order[2], order[0]};
      std::tie(cs[0], cs[1], cs[2]) =
          std::tuple{cs[order[0]], cs[order[1]], cs[order[2]]};
      std::tie(dReal(0), dReal(1), dReal(2)) =
          std::tuple{dReal(order[0]), dReal(order[1]), dReal(order[2])};
      helpers::print(dReal, "D_REAL (sorted)", true);

      // swap columns of p according to order
      matrix4x4 pOrig = p;
      for (int i = 0; i < static_cast<int>(order.size()); ++i) {
        p.col(i) = pOrig.col(order[i]);
      }
      if (p.determinant().real() < 0.0) {
        std::cerr << "SECOND CORRECTION?\n";
        auto lastColumnIndex = p.cols() - 1;
        p.col(lastColumnIndex) *= -1.0;
      }

      matrix4x4 temp = dReal.asDiagonal();
      temp *= IM;
      temp = temp.exp();

      // temp = temp.conjugate();
      // temp += matrix4x4::Constant(0.0);
      helpers::print(temp, "TEMP", true);
      helpers::print(p, "P", true);
      assert(std::abs(p.determinant() - 1.0) < SANITY_CHECK_PRECISION);
      // https://threeplusone.com/pubs/on_gates.pdf
      // uP = V, m2 = V^T*V, temp = D, p = Q1
      matrix4x4 k1 = uP * p * temp;
      helpers::print(k1, "K1 (1)", true);
      assert((k1.transpose() * k1).isIdentity()); // k1 must be orthogonal
      assert(k1.determinant().real() > 0.0);
      k1 = magicBasisTransform(k1, MagicBasisTransform::Into);
      matrix4x4 k2 = p.transpose().conjugate();
      helpers::print(k2, "K2 (1)", true);
      assert((k2.transpose() * k2).isIdentity()); // k2 must be orthogonal
      assert(k2.determinant().real() > 0.0);
      k2 = magicBasisTransform(k2, MagicBasisTransform::Into);

      helpers::print(
          (k1 *
           magicBasisTransform(temp.conjugate(), MagicBasisTransform::Into) *
           k2)
              .eval(),
          "SANITY CHECK (1)", true);
      assert((k1 *
              magicBasisTransform(temp.conjugate(), MagicBasisTransform::Into) *
              k2)
                 .isApprox(u, SANITY_CHECK_PRECISION));

      auto [K1l, K1r, phase_l] = decomposeTwoQubitProductGate(k1);
      auto [K2l, K2r, phase_r] = decomposeTwoQubitProductGate(k2);
      assert(helpers::kroneckerProduct(K1l, K1r).isApprox(
          k1, SANITY_CHECK_PRECISION));
      assert(helpers::kroneckerProduct(K2l, K2r).isApprox(
          k2, SANITY_CHECK_PRECISION));
      globalPhase += phase_l + phase_r;

      // Flip into Weyl chamber
      if (cs[0] > qc::PI_2) {
        cs[0] -= 3.0 * qc::PI_2;
        K1l = K1l * IPY;
        K1r = K1r * IPY;
        globalPhase += qc::PI_2;
      }
      if (cs[1] > qc::PI_2) {
        cs[1] -= 3.0 * qc::PI_2;
        K1l = K1l * IPX;
        K1r = K1r * IPX;
        globalPhase += qc::PI_2;
      }
      auto conjs = 0;
      if (cs[0] > qc::PI_4) {
        cs[0] = qc::PI_2 - cs[0];
        K1l = K1l * IPY;
        K2r = IPY * K2r;
        conjs += 1;
        globalPhase -= qc::PI_2;
      }
      if (cs[1] > qc::PI_4) {
        cs[1] = qc::PI_2 - cs[1];
        K1l = K1l * IPX;
        K2r = IPX * K2r;
        conjs += 1;
        globalPhase += qc::PI_2;
        if (conjs == 1) {
          globalPhase -= qc::PI;
        }
      }
      if (cs[2] > qc::PI_2) {
        cs[2] -= 3.0 * qc::PI_2;
        K1l = K1l * IPZ;
        K1r = K1r * IPZ;
        globalPhase += qc::PI_2;
        if (conjs == 1) {
          globalPhase -= qc::PI;
        }
      }
      if (conjs == 1) {
        cs[2] = qc::PI_2 - cs[2];
        K1l = K1l * IPZ;
        K2r = IPZ * K2r;
        globalPhase += qc::PI_2;
      }
      if (cs[2] > qc::PI_4) {
        cs[2] -= qc::PI_2;
        K1l = K1l * IPZ;
        K1r = K1r * IPZ;
        globalPhase -= qc::PI_2;
      }

      helpers::print(K1l, "K1l (1)", true);
      helpers::print(K2l, "K2l (1)", true);
      helpers::print(K1r, "K1r (1)", true);
      helpers::print(K2r, "K2r (1)", true);

      helpers::print(cs, "CS (2)", true);
      helpers::print(K1l, "K1l (2)", true);
      helpers::print(K2l, "K2l (2)", true);
      helpers::print(K1r, "K1r (2)", true);
      helpers::print(K2r, "K2r (2)", true);
      auto [a, b, c] = std::tie(cs[1], cs[0], cs[2]);
      auto getCanonicalMatrix = [](fp a, fp b, fp c) -> matrix4x4 {
        auto xx = getTwoQubitMatrix({
            .type = qc::RXX,
            .parameter = {a},
            .qubitId = {0, 1},
        });
        auto yy = getTwoQubitMatrix({
            .type = qc::RYY,
            .parameter = {b},
            .qubitId = {0, 1},
        });
        auto zz = getTwoQubitMatrix({
            .type = qc::RZZ,
            .parameter = {c},
            .qubitId = {0, 1},
        });
        return zz * yy * xx;
      };
      helpers::print(getCanonicalMatrix(a * -2.0, b * -2.0, c * -2.0),
                     "SANITY CHECK (2.1)", true);
      helpers::print(helpers::kroneckerProduct(K1l, K1r), "SANITY CHECK (2.2)",
                     true);
      helpers::print(helpers::kroneckerProduct(K2l, K2r), "SANITY CHECK (2.3)",
                     true);
      helpers::print((helpers::kroneckerProduct(K1l, K1r) *
                      getCanonicalMatrix(a * -2.0, b * -2.0, c * -2.0) *
                      helpers::kroneckerProduct(K2l, K2r) *
                      std::exp(IM * globalPhase))
                         .eval(),
                     "SANITY CHECK (2.x)", true);
      std::cerr << "gphase: " << globalPhase << ", phase_l: " << phase_l
                << ", phase_r: " << phase_r << '\n';
      assert((helpers::kroneckerProduct(K1l, K1r) *
              getCanonicalMatrix(a * -2.0, b * -2.0, c * -2.0) *
              helpers::kroneckerProduct(K2l, K2r) * std::exp(IM * globalPhase))
                 .isApprox(unitaryMatrix, SANITY_CHECK_PRECISION));
      auto isClose = [&](fp ap, fp bp, fp cp) -> bool {
        auto da = a - ap;
        auto db = b - bp;
        auto dc = c - cp;
        auto tr = static_cast<fp>(4.) *
                  qfp(std::cos(da) * std::cos(db) * std::cos(dc),
                      std::sin(da) * std::sin(db) * std::sin(dc));
        if (fidelity) {
          return traceToFid(tr) >= *fidelity;
        }
        return false;
      };

      auto closestAbc = closestPartialSwap(a, b, c);
      auto closestAbMinusC = closestPartialSwap(a, b, -c);
      auto flippedFromOriginal = false;

      auto getDefaultSpecialzation = [&]() {
        if (isClose(0., 0., 0.)) {
          return Specialization::IdEquiv;
        }
        if (isClose(qc::PI_4, qc::PI_4, qc::PI_4) ||
            isClose(qc::PI_4, qc::PI_4, -qc::PI_4)) {
          return Specialization::SWAPEquiv;
        }
        if (isClose(closestAbc, closestAbc, closestAbc)) {
          return Specialization::PartialSWAPEquiv;
        }
        if (isClose(closestAbMinusC, closestAbMinusC, -closestAbMinusC)) {
          return Specialization::PartialSWAPFlipEquiv;
        }
        if (isClose(a, 0., 0.)) {
          return Specialization::ControlledEquiv;
        }
        if (isClose(qc::PI_4, qc::PI_4, c)) {
          return Specialization::MirrorControlledEquiv;
        }
        if (isClose((a + b) / 2., (a + b) / 2., c)) {
          return Specialization::FSimaabEquiv;
        }
        if (isClose(a, (b + c) / 2., (b + c) / 2.)) {
          return Specialization::FSimabbEquiv;
        }
        if (isClose(a, (b - c) / 2., (c - b) / 2.)) {
          return Specialization::FSimabmbEquiv;
        }
        return Specialization::General;
      };
      auto actualSpecialization =
          specialization.value_or(getDefaultSpecialzation());
      TwoQubitWeylDecomposition general{
          .a = a,
          .b = b,
          .c = c,
          .globalPhase = globalPhase,
          .k1l = K1l,
          .k2l = K2l,
          .k1r = K1r,
          .k2r = K2r,
          .specialization = Specialization::General,
          .defaultEulerBasis = defaultEulerBasis,
          .requestedFidelity = fidelity,
          .calculatedFidelity = -1.0,
          .unitaryMatrix = unitaryMatrix,
      };
      auto getSpecializedDecomposition = [&]() {
        // :math:`U \sim U_d(0,0,0) \sim Id`
        //
        // This gate binds 0 parameters, we make it canonical by
        // setting
        // :math:`K2_l = Id` , :math:`K2_r = Id`.
        if (actualSpecialization == Specialization::IdEquiv) {
          return TwoQubitWeylDecomposition{
              .a = 0.,
              .b = 0.,
              .c = 0.,
              .globalPhase = general.globalPhase,
              .k1l = general.k1l * general.k2l,
              .k2l = IDENTITY_GATE,
              .k1r = general.k1r * general.k2r,
              .k2r = IDENTITY_GATE,
              .specialization = actualSpecialization,
              .defaultEulerBasis = general.defaultEulerBasis,
              .requestedFidelity = general.requestedFidelity,
              .calculatedFidelity = general.calculatedFidelity,
              .unitaryMatrix = general.unitaryMatrix,
          };
        }
        // :math:`U \sim U_d(\pi/4, \pi/4, \pi/4) \sim U(\pi/4, \pi/4,
        // -\pi/4) \sim \text{SWAP}`
        //
        // This gate binds 0 parameters, we make it canonical by
        // setting
        // :math:`K2_l = Id` , :math:`K2_r = Id`.
        if (actualSpecialization == Specialization::SWAPEquiv) {
          if (c > 0.) {
            return TwoQubitWeylDecomposition{
                .a = qc::PI_4,
                .b = qc::PI_4,
                .c = qc::PI_4,
                .globalPhase = general.globalPhase,
                .k1l = general.k1l * general.k2r,
                .k2l = IDENTITY_GATE,
                .k1r = general.k1r * general.k2l,
                .k2r = IDENTITY_GATE,
                .specialization = actualSpecialization,
                .defaultEulerBasis = general.defaultEulerBasis,
                .requestedFidelity = general.requestedFidelity,
                .calculatedFidelity = general.calculatedFidelity,
                .unitaryMatrix = general.unitaryMatrix,
            };
          }
          flippedFromOriginal = true;
          return TwoQubitWeylDecomposition{
              .a = qc::PI_4,
              .b = qc::PI_4,
              .c = qc::PI_4,
              .globalPhase = globalPhase + qc::PI_2,
              .k1l = general.k1l * IPZ * general.k2r,
              .k2l = IDENTITY_GATE,
              .k1r = general.k1r * IPZ * general.k2l,
              .k2r = IDENTITY_GATE,
              .specialization = actualSpecialization,
              .defaultEulerBasis = general.defaultEulerBasis,
              .requestedFidelity = general.requestedFidelity,
              .calculatedFidelity = general.calculatedFidelity,
              .unitaryMatrix = general.unitaryMatrix,
          };
        }
        // :math:`U \sim U_d(\alpha\pi/4, \alpha\pi/4, \alpha\pi/4) \sim
        // \text{SWAP}^\alpha`
        //
        // This gate binds 3 parameters, we make it canonical by setting:
        //
        // :math:`K2_l = Id`.
        if (actualSpecialization == Specialization::PartialSWAPEquiv) {
          auto closest = closestPartialSwap(a, b, c);
          auto k2lDag = general.k2l.transpose().conjugate();

          return TwoQubitWeylDecomposition{
              .a = closest,
              .b = closest,
              .c = closest,
              .globalPhase = general.globalPhase,
              .k1l = general.k1l * general.k2l,
              .k2l = IDENTITY_GATE,
              .k1r = general.k1r * general.k2l,
              .k2r = k2lDag * general.k2r,
              .specialization = actualSpecialization,
              .defaultEulerBasis = general.defaultEulerBasis,
              .requestedFidelity = general.requestedFidelity,
              .calculatedFidelity = general.calculatedFidelity,
              .unitaryMatrix = general.unitaryMatrix,
          };
        }
        // :math:`U \sim U_d(\alpha\pi/4, \alpha\pi/4, -\alpha\pi/4) \sim
        // \text{SWAP}^\alpha`
        //
        // (a non-equivalent root of SWAP from the
        // TwoQubitWeylPartialSWAPEquiv similar to how :math:`x = (\pm
        // \sqrt(x))^2`)
        //
        // This gate binds 3 parameters, we make it canonical by setting:
        //
        // :math:`K2_l = Id`
        if (actualSpecialization == Specialization::PartialSWAPFlipEquiv) {
          auto closest = closestPartialSwap(a, b, -c);
          auto k2lDag = general.k2l.transpose().conjugate();

          return TwoQubitWeylDecomposition{
              .a = closest,
              .b = closest,
              .c = -closest,
              .globalPhase = general.globalPhase,
              .k1l = general.k1l * general.k2l,
              .k2l = IDENTITY_GATE,
              .k1r = general.k1r * IPZ * general.k2l * IPZ,
              .k2r = IPZ * k2lDag * IPZ * general.k2r,
              .specialization = actualSpecialization,
              .defaultEulerBasis = general.defaultEulerBasis,
              .requestedFidelity = general.requestedFidelity,
              .calculatedFidelity = general.calculatedFidelity,
              .unitaryMatrix = general.unitaryMatrix,
          };
        }
        // :math:`U \sim U_d(\alpha, 0, 0) \sim \text{Ctrl-U}`
        //
        // This gate binds 4 parameters, we make it canonical by setting:
        //
        //      :math:`K2_l = Ry(\theta_l) Rx(\lambda_l)` ,
        //      :math:`K2_r = Ry(\theta_r) Rx(\lambda_r)` .
        if (actualSpecialization == Specialization::ControlledEquiv) {
          auto eulerBasis = EulerBasis::XYX;
          auto [k2ltheta, k2lphi, k2llambda, k2lphase] =
              anglesFromUnitary(general.k2l, eulerBasis);
          auto [k2rtheta, k2rphi, k2rlambda, k2rphase] =
              anglesFromUnitary(general.k2r, eulerBasis);
          return TwoQubitWeylDecomposition{
              .a = a,
              .b = 0.,
              .c = 0.,
              .globalPhase = globalPhase + k2lphase + k2rphase,
              .k1l = general.k1l * rxMatrix(k2lphi),
              .k2l = ryMatrix(k2ltheta) * rxMatrix(k2llambda),
              .k1r = general.k1r * rxMatrix(k2rphi),
              .k2r = ryMatrix(k2rtheta) * rxMatrix(k2rlambda),
              .specialization = actualSpecialization,
              .defaultEulerBasis = eulerBasis,
              .requestedFidelity = general.requestedFidelity,
              .calculatedFidelity = general.calculatedFidelity,
              .unitaryMatrix = general.unitaryMatrix,
          };
        }
        // :math:`U \sim U_d(\pi/4, \pi/4, \alpha) \sim \text{SWAP} \cdot
        // \text{Ctrl-U}`
        //
        // This gate binds 4 parameters, we make it canonical by setting:
        //
        // :math:`K2_l = Ry(\theta_l)\cdot Rz(\lambda_l)` , :math:`K2_r =
        // Ry(\theta_r)\cdot Rz(\lambda_r)`
        if (actualSpecialization == Specialization::MirrorControlledEquiv) {
          auto [k2ltheta, k2lphi, k2llambda, k2lphase] =
              anglesFromUnitary(general.k2l, EulerBasis::ZYZ);
          auto [k2rtheta, k2rphi, k2rlambda, k2rphase] =
              anglesFromUnitary(general.k2r, EulerBasis::ZYZ);
          return TwoQubitWeylDecomposition{
              .a = qc::PI_4,
              .b = qc::PI_4,
              .c = c,
              .globalPhase = globalPhase + k2lphase + k2rphase,
              .k1l = general.k1l * rzMatrix(k2rphi),
              .k2l = ryMatrix(k2ltheta) * rzMatrix(k2llambda),
              .k1r = general.k1r * rzMatrix(k2lphi),
              .k2r = ryMatrix(k2rtheta) * rzMatrix(k2rlambda),
              .specialization = actualSpecialization,
              .defaultEulerBasis = general.defaultEulerBasis,
              .requestedFidelity = general.requestedFidelity,
              .calculatedFidelity = general.calculatedFidelity,
              .unitaryMatrix = general.unitaryMatrix,
          };
        }
        // :math:`U \sim U_d(\alpha, \alpha, \beta), \alpha \geq |\beta|`
        //
        // This gate binds 5 parameters, we make it canonical by setting:
        //
        // :math:`K2_l = Ry(\theta_l)\cdot Rz(\lambda_l)`.
        if (actualSpecialization == Specialization::FSimaabEquiv) {
          auto [k2ltheta, k2lphi, k2llambda, k2lphase] =
              anglesFromUnitary(general.k2l, EulerBasis::ZYZ);
          return TwoQubitWeylDecomposition{
              .a = (a + b) / 2.,
              .b = (a + b) / 2.,
              .c = c,
              .globalPhase = globalPhase + k2lphase,
              .k1l = general.k1l * rzMatrix(k2lphi),
              .k2l = ryMatrix(k2ltheta) * rzMatrix(k2llambda),
              .k1r = general.k1r * rzMatrix(k2lphi),
              .k2r = rzMatrix(-k2lphi) * general.k2r,
              .specialization = actualSpecialization,
              .defaultEulerBasis = general.defaultEulerBasis,
              .requestedFidelity = general.requestedFidelity,
              .calculatedFidelity = general.calculatedFidelity,
              .unitaryMatrix = general.unitaryMatrix,
          };
        }
        // :math:`U \sim U_d(\alpha, \beta, -\beta), \alpha \geq \beta \geq 0`
        //
        // This gate binds 5 parameters, we make it canonical by setting:
        //
        // :math:`K2_l = Ry(\theta_l)Rx(\lambda_l)`
        if (actualSpecialization == Specialization::FSimabbEquiv) {
          auto eulerBasis = EulerBasis::XYX;
          auto [k2ltheta, k2lphi, k2llambda, k2lphase] =
              anglesFromUnitary(general.k2l, eulerBasis);
          return TwoQubitWeylDecomposition{
              .a = a,
              .b = (b + c) / 2.,
              .c = (b + c) / 2.,
              .globalPhase = globalPhase + k2lphase,
              .k1l = general.k1l * rxMatrix(k2lphi),
              .k2l = ryMatrix(k2ltheta) * rxMatrix(k2llambda),
              .k1r = general.k1r * rxMatrix(k2lphi),
              .k2r = rxMatrix(-k2lphi) * general.k2r,
              .specialization = actualSpecialization,
              .defaultEulerBasis = eulerBasis,
              .requestedFidelity = general.requestedFidelity,
              .calculatedFidelity = general.calculatedFidelity,
              .unitaryMatrix = general.unitaryMatrix,
          };
        }
        // :math:`U \sim U_d(\alpha, \beta, -\beta), \alpha \geq \beta \geq 0`
        //
        // This gate binds 5 parameters, we make it canonical by setting:
        //
        // :math:`K2_l = Ry(\theta_l)Rx(\lambda_l)`
        if (actualSpecialization == Specialization::FSimabmbEquiv) {
          auto eulerBasis = EulerBasis::XYX;
          auto [k2ltheta, k2lphi, k2llambda, k2lphase] =
              anglesFromUnitary(general.k2l, eulerBasis);
          return TwoQubitWeylDecomposition{
              .a = a,
              .b = (b - c) / 2.,
              .c = -((b - c) / 2.),
              .globalPhase = globalPhase + k2lphase,
              .k1l = general.k1l * rxMatrix(k2lphi),
              .k2l = ryMatrix(k2ltheta) * rxMatrix(k2llambda),
              .k1r = general.k1r * IPZ * rxMatrix(k2lphi) * IPZ,
              .k2r = IPZ * rxMatrix(-k2lphi) * IPZ * general.k2r,
              .specialization = actualSpecialization,
              .defaultEulerBasis = eulerBasis,
              .requestedFidelity = general.requestedFidelity,
              .calculatedFidelity = general.calculatedFidelity,
              .unitaryMatrix = general.unitaryMatrix,
          };
        }
        // U has no special symmetry.
        //
        // This gate binds all 6 possible parameters, so there is no need to
        // make the single-qubit pre-/post-gates canonical.
        if (actualSpecialization == Specialization::General) {
          return general;
        }
        throw std::logic_error{"Unknown specialization"};
      };

      TwoQubitWeylDecomposition specialized = getSpecializedDecomposition();

      auto getTr = [&]() {
        if (flippedFromOriginal) {
          auto [da, db, dc] = std::array{
              qc::PI_2 - a - specialized.a,
              b - specialized.b,
              -c - specialized.c,
          };
          return static_cast<fp>(4.) *
                 qfp(std::cos(da) * std::cos(db) * std::cos(dc),
                     std::sin(da) * std::sin(db) * std::sin(dc));
        }
        auto [da, db, dc] =
            std::array{a - specialized.a, b - specialized.b, c - specialized.c};
        return static_cast<fp>(4.) *
               qfp(std::cos(da) * std::cos(db) * std::cos(dc),
                   std::sin(da) * std::sin(db) * std::sin(dc));
      };
      auto tr = getTr();
      specialized.calculatedFidelity = traceToFid(tr);
      if (specialized.requestedFidelity) {
        if (specialized.calculatedFidelity + 1.0e-13 <
            *specialized.requestedFidelity) {
          throw std::runtime_error{
              "Specialization: {:?} calculated fidelity: {} is worse than "
              "requested fidelity: {}",
          };
        }
      }
      specialized.globalPhase += std::arg(tr);

      helpers::print(
          (helpers::kroneckerProduct(specialized.k1l, specialized.k1r) *
           getCanonicalMatrix(specialized.a * -2.0, specialized.b * -2.0,
                              specialized.c * -2.0) *
           helpers::kroneckerProduct(specialized.k2l, specialized.k2r) *
           std::exp(IM * specialized.globalPhase))
              .eval(),
          "SANITY CHECK (3)", true);
      assert((helpers::kroneckerProduct(specialized.k1l, specialized.k1r) *
              getCanonicalMatrix(specialized.a * -2.0, specialized.b * -2.0,
                                 specialized.c * -2.0) *
              helpers::kroneckerProduct(specialized.k2l, specialized.k2r) *
              std::exp(IM * specialized.globalPhase))
                 .isApprox(unitaryMatrix, SANITY_CHECK_PRECISION));

      return specialized;
    }
  };

  static constexpr auto DEFAULT_FIDELITY = 1.0 - 1.0e-9;

  struct TwoQubitBasisDecomposer {
    QubitGateSequence::Gate basisGate;
    fp basisFidelity;
    EulerBasis eulerBasis;
    TwoQubitWeylDecomposition basisDecomposer;
    bool superControlled;
    matrix2x2 u0l;
    matrix2x2 u0r;
    matrix2x2 u1l;
    matrix2x2 u1ra;
    matrix2x2 u1rb;
    matrix2x2 u2la;
    matrix2x2 u2lb;
    matrix2x2 u2ra;
    matrix2x2 u2rb;
    matrix2x2 u3l;
    matrix2x2 u3r;
    matrix2x2 q0l;
    matrix2x2 q0r;
    matrix2x2 q1la;
    matrix2x2 q1lb;
    matrix2x2 q1ra;
    matrix2x2 q1rb;
    matrix2x2 q2l;
    matrix2x2 q2r;

  public:
    static TwoQubitBasisDecomposer
    newInner(const OneQubitGateSequence::Gate& basisGate = {.type = qc::X,
                                                            .parameter = {},
                                                            .qubitId = {0, 1}},
             fp basisFidelity = DEFAULT_FIDELITY,
             EulerBasis eulerBasis = EulerBasis::ZYZ) {
      auto relativeEq = [](auto&& lhs, auto&& rhs, auto&& epsilon,
                           auto&& maxRelative) {
        // Handle same infinities
        if (lhs == rhs) {
          return true;
        }

        // Handle remaining infinities
        if (std::isinf(lhs) || std::isinf(rhs)) {
          return false;
        }

        auto absDiff = std::abs(lhs - rhs);

        // For when the numbers are really close together
        if (absDiff <= epsilon) {
          return true;
        }

        auto absLhs = std::abs(lhs);
        auto absRhs = std::abs(rhs);
        if (absRhs > absLhs) {
          return absDiff <= absRhs * maxRelative;
        }
        return absDiff <= absLhs * maxRelative;
      };
      const matrix2x2 k12RArr{
          {qfp(0., FRAC1_SQRT2), qfp(FRAC1_SQRT2, 0.)},
          {qfp(-FRAC1_SQRT2, 0.), qfp(0., -FRAC1_SQRT2)},
      };
      const matrix2x2 k12LArr{
          {qfp(0.5, 0.5), qfp(0.5, 0.5)},
          {qfp(-0.5, 0.5), qfp(0.5, -0.5)},
      };

      auto basisDecomposer = TwoQubitWeylDecomposition::newInner(
          getTwoQubitMatrix(basisGate), basisFidelity, std::nullopt);
      auto superControlled =
          relativeEq(basisDecomposer.a, qc::PI_4, 1e-13, 1e-09) &&
          relativeEq(basisDecomposer.c, 0.0, 1e-13, 1e-09);

      // Create some useful matrices U1, U2, U3 are equivalent to the basis,
      // expand as Ui = Ki1.Ubasis.Ki2
      auto b = basisDecomposer.b;
      auto temp = qfp(0.5, -0.5);
      matrix2x2 k11l{
          {temp * (M_IM * std::exp(qfp(0., -b))), temp * std::exp(qfp(0., -b))},
          {temp * (M_IM * std::exp(qfp(0., b))), temp * -std::exp(qfp(0., b))}};
      matrix2x2 k11r{{FRAC1_SQRT2 * (IM * std::exp(qfp(0., -b))),
                      FRAC1_SQRT2 * -std::exp(qfp(0., -b))},
                     {FRAC1_SQRT2 * std::exp(qfp(0., b)),
                      FRAC1_SQRT2 * (M_IM * std::exp(qfp(0., b)))}};
      matrix2x2 k32lK21l{{FRAC1_SQRT2 * qfp(1., std::cos(2. * b)),
                          FRAC1_SQRT2 * (IM * std::sin(2. * b))},
                         {FRAC1_SQRT2 * (IM * std::sin(2. * b)),
                          FRAC1_SQRT2 * qfp(1., -std::cos(2. * b))}};
      temp = qfp(0.5, 0.5);
      matrix2x2 k21r{
          {temp * (M_IM * std::exp(qfp(0., -2. * b))),
           temp * std::exp(qfp(0., -2. * b))},
          {temp * (IM * std::exp(qfp(0., 2. * b))),
           temp * std::exp(qfp(0., 2. * b))},
      };
      const matrix2x2 k22LArr{
          {qfp(FRAC1_SQRT2, 0.), qfp(-FRAC1_SQRT2, 0.)},
          {qfp(FRAC1_SQRT2, 0.), qfp(FRAC1_SQRT2, 0.)},
      };
      const matrix2x2 k22RArr{{C_ZERO, C_ONE}, {C_M_ONE, C_ZERO}};
      matrix2x2 k31l{
          {FRAC1_SQRT2 * std::exp(qfp(0., -b)),
           FRAC1_SQRT2 * std::exp(qfp(0., -b))},
          {FRAC1_SQRT2 * -std::exp(qfp(0., b)),
           FRAC1_SQRT2 * std::exp(qfp(0., b))},
      };
      matrix2x2 k31r{
          {IM * std::exp(qfp(0., b)), C_ZERO},
          {C_ZERO, M_IM * std::exp(qfp(0., -b))},
      };
      temp = qfp(0.5, 0.5);
      matrix2x2 k32r{
          {temp * std::exp(qfp(0., b)), temp * -std::exp(qfp(0., -b))},
          {temp * (M_IM * std::exp(qfp(0., b))),
           temp * (M_IM * std::exp(qfp(0., -b)))},
      };
      auto k1ld = basisDecomposer.k1l.transpose().conjugate();
      auto k1rd = basisDecomposer.k1r.transpose().conjugate();
      auto k2ld = basisDecomposer.k2l.transpose().conjugate();
      auto k2rd = basisDecomposer.k2r.transpose().conjugate();
      // Pre-build the fixed parts of the matrices used in 3-part
      // decomposition
      auto u0l = k31l * k1ld;
      auto u0r = k31r * k1rd;
      auto u1l = k2ld * k32lK21l * k1ld;
      auto u1ra = k2rd * k32r;
      auto u1rb = k21r * k1rd;
      auto u2la = k2ld * k22LArr;
      auto u2lb = k11l * k1ld;
      auto u2ra = k2rd * k22RArr;
      auto u2rb = k11r * k1rd;
      auto u3l = k2ld * k12LArr;
      auto u3r = k2rd * k12RArr;
      // Pre-build the fixed parts of the matrices used in the 2-part
      // decomposition
      auto q0l = k12LArr.transpose().conjugate() * k1ld;
      auto q0r = k12RArr.transpose().conjugate() * IPZ * k1rd;
      auto q1la = k2ld * k11l.transpose().conjugate();
      auto q1lb = k11l * k1ld;
      auto q1ra = k2rd * IPZ * k11r.transpose().conjugate();
      auto q1rb = k11r * k1rd;
      auto q2l = k2ld * k12LArr;
      auto q2r = k2rd * k12RArr;

      return TwoQubitBasisDecomposer{
          .basisGate = basisGate,
          .basisFidelity = basisFidelity,
          .eulerBasis = eulerBasis,
          .basisDecomposer = basisDecomposer,
          .superControlled = superControlled,
          .u0l = u0l,
          .u0r = u0r,
          .u1l = u1l,
          .u1ra = u1ra,
          .u1rb = u1rb,
          .u2la = u2la,
          .u2lb = u2lb,
          .u2ra = u2ra,
          .u2rb = u2rb,
          .u3l = u3l,
          .u3r = u3r,
          .q0l = q0l,
          .q0r = q0r,
          .q1la = q1la,
          .q1lb = q1lb,
          .q1ra = q1ra,
          .q1rb = q1rb,
          .q2l = q2l,
          .q2r = q2r,
      };
    }

    std::optional<TwoQubitGateSequence>
    twoQubitDecompose(const matrix4x4& unitaryMatrix,
                      std::optional<fp> basisFidelity, bool approximate,
                      std::optional<std::uint8_t> numBasisGateUses) {
      auto getBasisFidelity = [&]() {
        if (approximate) {
          return basisFidelity.value_or(this->basisFidelity);
        }
        return static_cast<fp>(1.0);
      };
      fp actualBasisFidelity = getBasisFidelity();
      auto targetDecomposed = TwoQubitWeylDecomposition::newInner(
          unitaryMatrix, actualBasisFidelity, std::nullopt);
      auto traces = this->traces(targetDecomposed);
      auto getDefaultNbasis = [&]() {
        auto minValue = std::numeric_limits<fp>::min();
        auto minIndex = -1;
        for (int i = 0; i < static_cast<int>(traces.size()); ++i) {
          auto value = traceToFid(traces[i]) * std::pow(actualBasisFidelity, i);
          if (value > minValue) {
            minIndex = i;
            minValue = value;
          }
        }
        return minIndex;
      };
      // number of basis gates that need to be inserted
      auto bestNbasis = numBasisGateUses.value_or(getDefaultNbasis());
      auto chooseDecomposition = [&]() {
        if (bestNbasis == 0) {
          return decomp0Inner(targetDecomposed);
        }
        if (bestNbasis == 1) {
          return decomp1Inner(targetDecomposed);
        }
        if (bestNbasis == 2) {
          return decomp2SupercontrolledInner(targetDecomposed);
        }
        if (bestNbasis == 3) {
          return decomp3SupercontrolledInner(targetDecomposed);
        }
        throw std::logic_error{"Invalid basis to use"};
      };
      auto decomposition = chooseDecomposition();
      std::cerr << "NBasis: " << static_cast<int>(bestNbasis)
                << "; basis_fid: " << actualBasisFidelity
                << "; Traces: " << traces[0] << ", " << traces[1] << ", "
                << traces[2] << ", " << traces[3];
      std::cerr << "\nDecompositions:\n";
      for (auto x : decomposition) {
        helpers::print(x, "", true);
      }
      std::vector<EulerBasis> target1qBasisList; // TODO: simplify because list
                                                 // only has one element?
      target1qBasisList.push_back(eulerBasis);
      llvm::SmallVector<std::optional<TwoQubitGateSequence>, 8>
          eulerDecompositions;
      for (auto&& decomp : decomposition) {
        assert(helpers::isUnitaryMatrix(decomp));
        auto eulerDecomp = unitaryToGateSequenceInner(
            decomp, target1qBasisList, 0, {}, true, 1.0 - actualBasisFidelity);
        eulerDecompositions.push_back(eulerDecomp);
      }
      TwoQubitGateSequence gates{.globalPhase = targetDecomposed.globalPhase};
      // Worst case length is 5x 1q gates for each 1q decomposition + 1x 2q
      // gate We might overallocate a bit if the euler basis is different but
      // the worst case is just 16 extra elements with just a String and 2
      // smallvecs each. This is only transient though as the circuit
      // sequences aren't long lived and are just used to create a
      // QuantumCircuit or DAGCircuit when we return to Python space.
      constexpr auto twoQubitSequenceDefaultCapacity = 21;
      gates.gates.reserve(twoQubitSequenceDefaultCapacity);
      gates.globalPhase -= bestNbasis * basisDecomposer.globalPhase;
      if (bestNbasis == 2) {
        gates.globalPhase += qc::PI;
      }

      auto addEulerDecomposition = [&](std::size_t index, std::size_t qubitId) {
        if (auto&& eulerDecomp = eulerDecompositions[index]) {
          for (auto&& gate : eulerDecomp->gates) {
            gates.gates.push_back({.type = gate.type,
                                   .parameter = gate.parameter,
                                   .qubitId = {qubitId}});
          }
          gates.globalPhase += eulerDecomp->globalPhase;
        }
      };

      // TODO: check if this actually uses the correct qubitIds
      for (std::size_t i = 0; i < bestNbasis; ++i) {
        addEulerDecomposition(2 * i, 0);
        addEulerDecomposition((2 * i) + 1, 1);

        gates.gates.push_back(basisGate);
      }

      addEulerDecomposition(2UL * bestNbasis, 0);
      addEulerDecomposition((2UL * bestNbasis) + 1, 1);

      // large global phases can be generated by the decomposition, thus limit
      // it to (-2*pi, +2*pi); TODO: can be removed, should be done by something
      // like constant folding
      gates.globalPhase = std::fmod(gates.globalPhase, qc::TAU);

      return gates;
    }

  private:
    [[nodiscard]] static std::vector<matrix2x2>
    decomp0Inner(const TwoQubitWeylDecomposition& target) {
      return {
          target.k1r * target.k2r,
          target.k1l * target.k2l,
      };
    }

    [[nodiscard]] std::vector<matrix2x2>
    decomp1Inner(const TwoQubitWeylDecomposition& target) const {
      // FIXME: fix for z!=0 and c!=0 using closest reflection (not always in
      // the Weyl chamber)
      return {
          basisDecomposer.k2r.transpose().conjugate() * target.k2r,
          basisDecomposer.k2l.transpose().conjugate() * target.k2l,
          target.k1r * basisDecomposer.k1r.transpose().conjugate(),
          target.k1l * basisDecomposer.k1l.transpose().conjugate(),
      };
    }

    [[nodiscard]] std::vector<matrix2x2>
    decomp2SupercontrolledInner(const TwoQubitWeylDecomposition& target) const {
      return {
          q2r * target.k2r,
          q2l * target.k2l,
          q1ra * rzMatrix(2. * target.b) * q1rb,
          q1la * rzMatrix(-2. * target.a) * q1lb,
          target.k1r * q0r,
          target.k1l * q0l,
      };
    }

    [[nodiscard]] std::vector<matrix2x2>
    decomp3SupercontrolledInner(const TwoQubitWeylDecomposition& target) const {
      return {
          u3r * target.k2r,
          u3l * target.k2l,
          u2ra * rzMatrix(2. * target.b) * u2rb,
          u2la * rzMatrix(-2. * target.a) * u2lb,
          u1ra * rzMatrix(-2. * target.c) * u1rb,
          u1l,
          target.k1r * u0r,
          target.k1l * u0l,
      };
    }

    [[nodiscard]] std::array<qfp, 4>
    traces(TwoQubitWeylDecomposition target) const {
      return {
          static_cast<fp>(4.) *
              qfp(std::cos(target.a) * std::cos(target.b) * std::cos(target.c),
                  std::sin(target.a) * std::sin(target.b) * std::sin(target.c)),
          static_cast<fp>(4.) * qfp(std::cos(qc::PI_4 - target.a) *
                                        std::cos(basisDecomposer.b - target.b) *
                                        std::cos(target.c),
                                    std::sin(qc::PI_4 - target.a) *
                                        std::sin(basisDecomposer.b - target.b) *
                                        std::sin(target.c)),
          qfp(4. * std::cos(target.c), 0.),
          qfp(4., 0.),
      };
    }

    static OneQubitGateSequence generateCircuit(EulerBasis targetBasis,
                                                const matrix2x2& unitaryMatrix,
                                                bool simplify,
                                                std::optional<fp> atol) {
      auto [theta, phi, lambda, phase] =
          anglesFromUnitary(unitaryMatrix, targetBasis);

      switch (targetBasis) {
      case EulerBasis::ZYZ:
        return calculateRotationGates(theta, phi, lambda, phase, qc::RZ, qc::RY,
                                      simplify, atol);
      case EulerBasis::ZXZ:
        return calculateRotationGates(theta, phi, lambda, phase, qc::RZ, qc::RX,
                                      simplify, atol);
      case EulerBasis::XZX:
        return calculateRotationGates(theta, phi, lambda, phase, qc::RX, qc::RZ,
                                      simplify, atol);
      case EulerBasis::XYX:
        return calculateRotationGates(theta, phi, lambda, phase, qc::RX, qc::RY,
                                      simplify, atol);
      default:
        // TODO: allow other bases
        throw std::invalid_argument{"Unsupported base for circuit generation!"};
      }
    }

    static OneQubitGateSequence unitaryToGateSequenceInner(
        matrix2x2 unitaryMat, const std::vector<EulerBasis>& targetBasisList,
        std::size_t /*qubit*/,
        const std::vector<std::unordered_map<std::string, fp>>&
        /*error_map*/, // TODO: remove error_map+qubit for platform
                       // independence
        bool simplify, std::optional<fp> atol) {
      auto calculateError = [](const OneQubitGateSequence& sequence) -> fp {
        return static_cast<fp>(sequence.complexity());
      };

      auto minError = std::numeric_limits<fp>::max();
      OneQubitGateSequence bestCircuit;
      for (auto targetBasis : targetBasisList) {
        auto circuit = generateCircuit(targetBasis, unitaryMat, simplify, atol);
        helpers::print(circuit.getUnitaryMatrix(), "SANITY CHECK (4.1)", true);
        helpers::print(helpers::kroneckerProduct(IDENTITY_GATE, unitaryMat),
                       "SANITY CHECK (4.2)", true);
        assert(circuit.getUnitaryMatrix().isApprox(
            helpers::kroneckerProduct(IDENTITY_GATE, unitaryMat),
            SANITY_CHECK_PRECISION));
        auto error = calculateError(circuit);
        if (error < minError) {
          bestCircuit = circuit;
          minError = error;
        }
      }
      return bestCircuit;
    }

    // TODO: copied+adapted from single-qubit decomposition
    /**
     * @note Adapted from circuit_kak() in the IBM Qiskit framework.
     *       (C) Copyright IBM 2022
     *
     *       This code is licensed under the Apache License, Version 2.0. You
     * may obtain a copy of this license in the LICENSE.txt file in the root
     *       directory of this source tree or at
     *       http://www.apache.org/licenses/LICENSE-2.0.
     *
     *       Any modifications or derivative works of this code must retain
     * this copyright notice, and modified files need to carry a notice
     *       indicating that they have been altered from the originals.
     */
    [[nodiscard]] static OneQubitGateSequence
    calculateRotationGates(fp theta, fp phi, fp lambda, fp phase,
                           qc::OpType kGate, qc::OpType aGate, bool simplify,
                           std::optional<fp> atol) {
      fp angleZeroEpsilon = atol.value_or(1e-12);
      if (!simplify) {
        angleZeroEpsilon = -1.0;
      }

      fp globalPhase = phase - ((phi + lambda) / 2.);

      std::vector<OneQubitGateSequence::Gate> gates;
      if (std::abs(theta) <= angleZeroEpsilon) {
        lambda += phi;
        lambda = mod2pi(lambda);
        if (std::abs(lambda) > angleZeroEpsilon) {
          gates.push_back({kGate, {lambda}});
          globalPhase += lambda / 2.0;
        }
        return {gates, globalPhase};
      }

      if (std::abs(theta - qc::PI) <= angleZeroEpsilon) {
        globalPhase += phi;
        lambda -= phi;
        phi = 0.0;
      }
      if (std::abs(mod2pi(lambda + qc::PI)) <= angleZeroEpsilon ||
          std::abs(mod2pi(phi + qc::PI)) <= angleZeroEpsilon) {
        lambda += qc::PI;
        theta = -theta;
        phi += qc::PI;
      }
      lambda = mod2pi(lambda);
      if (std::abs(lambda) > angleZeroEpsilon) {
        globalPhase += lambda / 2.0;
        gates.push_back({kGate, {lambda}});
      }
      gates.push_back({aGate, {theta}});
      phi = mod2pi(phi);
      if (std::abs(phi) > angleZeroEpsilon) {
        globalPhase += phi / 2.0;
        gates.push_back({kGate, {phi}});
      }
      return {gates, globalPhase};
    }
  };

private:
  llvm::SmallVector<QubitGateSequence::Gate> decomposerBasisGate;
  llvm::SmallVector<EulerBasis> decomposerEulerBasis;
};

const matrix2x2 GateDecompositionPattern::IDENTITY_GATE = matrix2x2::Identity();
const matrix2x2 GateDecompositionPattern::H_GATE{{1.0 / SQRT2, 1.0 / SQRT2},
                                                 {1.0 / SQRT2, -1.0 / SQRT2}};
const matrix2x2 GateDecompositionPattern::IPZ{{IM, C_ZERO}, {C_ZERO, M_IM}};
const matrix2x2 GateDecompositionPattern::IPY{{C_ZERO, C_ONE},
                                              {C_M_ONE, C_ZERO}};
const matrix2x2 GateDecompositionPattern::IPX{{C_ZERO, IM}, {IM, C_ZERO}};

/**
 * @brief Populates the given pattern set with patterns for gate
 * decomposition.
 */
void populateGateDecompositionPatterns(mlir::RewritePatternSet& patterns) {
  llvm::SmallVector<GateDecompositionPattern::QubitGateSequence::Gate> basisGates;
  llvm::SmallVector<GateDecompositionPattern::EulerBasis> eulerBases;
  basisGates.push_back({.type = qc::X, .parameter = {}, .qubitId = {0, 1}});
  basisGates.push_back({.type = qc::X, .parameter = {}, .qubitId = {1, 0}});
  eulerBases.push_back(GateDecompositionPattern::EulerBasis::ZYZ);
  patterns.add<GateDecompositionPattern>(patterns.getContext(), basisGates,
                                         eulerBases);
}

} // namespace mqt::ir::opt

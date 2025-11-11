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

#include <array>
#include <cstddef>
#include <eigen3/unsupported/Eigen/MatrixFunctions>
#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <string>

namespace mqt::ir::opt {

/**
 * @brief This pattern attempts to cancel consecutive self-inverse operations.
 */
struct GateDecompositionPattern final
    : mlir::OpInterfaceRewritePattern<UnitaryInterface> {

  explicit GateDecompositionPattern(mlir::MLIRContext* context)
      : OpInterfaceRewritePattern(context) {}

  mlir::LogicalResult
  matchAndRewrite(UnitaryInterface op,
                  mlir::PatternRewriter& rewriter) const override {
    auto series = TwoQubitSeries::getTwoQubitSeries(op);
    llvm::errs() << "SERIES SIZE: " << series.gates.size() << '\n';
    for (auto&& gate : series.gates) {
      std::cerr << gate.op->getName().stripDialect().str() << ", ";
    }
    std::cerr << '\n';
    static int a{};
    if (a++ > 0) {
      return mlir::failure();
    }

    if (series.gates.size() < 3) {
      // too short
      return mlir::failure();
    }
    if (llvm::is_contained(series.inQubits, mlir::Value{}) ||
        llvm::is_contained(series.outQubits, mlir::Value{})) {
      // only a single-qubit series
      return mlir::failure();
    }

    matrix4x4 unitaryMatrix =
        helpers::kroneckerProduct(identityGate, identityGate);
    int i{};
    for (auto&& gate : series.gates) {
      auto gateMatrix =
          getTwoQubitMatrix({.type = helpers::getQcType(gate.op),
                             .parameter = helpers::getParameters(gate.op),
                             .qubit_id = gate.qubitIds});
      unitaryMatrix = gateMatrix * unitaryMatrix;
      helpers::print(gateMatrix, "GATE MATRIX " + std::to_string(i++), true);
    }
    helpers::print(unitaryMatrix, "UNITARY MATRIX", true);

    auto decomposer = TwoQubitBasisDecomposer::new_inner();
    auto sequence = decomposer.twoQubitDecompose(
        unitaryMatrix, DEFAULT_FIDELITY, true, std::nullopt);
    if (!sequence) {
      llvm::errs() << "NO SEQUENCE GENERATED!\n";
      return mlir::failure();
    }
    if (sequence->complexity() >= series.complexity) {
      // TODO: add more sophisticated metric to determine complexity of
      // series/sequence
      llvm::errs() << "SEQUENCE LONGER THAN INPUT (" << sequence->gates.size()
                   << ")\n";
      return mlir::failure();
    }

    applySeries(rewriter, series, *sequence);

    return mlir::success();
  }

  struct TwoQubitSeries {
    std::size_t complexity{0};
    std::array<mlir::Value, 2> inQubits;
    std::array<mlir::Value, 2> outQubits;

    struct Gate {
      UnitaryInterface op;
      llvm::SmallVector<std::size_t, 2> qubitIds;
    };
    llvm::SmallVector<Gate, 8> gates;

    [[nodiscard]] static TwoQubitSeries getTwoQubitSeries(UnitaryInterface op) {
      TwoQubitSeries result(op);

      auto findNextInSeries = [&](mlir::Operation* user) {
        auto userUnitary = mlir::dyn_cast<UnitaryInterface>(user);
        if (!userUnitary) {
          return false;
        }

        if (helpers::isSingleQubitOperation(userUnitary)) {
          return result.appendSingleQubitGate(userUnitary);
        }
        if (helpers::isTwoQubitOperation(userUnitary)) {
          return result.appendTwoQubitGate(userUnitary);
        }
        return false;
      };

      bool isFirstQubitOngoing = result.outQubits[0] != mlir::Value{};
      bool isSecondQubitOngoing = result.outQubits[1] != mlir::Value{};
      while (isFirstQubitOngoing || isSecondQubitOngoing) {
        // TODO: can cause issues; instead: per iteration, collect all
        // single-qubit operations, then take one two-qubit operation; repeat
        if (result.outQubits[0]) {
          assert(result.outQubits[0].hasOneUse());
          isFirstQubitOngoing =
              findNextInSeries(*result.outQubits[0].getUsers().begin());
        }
        if (result.outQubits[1]) {
          assert(result.outQubits[1].hasOneUse());
          isSecondQubitOngoing =
              findNextInSeries(*result.outQubits[1].getUsers().begin());
        }
      }
      return result;
    }

  private:
    explicit TwoQubitSeries(UnitaryInterface initialOperation) {
      auto&& in = initialOperation.getAllInQubits();
      auto&& out = initialOperation->getResults();
      if (helpers::isSingleQubitOperation(initialOperation)) {
        inQubits = {in[0], mlir::Value{}};
        outQubits = {out[0], mlir::Value{}};
        gates.push_back({initialOperation, {0}});
        complexity += 1;
      } else if (helpers::isTwoQubitOperation(initialOperation)) {
        inQubits = {in[0], in[1]};
        outQubits = {out[0], out[1]};
        gates.push_back({initialOperation, {0, 1}});
        complexity += 2;
      }
    }

    /**
     * @return true if series continues, otherwise false
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

      gates.push_back({nextGate, {qubitId}});
      complexity += 1;
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
      *firstQubitIt = nextGate->getResult(0);
      std::size_t firstQubitId = std::distance(outQubits.begin(), firstQubitIt);
      *secondQubitIt = nextGate->getResult(1);
      std::size_t secondQubitId =
          std::distance(outQubits.begin(), secondQubitIt);

      gates.push_back({nextGate, {firstQubitId, secondQubitId}});
      complexity += 2;
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
                           llvm::SmallVector<fp, 3> parameters) {
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

  struct QubitGateSequence {
    struct Gate {
      qc::OpType type{qc::I};
      llvm::SmallVector<fp, 3> parameter;
      llvm::SmallVector<std::size_t, 2> qubit_id = {0};
    };
    std::vector<Gate> gates;
    std::size_t complexity() {
      std::size_t c{};
      for (auto&& gate : gates) {
        c += gate.qubit_id.size();
      }
      return c;
    }
    fp globalPhase;
  };
  using OneQubitGateSequence = QubitGateSequence;
  using TwoQubitGateSequence = QubitGateSequence;

  static void applySeries(mlir::PatternRewriter& rewriter,
                          TwoQubitSeries& series,
                          const TwoQubitGateSequence& sequence) {
    auto& lastSeriesOp = series.gates.back().op;
    auto location = lastSeriesOp->getLoc();
    rewriter.setInsertionPointAfter(lastSeriesOp);

    if (sequence.globalPhase != 0.0) {
      createOneParameterGate<GPhaseOp>(rewriter, location, sequence.globalPhase,
                                       {});
    }

    auto inQubits = series.inQubits;
    auto updateInQubits =
        [&inQubits](const TwoQubitGateSequence::Gate& gateDescription,
                    auto&& newGate) {
          auto results = newGate.getAllOutQubits();
          if (gateDescription.qubit_id.size() == 2) {
            inQubits[gateDescription.qubit_id[0]] = results[0];
            inQubits[gateDescription.qubit_id[1]] = results[1];
          } else if (gateDescription.qubit_id.size() == 1) {
            inQubits[gateDescription.qubit_id[0]] = results[0];
          } else {
            throw std::logic_error{"Invalid number of qubit IDs!"};
          }
        };

    std::cerr << "SERIES: ";
    for (auto&& gate : series.gates) {
      std::cerr << gate.op->getName().stripDialect().str() << ", ";
    }
    std::cerr << '\n';
    std::cerr << "GATE SEQUENCE!: " << std::flush;
    for (auto&& gate : sequence.gates) {
      std::cerr << qc::toString(gate.type) << ", ";
      if (gate.type == qc::X) {
        mlir::SmallVector<mlir::Value, 1> inCtrlQubits;
        if (gate.qubit_id.size() > 1) {
          inCtrlQubits.push_back(inQubits[gate.qubit_id[1]]);
        }
        auto newGate = createGate<XOp>(rewriter, location, {inQubits[0]},
                                       inCtrlQubits, gate.parameter);
        updateInQubits(gate, newGate);
      } else if (gate.type == qc::RX) {
        mlir::SmallVector<mlir::Value, 2> qubits;
        for (auto&& x : gate.qubit_id) {
          qubits.push_back(inQubits[x]);
        }
        auto newGate =
            createGate<RXOp>(rewriter, location, qubits, {}, gate.parameter);
        updateInQubits(gate, newGate);
      } else if (gate.type == qc::RY) {
        mlir::SmallVector<mlir::Value, 2> qubits;
        for (auto&& x : gate.qubit_id) {
          qubits.push_back(inQubits[x]);
        }
        auto newGate =
            createGate<RYOp>(rewriter, location, qubits, {}, gate.parameter);
        updateInQubits(gate, newGate);
      } else if (gate.type == qc::RZ) {
        mlir::SmallVector<mlir::Value, 2> qubits;
        for (auto&& x : gate.qubit_id) {
          qubits.push_back(inQubits[x]);
        }
        auto newGate =
            createGate<RZOp>(rewriter, location, qubits, {}, gate.parameter);
        updateInQubits(gate, newGate);
      } else {
        throw std::runtime_error{"Unknown gate type!"};
      }
    }

    rewriter.replaceAllUsesWith(series.outQubits, inQubits);
    for (auto&& gate : llvm::reverse(series.gates)) {
      rewriter.eraseOp(gate.op);
    }
  }

  enum class Specialization {
    General,
    IdEquiv,
    SWAPEquiv,
    PartialSWAPEquiv,
    PartialSWAPFlipEquiv,
    ControlledEquiv,
    MirrorControlledEquiv,
    // These next 3 gates use the definition of fSim from eq (1) in:
    // https://arxiv.org/pdf/2001.08343.pdf
    fSimaabEquiv,
    fSimabbEquiv,
    fSimabmbEquiv,
  };

  enum class MagicBasisTransform {
    Into,
    OutOf,
  };

  enum class EulerBasis {
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

  static constexpr auto sqrt2 = static_cast<fp>(1.4142135623730950488L);
  static const matrix2x2 identityGate;
  static const matrix2x2 hGate;

  static fp remEuclid(fp a, fp b) {
    auto r = std::fmod(a, b);
    return (r < 0.0) ? r + std::abs(b) : r;
  }

  // Wrap angle into interval [-π,π). If within atol of the endpoint, clamp
  // to -π
  static fp mod2pi(fp angle,
                   fp angleZeroEpsilon = std::numeric_limits<fp>::epsilon()) {
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
  static std::pair<rmatrix4x4, rdiagonal4x4>
  self_adjoint_eigen_lower(rmatrix4x4 A) {
    // rdiagonal4x4 S;
    // auto U = self_adjoint_evd(A, S);

    // rmatrix4x4 U;
    // jacobi_eigen_decomposition(A, U, S);

    auto [U, S] = helpers::self_adjoint_evd(A);

    // TODO: not in original code
    if (std::abs(U.determinant() + 1.0) < 1e-5) {
      std::cerr << "CORRECTION!\n";
      // if determinant of eigenvector matrix is -1.0, multiply first
      // eigenvector by -1.0
      U.col(0) *= -1.0;
    }

    return std::make_pair(U, S);
  }

  static std::tuple<matrix2x2, matrix2x2, fp>
  decompose_two_qubit_product_gate(matrix4x4 special_unitary) {
    helpers::print(special_unitary, "SPECIAL_UNITARY");
    // first quadrant
    matrix2x2 r{{special_unitary(0, 0), special_unitary(0, 1)},
                {special_unitary(1, 0), special_unitary(1, 1)}};
    auto det_r = r.determinant();
    if (std::abs(det_r) < 0.1) {
      // third quadrant
      r = matrix2x2{{special_unitary(2, 0), special_unitary(2, 1)},
                    {special_unitary(3, 0), special_unitary(3, 1)}};
      det_r = r.determinant();
    }
    std::cerr << "DET_R: " << det_r << '\n';
    if (std::abs(det_r) < 0.1) {
      throw std::runtime_error{
          "decompose_two_qubit_product_gate: unable to decompose: det_r < 0.1"};
    }
    r /= std::sqrt(det_r);
    helpers::print(r, "R");
    // transpose with complex conjugate of each element
    matrix2x2 r_t_conj = r.transpose().conjugate();

    auto temp = helpers::kroneckerProduct(identityGate, r_t_conj);
    helpers::print(temp, "TEMP (decompose_two_qubit_product_gate, 1)");
    temp = special_unitary * temp;
    helpers::print(temp, "TEMP (decompose_two_qubit_product_gate, 2)");

    // [[a, b, c, d],
    //  [e, f, g, h], => [[a, c],
    //  [i, j, k, l],     [i, k]]
    //  [m, n, o, p]]
    matrix2x2 l{{temp(0, 0), temp(0, 2)}, {temp(2, 0), temp(2, 2)}};
    auto det_l = l.determinant();
    if (std::abs(det_l) < 0.9) {
      throw std::runtime_error{
          "decompose_two_qubit_product_gate: unable to decompose: detL < 0.9"};
    }
    l /= std::sqrt(det_l);
    auto phase = std::arg(det_l) / 2.;

    return {l, r, phase};
  }

  static matrix4x4 magic_basis_transform(const matrix4x4& unitary,
                                         MagicBasisTransform direction) {
    const matrix4x4 B_NON_NORMALIZED{
        {C_ONE, IM, C_ZERO, C_ZERO},
        {C_ZERO, C_ZERO, IM, C_ONE},
        {C_ZERO, C_ZERO, IM, C_M_ONE},
        {C_ONE, M_IM, C_ZERO, C_ZERO},
    };

    const matrix4x4 B_NON_NORMALIZED_DAGGER{
        {qfp(0.5, 0.), C_ZERO, C_ZERO, qfp(0.5, 0.)},
        {qfp(0., -0.5), C_ZERO, C_ZERO, qfp(0., 0.5)},
        {C_ZERO, qfp(0., -0.5), qfp(0., -0.5), C_ZERO},
        {C_ZERO, qfp(0.5, 0.), qfp(-0.5, 0.), C_ZERO},
    };
    helpers::print(unitary, "UNITARY in MAGIC BASIS TRANSFORM");
    if (direction == MagicBasisTransform::OutOf) {
      return B_NON_NORMALIZED_DAGGER * unitary * B_NON_NORMALIZED;
    }
    if (direction == MagicBasisTransform::Into) {
      return B_NON_NORMALIZED * unitary * B_NON_NORMALIZED_DAGGER;
    }
    throw std::logic_error{"Unknown MagicBasisTransform direction!"};
  }

  static fp trace_to_fid(const qfp& x) {
    auto x_abs = std::abs(x);
    return (4.0 + x_abs * x_abs) / 20.0;
  }

  static fp closest_partial_swap(fp a, fp b, fp c) {
    auto m = (a + b + c) / 3.;
    auto [am, bm, cm] = std::array{a - m, b - m, c - m};
    auto [ab, bc, ca] = std::array{a - b, b - c, c - a};
    return m + am * bm * cm * (6. + ab * ab + bc * bc + ca * ca) / 18.;
  }

  static matrix2x2 rx_matrix(fp theta) {
    auto half_theta = theta / 2.;
    auto cos = qfp(std::cos(half_theta), 0.);
    auto isin = qfp(0., -std::sin(half_theta));
    return matrix2x2{{cos, isin}, {isin, cos}};
  }

  static matrix2x2 ry_matrix(fp theta) {
    auto half_theta = theta / 2.;
    auto cos = qfp(std::cos(half_theta), 0.);
    auto sin = qfp(std::sin(half_theta), 0.);
    return matrix2x2{{cos, -sin}, {sin, cos}};
  }

  static matrix2x2 rz_matrix(fp theta) {
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

  static matrix4x4 rzzMatrix(const fp theta) {
    const auto cosTheta = std::cos(theta / 2.);
    const auto sinTheta = std::sin(theta / 2.);

    return matrix4x4{{qfp{cosTheta, -sinTheta}, C_ZERO, C_ZERO, C_ZERO},
                     {C_ZERO, {cosTheta, sinTheta}, C_ZERO, C_ZERO},
                     {C_ZERO, C_ZERO, {cosTheta, sinTheta}, C_ZERO},
                     {C_ZERO, C_ZERO, C_ZERO, {cosTheta, -sinTheta}}};
  }

  static std::array<fp, 4> angles_from_unitary(const matrix2x2& matrix,
                                               EulerBasis basis) {
    if (basis == EulerBasis::XYX) {
      return params_xyx_inner(matrix);
    }
    if (basis == EulerBasis::ZYZ) {
      return params_zyz_inner(matrix);
    }
    if (basis == EulerBasis::ZXZ) {
      return params_zxz_inner(matrix);
    }
    throw std::invalid_argument{"Unknown EulerBasis for angles_from_unitary"};
  }

  static std::array<fp, 4> params_zyz_inner(const matrix2x2& matrix) {
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

  static std::array<fp, 4> params_zxz_inner(const matrix2x2& matrix) {
    auto [theta, phi, lam, phase] = params_zyz_inner(matrix);
    return {theta, phi + qc::PI / 2., lam - qc::PI / 2., phase};
  }

  static std::array<fp, 4> params_xyx_inner(const matrix2x2& matrix) {
    auto mat_zyz = matrix2x2{
        {static_cast<fp>(0.5) *
             (matrix(0, 0) + matrix(0, 1) + matrix(1, 0) + matrix(1, 1)),
         static_cast<fp>(0.5) *
             (matrix(0, 0) - matrix(0, 1) + matrix(1, 0) - matrix(1, 1))},
        {static_cast<fp>(0.5) *
             (matrix(0, 0) + matrix(0, 1) - matrix(1, 0) - matrix(1, 1)),
         static_cast<fp>(0.5) *
             (matrix(0, 0) - matrix(0, 1) - matrix(1, 0) + matrix(1, 1))},
    };
    auto [theta, phi, lam, phase] = params_zyz_inner(mat_zyz);
    auto new_phi = mod2pi(phi + qc::PI, 0.);
    auto new_lam = mod2pi(lam + qc::PI, 0.);
    return {
        theta,
        new_phi,
        new_lam,
        phase + (new_phi + new_lam - phi - lam) / 2.,
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
      return rx_matrix(gate.parameter[0]);
    }
    if (gate.type == qc::RY) {
      return ry_matrix(gate.parameter[0]);
    }
    if (gate.type == qc::RZ) {
      return rz_matrix(gate.parameter[0]);
    }
    if (gate.type == qc::X) {
      return matrix2x2{{0, 1}, {1, 0}};
    }
    if (gate.type == qc::I) {
      return identityGate;
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

    if (gate.qubit_id.empty()) {
      return kroneckerProduct(identityGate, identityGate);
    }
    if (gate.qubit_id.size() == 1) {
      if (gate.qubit_id[0] == 0) {
        return kroneckerProduct(identityGate, getSingleQubitMatrix(gate));
      }
      if (gate.qubit_id[0] == 1) {
        return kroneckerProduct(getSingleQubitMatrix(gate), identityGate);
      }
      throw std::logic_error{"Invalid qubit ID in getTwoQubitMatrix"};
    }
    if (gate.qubit_id.size() == 2) {
      if (gate.type == qc::X) {
        // controlled X (CX)
        if (gate.qubit_id == llvm::SmallVector<std::size_t, 2>{1, 0}) {
          return matrix4x4{
              {1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}};
        }
        if (gate.qubit_id == llvm::SmallVector<std::size_t, 2>{0, 1}) {
          return matrix4x4{
              {1, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}, {0, 1, 0, 0}};
        }
      }
      if (gate.type == qc::RZ) {
        throw std::invalid_argument{"RZ for two-qubit gate matrix"};
        // TODO: check qubit order
        return rzzMatrix(gate.parameter[0]);
      }
      if (gate.type == qc::RX) {
        throw std::invalid_argument{"RX for two-qubit gate matrix"};
        // TODO: check qubit order
        return rxxMatrix(gate.parameter[0]);
      }
      if (gate.type == qc::RZZ) {
        // TODO: check qubit order
        return rzzMatrix(gate.parameter[0]);
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
    fp global_phase;
    matrix2x2 K1l;
    matrix2x2 K2l;
    matrix2x2 K1r;
    matrix2x2 K2r;
    Specialization specialization;
    EulerBasis default_euler_basis;
    std::optional<fp> requested_fidelity;
    fp calculated_fidelity;
    matrix4x4 unitary_matrix;

    static TwoQubitWeylDecomposition
    new_inner(matrix4x4 unitary_matrix, std::optional<fp> fidelity,
              std::optional<Specialization> _specialization) {
      auto& u = unitary_matrix;
      auto det_u = u.determinant();
      std::cerr << "DET_U: " << det_u << '\n';
      auto det_pow = std::pow(det_u, static_cast<fp>(-0.25));
      u *= det_pow;
      helpers::print(u, "U", true);
      auto global_phase = std::arg(det_u) / 4.;
      auto u_p = magic_basis_transform(u, MagicBasisTransform::OutOf);
      helpers::print(u_p, "U_P", true);
      matrix4x4 m2 = u_p.transpose() * u_p;
      auto default_euler_basis = EulerBasis::ZYZ;
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
        fp rand_a;
        fp rand_b;
        // For debugging the algorithm use the same RNG values from the
        // previous Python implementation for the first random trial.
        // In most cases this loop only executes a single iteration and
        // using the same rng values rules out possible RNG differences
        // as the root cause of a test failure
        if (i == 0) {
          rand_a = 1.2602066112249388;
          rand_b = 0.22317849046722027;
        } else {
          rand_a = dist(state);
          rand_b = dist(state);
        }
        rmatrix4x4 m2_real = rand_a * m2.real() + rand_b * m2.imag();
        rmatrix4x4 p_inner_real = self_adjoint_eigen_lower(m2_real).first;
        matrix4x4 p_inner = p_inner_real;
        diagonal4x4 d_inner = (p_inner.transpose() * m2 * p_inner).diagonal();

        helpers::print(d_inner, "D_INNER", true);
        helpers::print(p_inner, "P_INNER", true);
        matrix4x4 diag_d = d_inner.asDiagonal();

        matrix4x4 compare = p_inner * diag_d * p_inner.transpose();
        helpers::print(compare, "COMPARE");
        found = (compare - m2).cwiseAbs().cwiseLessOrEqual(1.0e-13).all();
        if (found) {
          p = p_inner;
          d = d_inner;
          break;
        }
      }
      if (!found) {
        throw std::runtime_error{
            "TwoQubitWeylDecomposition: failed to diagonalize M2."};
      }
      rdiagonal4x4 d_real = -1.0 * d.cwiseArg() / 2.0;
      helpers::print(d_real, "D_REAL", true);
      d_real(3) = -d_real(0) - d_real(1) - d_real(2);
      std::array<fp, 3> cs;
      for (std::size_t i = 0; i < cs.size(); ++i) {
        assert(i < d_real.size());
        cs[i] = remEuclid((d_real(i) + d_real(3)) / 2.0, qc::TAU);
      }
      helpers::print(cs, "CS", true);
      decltype(cs) cstemp;
      llvm::transform(cs, cstemp.begin(), [](auto&& x) {
        auto tmp = remEuclid(x, qc::PI_2);
        return std::min(tmp, qc::PI_2 - tmp);
      });
      std::array<std::size_t, cstemp.size()> order{
          2, 1, 0}; // TODO: needs to be adjusted depending on eigenvector
                    // order in eigen decomposition algorithm?
      llvm::stable_sort(order,
                        [&](auto a, auto b) { return cstemp[a] < cstemp[b]; });
      std::tie(order[0], order[1], order[2]) =
          std::tuple{order[1], order[2], order[0]};
      helpers::print(order, "ORDER", true);
      std::tie(cs[0], cs[1], cs[2]) =
          std::tuple{cs[order[0]], cs[order[1]], cs[order[2]]};
      std::tie(d_real(0), d_real(1), d_real(2)) =
          std::tuple{d_real(order[0]), d_real(order[1]), d_real(order[2])};
      helpers::print(d_real, "D_REAL (sorted)", true);

      // swap columns of p according to order
      auto p_orig = p;
      for (std::size_t i = 0; i < order.size(); ++i) {
        p.col(i) = p_orig.col(order[i]);
      }

      if (p.determinant().real() < 0.0) {
        auto lastColumnIndex = p.cols() - 1;
        p.col(lastColumnIndex) = -p.col(lastColumnIndex);
      }

      matrix4x4 temp = d_real.asDiagonal();
      temp *= IM;
      temp = temp.exp();
      helpers::print(temp, "TEMP");
      helpers::print(p, "P", true);
      auto k1 =
          magic_basis_transform(u_p * p * temp, MagicBasisTransform::Into);
      auto k2 = magic_basis_transform(p.transpose(), MagicBasisTransform::Into);

      auto [K1l, K1r, phase_l] = decompose_two_qubit_product_gate(k1);
      auto [K2l, K2r, phase_r] = decompose_two_qubit_product_gate(k2);
      global_phase += phase_l + phase_r;

      // Flip into Weyl chamber
      if (cs[0] > qc::PI_2) {
        cs[0] -= 3.0 * qc::PI_2;
        K1l = K1l * IPY;
        K1r = K1r * IPY;
        global_phase += qc::PI_2;
      }
      if (cs[1] > qc::PI_2) {
        cs[1] -= 3.0 * qc::PI_2;
        K1l = K1l * IPX;
        K1r = K1r * IPX;
        global_phase += qc::PI_2;
      }
      auto conjs = 0;
      if (cs[0] > qc::PI_4) {
        cs[0] = qc::PI_2 - cs[0];
        K1l = K1l * IPY;
        K2r = IPY * K2r;
        conjs += 1;
        global_phase -= qc::PI_2;
      }
      if (cs[1] > qc::PI_4) {
        cs[1] = qc::PI_2 - cs[1];
        K1l = K1l * IPX;
        K2r = IPX * K2r;
        conjs += 1;
        global_phase += qc::PI_2;
        if (conjs == 1) {
          global_phase -= qc::PI;
        }
      }
      if (cs[2] > qc::PI_2) {
        cs[2] -= 3.0 * qc::PI_2;
        K1l = K1l * IPZ;
        K1r = K1r * IPZ;
        global_phase += qc::PI_2;
        if (conjs == 1) {
          global_phase -= qc::PI;
        }
      }
      if (conjs == 1) {
        cs[2] = qc::PI_2 - cs[2];
        K1l = K1l * IPZ;
        K2r = IPZ * K2r;
        global_phase += qc::PI_2;
      }
      if (cs[2] > qc::PI_4) {
        cs[2] -= qc::PI_2;
        K1l = K1l * IPZ;
        K1r = K1r * IPZ;
        global_phase -= qc::PI_2;
      }
      auto [a, b, c] = std::tie(cs[1], cs[0], cs[2]);
      auto is_close = [&](fp ap, fp bp, fp cp) -> bool {
        auto da = a - ap;
        auto db = b - bp;
        auto dc = c - cp;
        auto tr = static_cast<fp>(4.) *
                  qfp(std::cos(da) * std::cos(db) * std::cos(dc),
                      std::sin(da) * std::sin(db) * std::sin(dc));
        if (fidelity) {
          return trace_to_fid(tr) >= *fidelity;
        }
        return false;
      };

      auto closest_abc = closest_partial_swap(a, b, c);
      auto closest_ab_minus_c = closest_partial_swap(a, b, -c);
      auto flipped_from_original = false;

      auto get_default_specialzation = [&]() {
        if (is_close(0., 0., 0.)) {
          return Specialization::IdEquiv;
        } else if (is_close(qc::PI_4, qc::PI_4, qc::PI_4) ||
                   is_close(qc::PI_4, qc::PI_4, -qc::PI_4)) {
          return Specialization::SWAPEquiv;
        } else if (is_close(closest_abc, closest_abc, closest_abc)) {
          return Specialization::PartialSWAPEquiv;
        } else if (is_close(closest_ab_minus_c, closest_ab_minus_c,
                            -closest_ab_minus_c)) {
          return Specialization::PartialSWAPFlipEquiv;
        } else if (is_close(a, 0., 0.)) {
          return Specialization::ControlledEquiv;
        } else if (is_close(qc::PI_4, qc::PI_4, c)) {
          return Specialization::MirrorControlledEquiv;
        } else if (is_close((a + b) / 2., (a + b) / 2., c)) {
          return Specialization::fSimaabEquiv;
        } else if (is_close(a, (b + c) / 2., (b + c) / 2.)) {
          return Specialization::fSimabbEquiv;
        } else if (is_close(a, (b - c) / 2., (c - b) / 2.)) {
          return Specialization::fSimabmbEquiv;
        } else {
          return Specialization::General;
        }
      };
      auto specialization =
          _specialization.value_or(get_default_specialzation());
      TwoQubitWeylDecomposition general{
          .a=a,
          .b=b,
          .c=c,
          .global_phase=global_phase,
          .K1l=K1l,
          .K2l=K2l,
          .K1r=K1r,
          .K2r=K2r,
          .specialization=Specialization::General,
          .default_euler_basis=default_euler_basis,
          .requested_fidelity=fidelity,
          .calculated_fidelity=-1.0,
          .unitary_matrix=unitary_matrix,
      };
      auto get_specialized_decomposition = [&]() {
        // :math:`U \sim U_d(0,0,0) \sim Id`
        //
        // This gate binds 0 parameters, we make it canonical by
        // setting
        // :math:`K2_l = Id` , :math:`K2_r = Id`.
        if (specialization == Specialization::IdEquiv) {
          return TwoQubitWeylDecomposition{
              .a=0.,
              .b=0.,
              .c=0.,
              .global_phase=general.global_phase,
              .K1l=general.K1l * general.K2l,
              .K2l=identityGate,
              .K1r=general.K1r * general.K2r,
              .K2r=identityGate,
              .specialization=specialization,
              .default_euler_basis=general.default_euler_basis,
              .requested_fidelity=general.requested_fidelity,
              .calculated_fidelity=general.calculated_fidelity,
              .unitary_matrix=general.unitary_matrix,
          };
        }
        // :math:`U \sim U_d(\pi/4, \pi/4, \pi/4) \sim U(\pi/4, \pi/4,
        // -\pi/4) \sim \text{SWAP}`
        //
        // This gate binds 0 parameters, we make it canonical by
        // setting
        // :math:`K2_l = Id` , :math:`K2_r = Id`.
        if (specialization == Specialization::SWAPEquiv) {
          if (c > 0.) {
            return TwoQubitWeylDecomposition{
                .a=qc::PI_4,
                .b=qc::PI_4,
                .c=qc::PI_4,
                .global_phase=general.global_phase,
                .K1l=general.K1l * general.K2r,
                .K2l=identityGate,
                .K1r=general.K1r * general.K2l,
                .K2r=identityGate,
                .specialization=specialization,
                .default_euler_basis=general.default_euler_basis,
                .requested_fidelity=general.requested_fidelity,
                .calculated_fidelity=general.calculated_fidelity,
                .unitary_matrix=general.unitary_matrix,
            };
          } else {
            flipped_from_original = true;
            return TwoQubitWeylDecomposition{
                .a=qc::PI_4,
                .b=qc::PI_4,
                .c=qc::PI_4,
                .global_phase=global_phase + qc::PI_2,
                .K1l=general.K1l * IPZ * general.K2r,
                .K2l=identityGate,
                .K1r=general.K1r * IPZ * general.K2l,
                .K2r=identityGate,
                .specialization=specialization,
                .default_euler_basis=general.default_euler_basis,
                .requested_fidelity=general.requested_fidelity,
                .calculated_fidelity=general.calculated_fidelity,
                .unitary_matrix=general.unitary_matrix,
            };
          }
        }
        // :math:`U \sim U_d(\alpha\pi/4, \alpha\pi/4, \alpha\pi/4) \sim
        // \text{SWAP}^\alpha`
        //
        // This gate binds 3 parameters, we make it canonical by setting:
        //
        // :math:`K2_l = Id`.
        if (specialization == Specialization::PartialSWAPEquiv) {
          auto closest = closest_partial_swap(a, b, c);
          auto k2l_dag = general.K2l.transpose().conjugate();

          return TwoQubitWeylDecomposition{
              .a=closest,
              .b=closest,
              .c=closest,
              .global_phase=general.global_phase,
              .K1l=general.K1l * general.K2l,
              .K2l=identityGate,
              .K1r=general.K1r * general.K2l,
              .K2r=k2l_dag * general.K2r,
              .specialization=specialization,
              .default_euler_basis=general.default_euler_basis,
              .requested_fidelity=general.requested_fidelity,
              .calculated_fidelity=general.calculated_fidelity,
              .unitary_matrix=general.unitary_matrix,
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
        if (specialization == Specialization::PartialSWAPFlipEquiv) {
          auto closest = closest_partial_swap(a, b, -c);
          auto k2l_dag = general.K2l.transpose().conjugate();

          return TwoQubitWeylDecomposition{
              .a=closest,
              .b=closest,
              .c=-closest,
              .global_phase=general.global_phase,
              .K1l=general.K1l * general.K2l,
              .K2l=identityGate,
              .K1r=general.K1r * IPZ * general.K2l * IPZ,
              .K2r=IPZ * k2l_dag * IPZ * general.K2r,
              .specialization=specialization,
              .default_euler_basis=general.default_euler_basis,
              .requested_fidelity=general.requested_fidelity,
              .calculated_fidelity=general.calculated_fidelity,
              .unitary_matrix=general.unitary_matrix,
          };
        }
        // :math:`U \sim U_d(\alpha, 0, 0) \sim \text{Ctrl-U}`
        //
        // This gate binds 4 parameters, we make it canonical by setting:
        //
        //      :math:`K2_l = Ry(\theta_l) Rx(\lambda_l)` ,
        //      :math:`K2_r = Ry(\theta_r) Rx(\lambda_r)` .
        if (specialization == Specialization::ControlledEquiv) {
          auto euler_basis = EulerBasis::XYX;
          auto [k2ltheta, k2lphi, k2llambda, k2lphase] =
              angles_from_unitary(general.K2l, euler_basis);
          auto [k2rtheta, k2rphi, k2rlambda, k2rphase] =
              angles_from_unitary(general.K2r, euler_basis);
          return TwoQubitWeylDecomposition{
              .a=a,
              .b=0.,
              .c=0.,
              .global_phase=global_phase + k2lphase + k2rphase,
              .K1l=general.K1l * rx_matrix(k2lphi),
              .K2l=ry_matrix(k2ltheta) * rx_matrix(k2llambda),
              .K1r=general.K1r * rx_matrix(k2rphi),
              .K2r=ry_matrix(k2rtheta) * rx_matrix(k2rlambda),
              .specialization=specialization,
              .default_euler_basis=euler_basis,
              .requested_fidelity=general.requested_fidelity,
              .calculated_fidelity=general.calculated_fidelity,
              .unitary_matrix=general.unitary_matrix,
          };
        }
        // :math:`U \sim U_d(\pi/4, \pi/4, \alpha) \sim \text{SWAP} \cdot
        // \text{Ctrl-U}`
        //
        // This gate binds 4 parameters, we make it canonical by setting:
        //
        // :math:`K2_l = Ry(\theta_l)\cdot Rz(\lambda_l)` , :math:`K2_r =
        // Ry(\theta_r)\cdot Rz(\lambda_r)`
        if (specialization == Specialization::MirrorControlledEquiv) {
          auto [k2ltheta, k2lphi, k2llambda, k2lphase] =
              angles_from_unitary(general.K2l, EulerBasis::ZYZ);
          auto [k2rtheta, k2rphi, k2rlambda, k2rphase] =
              angles_from_unitary(general.K2r, EulerBasis::ZYZ);
          return TwoQubitWeylDecomposition{
              .a=qc::PI_4,
              .b=qc::PI_4,
              .c=c,
              .global_phase=global_phase + k2lphase + k2rphase,
              .K1l=general.K1l * rz_matrix(k2rphi),
              .K2l=ry_matrix(k2ltheta) * rz_matrix(k2llambda),
              .K1r=general.K1r * rz_matrix(k2lphi),
              .K2r=ry_matrix(k2rtheta) * rz_matrix(k2rlambda),
              .specialization=specialization,
              .default_euler_basis=general.default_euler_basis,
              .requested_fidelity=general.requested_fidelity,
              .calculated_fidelity=general.calculated_fidelity,
              .unitary_matrix=general.unitary_matrix,
          };
        }
        // :math:`U \sim U_d(\alpha, \alpha, \beta), \alpha \geq |\beta|`
        //
        // This gate binds 5 parameters, we make it canonical by setting:
        //
        // :math:`K2_l = Ry(\theta_l)\cdot Rz(\lambda_l)`.
        if (specialization == Specialization::fSimaabEquiv) {
          auto [k2ltheta, k2lphi, k2llambda, k2lphase] =
              angles_from_unitary(general.K2l, EulerBasis::ZYZ);
          return TwoQubitWeylDecomposition{
              .a=(a + b) / 2.,
              .b=(a + b) / 2.,
              .c=c,
              .global_phase=global_phase + k2lphase,
              .K1l=general.K1l * rz_matrix(k2lphi),
              .K2l=ry_matrix(k2ltheta) * rz_matrix(k2llambda),
              .K1r=general.K1r * rz_matrix(k2lphi),
              .K2r=rz_matrix(-k2lphi) * general.K2r,
              .specialization=specialization,
              .default_euler_basis=general.default_euler_basis,
              .requested_fidelity=general.requested_fidelity,
              .calculated_fidelity=general.calculated_fidelity,
              .unitary_matrix=general.unitary_matrix,
          };
        }
        // :math:`U \sim U_d(\alpha, \beta, -\beta), \alpha \geq \beta \geq 0`
        //
        // This gate binds 5 parameters, we make it canonical by setting:
        //
        // :math:`K2_l = Ry(\theta_l)Rx(\lambda_l)`
        if (specialization == Specialization::fSimabbEquiv) {
          auto euler_basis = EulerBasis::XYX;
          auto [k2ltheta, k2lphi, k2llambda, k2lphase] =
              angles_from_unitary(general.K2l, euler_basis);
          return TwoQubitWeylDecomposition{
              .a=a,
              .b=(b + c) / 2.,
              .c=(b + c) / 2.,
              .global_phase=global_phase + k2lphase,
              .K1l=general.K1l * rx_matrix(k2lphi),
              .K2l=ry_matrix(k2ltheta) * rx_matrix(k2llambda),
              .K1r=general.K1r * rx_matrix(k2lphi),
              .K2r=rx_matrix(-k2lphi) * general.K2r,
              .specialization=specialization,
              .default_euler_basis=euler_basis,
              .requested_fidelity=general.requested_fidelity,
              .calculated_fidelity=general.calculated_fidelity,
              .unitary_matrix=general.unitary_matrix,
          };
        }
        // :math:`U \sim U_d(\alpha, \beta, -\beta), \alpha \geq \beta \geq 0`
        //
        // This gate binds 5 parameters, we make it canonical by setting:
        //
        // :math:`K2_l = Ry(\theta_l)Rx(\lambda_l)`
        if (specialization == Specialization::fSimabmbEquiv) {
          auto euler_basis = EulerBasis::XYX;
          auto [k2ltheta, k2lphi, k2llambda, k2lphase] =
              angles_from_unitary(general.K2l, euler_basis);
          return TwoQubitWeylDecomposition{
              .a=a,
              .b=(b - c) / 2.,
              .c=-((b - c) / 2.),
              .global_phase=global_phase + k2lphase,
              .K1l=general.K1l * rx_matrix(k2lphi),
              .K2l=ry_matrix(k2ltheta) * rx_matrix(k2llambda),
              .K1r=general.K1r * IPZ * rx_matrix(k2lphi) * IPZ,
              .K2r=IPZ * rx_matrix(-k2lphi) * IPZ * general.K2r,
              .specialization=specialization,
              .default_euler_basis=euler_basis,
              .requested_fidelity=general.requested_fidelity,
              .calculated_fidelity=general.calculated_fidelity,
              .unitary_matrix=general.unitary_matrix,
          };
        }
        // U has no special symmetry.
        //
        // This gate binds all 6 possible parameters, so there is no need to
        // make the single-qubit pre-/post-gates canonical.
        if (specialization == Specialization::General) {
          return general;
        }
        throw std::logic_error{"Unknown specialization"};
      };

      TwoQubitWeylDecomposition specialized = get_specialized_decomposition();

      auto get_tr = [&]() {
        if (flipped_from_original) {
          auto [da, db, dc] = std::array{
              qc::PI_2 - a - specialized.a,
              b - specialized.b,
              -c - specialized.c,
          };
          return static_cast<fp>(4.) *
                 qfp(std::cos(da) * std::cos(db) * std::cos(dc),
                     std::sin(da) * std::sin(db) * std::sin(dc));
        } else {
          auto [da, db, dc] = std::array{a - specialized.a, b - specialized.b,
                                         c - specialized.c};
          return static_cast<fp>(4.) *
                 qfp(std::cos(da) * std::cos(db) * std::cos(dc),
                     std::sin(da) * std::sin(db) * std::sin(dc));
        }
      };
      auto tr = get_tr();
      specialized.calculated_fidelity = trace_to_fid(tr);
      if (specialized.requested_fidelity) {
        if (specialized.calculated_fidelity + 1.0e-13 <
            *specialized.requested_fidelity) {
          throw std::runtime_error{
              "Specialization: {:?} calculated fidelity: {} is worse than "
              "requested fidelity: {}",
          };
        }
      }
      specialized.global_phase += std::arg(tr);
      return specialized;
    }
  };

  static constexpr auto DEFAULT_FIDELITY = 1.0 - 1.0e-9;

  struct TwoQubitBasisDecomposer {
    QubitGateSequence::Gate basis_gate;
    fp basis_fidelity;
    EulerBasis euler_basis;
    TwoQubitWeylDecomposition basis_decomposer;
    bool super_controlled;
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
    new_inner(OneQubitGateSequence::Gate basis_gate = {.type = qc::X,
                                                       .parameter = {},
                                                       .qubit_id = {0, 1}},
              fp basis_fidelity = 1.0, EulerBasis euler_basis = EulerBasis::ZYZ) {
      auto relative_eq = [](auto&& lhs, auto&& rhs, auto&& epsilon,
                            auto&& max_relative) {
        // Handle same infinities
        if (lhs == rhs) {
          return true;
        }

        // Handle remaining infinities
        if (std::isinf(lhs) || std::isinf(rhs)) {
          return false;
        }

        auto abs_diff = std::abs(lhs - rhs);

        // For when the numbers are really close together
        if (abs_diff <= epsilon) {
          return true;
        }

        auto abs_lhs = std::abs(lhs);
        auto abs_rhs = std::abs(rhs);
        if (abs_rhs > abs_lhs) {
          return abs_diff <= abs_rhs * max_relative;
        }
        return abs_diff <= abs_lhs * max_relative;
      };
      constexpr auto FRAC_1_SQRT_2 =
          static_cast<fp>(0.707106781186547524400844362104849039);
      const auto K12R_ARR = matrix2x2{
          {qfp(0., FRAC_1_SQRT_2), qfp(FRAC_1_SQRT_2, 0.)},
          {qfp(-FRAC_1_SQRT_2, 0.), qfp(0., -FRAC_1_SQRT_2)},
      };
      const auto K12L_ARR = matrix2x2{
          {qfp(0.5, 0.5), qfp(0.5, 0.5)},
          {qfp(-0.5, 0.5), qfp(0.5, -0.5)},
      };

      auto basis_decomposer = TwoQubitWeylDecomposition::new_inner(
          getTwoQubitMatrix(basis_gate), DEFAULT_FIDELITY, std::nullopt);
      auto super_controlled =
          relative_eq(basis_decomposer.a, qc::PI_4,
                      std::numeric_limits<fp>::epsilon(), 1e-09) &&
          relative_eq(basis_decomposer.c, 0.0,
                      std::numeric_limits<fp>::epsilon(), 1e-09);

      // Create some useful matrices U1, U2, U3 are equivalent to the basis,
      // expand as Ui = Ki1.Ubasis.Ki2
      auto b = basis_decomposer.b;
      auto temp = qfp(0.5, -0.5);
      auto k11l = matrix2x2{
          {temp * (M_IM * std::exp(qfp(0., -b))), temp * std::exp(qfp(0., -b))},
          {temp * (M_IM * std::exp(qfp(0., b))),
           temp * -(std::exp(qfp(0., b)))}};
      auto k11r = matrix2x2{{FRAC_1_SQRT_2 * std::exp((IM * qfp(0., -b))),
                             FRAC_1_SQRT_2 * -std::exp(qfp(0., -b))},
                            {FRAC_1_SQRT_2 * std::exp(qfp(0., b)),
                             FRAC_1_SQRT_2 * (M_IM * std::exp(qfp(0., b)))}};
      auto k32l_k21l = matrix2x2{{FRAC_1_SQRT_2 * std::cos(qfp(1., (2. * b))),
                                  FRAC_1_SQRT_2 * (IM * std::sin((2. * b)))},
                                 {FRAC_1_SQRT_2 * (IM * std::sin(2. * b)),
                                  FRAC_1_SQRT_2 * qfp(1., -std::cos(2. * b))}};
      temp = qfp(0.5, 0.5);
      auto k21r = matrix2x2{
          {temp * (M_IM * std::exp(qfp(0., -2. * b))),
           temp * std::exp(qfp(0., -2. * b))},
          {temp * (IM * std::exp(qfp(0., 2. * b))),
           temp * std::exp(qfp(0., 2. * b))},
      };
      const auto K22L_ARR = matrix2x2{
          {qfp(FRAC_1_SQRT_2, 0.), qfp(-FRAC_1_SQRT_2, 0.)},
          {qfp(FRAC_1_SQRT_2, 0.), qfp(FRAC_1_SQRT_2, 0.)},
      };
      const auto K22R_ARR = matrix2x2{{C_ZERO, C_ONE}, {C_M_ONE, C_ZERO}};
      auto k31l = matrix2x2{
          {FRAC_1_SQRT_2 * std::exp(qfp(0., -b)),
           FRAC_1_SQRT_2 * std::exp(qfp(0., -b))},
          {FRAC_1_SQRT_2 * -std::exp(qfp(0., b)),
           FRAC_1_SQRT_2 * std::exp(qfp(0., b))},
      };
      auto k31r = matrix2x2{
          {IM * std::exp(qfp(0., b)), C_ZERO},
          {C_ZERO, M_IM * std::exp(qfp(0., -b))},
      };
      temp = qfp(0.5, 0.5);
      auto k32r = matrix2x2{
          {temp * std::exp(qfp(0., b)), temp * -std::exp(qfp(0., -b))},
          {temp * (M_IM * std::exp(qfp(0., b))),
           temp * (M_IM * std::exp(qfp(0., -b)))},
      };
      auto k1ld = basis_decomposer.K1l.transpose().conjugate();
      auto k1rd = basis_decomposer.K1r.transpose().conjugate();
      auto k2ld = basis_decomposer.K2l.transpose().conjugate();
      auto k2rd = basis_decomposer.K2r.transpose().conjugate();
      // Pre-build the fixed parts of the matrices used in 3-part
      // decomposition
      auto u0l = k31l * k1ld;
      auto u0r = k31r * k1rd;
      auto u1l = k2ld * k32l_k21l * k1ld;
      auto u1ra = k2rd * k32r;
      auto u1rb = k21r * k1rd;
      auto u2la = k2ld * K22L_ARR;
      auto u2lb = k11l * k1ld;
      auto u2ra = k2rd * K22R_ARR;
      auto u2rb = k11r * k1rd;
      auto u3l = k2ld * K12L_ARR;
      auto u3r = k2rd * K12R_ARR;
      // Pre-build the fixed parts of the matrices used in the 2-part
      // decomposition
      auto q0l = K12L_ARR.transpose().conjugate() * k1ld;
      auto q0r = K12R_ARR.transpose().conjugate() * IPZ * k1rd;
      auto q1la = k2ld * k11l.transpose().conjugate();
      auto q1lb = k11l * k1ld;
      auto q1ra = k2rd * IPZ * k11r.transpose().conjugate();
      auto q1rb = k11r * k1rd;
      auto q2l = k2ld * K12L_ARR;
      auto q2r = k2rd * K12R_ARR;

      return TwoQubitBasisDecomposer{
          .basis_gate=basis_gate,
          .basis_fidelity=basis_fidelity,
          .euler_basis=euler_basis,
          .basis_decomposer=basis_decomposer,
          .super_controlled=super_controlled,
          .u0l=u0l,
          .u0r=u0r,
          .u1l=u1l,
          .u1ra=u1ra,
          .u1rb=u1rb,
          .u2la=u2la,
          .u2lb=u2lb,
          .u2ra=u2ra,
          .u2rb=u2rb,
          .u3l=u3l,
          .u3r=u3r,
          .q0l=q0l,
          .q0r=q0r,
          .q1la=q1la,
          .q1lb=q1lb,
          .q1ra=q1ra,
          .q1rb=q1rb,
          .q2l=q2l,
          .q2r=q2r,
      };
    }

    std::optional<TwoQubitGateSequence>
    twoQubitDecompose(const matrix4x4& unitaryMatrix,
                      std::optional<fp> _basis_fidelity, bool approximate,
                      std::optional<std::uint8_t> _num_basis_uses) {
      auto get_basis_fidelity = [&]() {
        if (approximate) {
          return _basis_fidelity.value_or(this->basis_fidelity);
        }
        return static_cast<fp>(1.0);
      };
      fp basis_fidelity = get_basis_fidelity();
      auto target_decomposed = TwoQubitWeylDecomposition::new_inner(
          unitaryMatrix, DEFAULT_FIDELITY, std::nullopt);
      auto traces = this->traces(target_decomposed);
      auto get_default_nbasis = [&]() {
        auto minValue = std::numeric_limits<fp>::min();
        auto minIndex = -1;
        for (std::size_t i = 0; i < traces.size(); ++i) {
          auto value = trace_to_fid(traces[i]) * std::pow(basis_fidelity, i);
          if (value > minValue) {
            minIndex = i;
            minValue = value;
          }
        }
        return minIndex;
      };
      auto best_nbasis = _num_basis_uses.value_or(get_default_nbasis());
      auto choose_decomposition = [&]() {
        if (best_nbasis == 0) {
          return decomp0_inner(target_decomposed);
        }
        if (best_nbasis == 1) {
          return decomp1_inner(target_decomposed);
        }
        if (best_nbasis == 2) {
          return decomp2_supercontrolled_inner(target_decomposed);
        }
        if (best_nbasis == 3) {
          return decomp3_supercontrolled_inner(target_decomposed);
        }
        throw std::logic_error{"Invalid basis to use"};
      };
      auto decomposition = choose_decomposition();
      std::cerr << "NBasis: " << (int)best_nbasis
                << "; basis_fid: " << basis_fidelity
                << "; Traces: " << traces[0] << ", " << traces[1] << ", "
                << traces[2] << ", " << traces[3];
      std::cerr << "\nDecomposition:\n";
      for (auto x : decomposition) {
        helpers::print(x, "", true);
      }
      std::vector<EulerBasis>
          target_1q_basis_list; // TODO: simplify because list only has one
                                // element?
      target_1q_basis_list.push_back(euler_basis);
      llvm::SmallVector<std::optional<TwoQubitGateSequence>, 8>
          euler_decompositions;
      for (auto&& decomp : decomposition) {
        auto euler_decomp = unitary_to_gate_sequence_inner(
            decomp, target_1q_basis_list, 0, {}, true, std::nullopt);
        euler_decompositions.push_back(euler_decomp);
      }
      TwoQubitGateSequence gates{.globalPhase = target_decomposed.global_phase};
      // Worst case length is 5x 1q gates for each 1q decomposition + 1x 2q
      // gate We might overallocate a bit if the euler basis is different but
      // the worst case is just 16 extra elements with just a String and 2
      // smallvecs each. This is only transient though as the circuit
      // sequences aren't long lived and are just used to create a
      // QuantumCircuit or DAGCircuit when we return to Python space.
      constexpr auto TWO_QUBIT_SEQUENCE_DEFAULT_CAPACITY = 21;
      gates.gates.reserve(TWO_QUBIT_SEQUENCE_DEFAULT_CAPACITY);
      gates.globalPhase -= best_nbasis * basis_decomposer.global_phase;
      if (best_nbasis == 2) {
        gates.globalPhase += qc::PI;
      }

      auto add_euler_decomposition = [&](std::size_t index,
                                         std::size_t qubit_id) {
        if (auto&& euler_decomp = euler_decompositions[index]) {
          for (auto&& gate : euler_decomp->gates) {
            gates.gates.push_back({.type = gate.type,
                                   .parameter = gate.parameter,
                                   .qubit_id = {qubit_id}});
            gates.globalPhase += euler_decomp->globalPhase;
          }
        }
      };

      for (std::size_t i = 0; i < best_nbasis; ++i) {
        add_euler_decomposition(2 * i, 0);
        add_euler_decomposition(2 * i + 1, 1);

        gates.gates.push_back(basis_gate);
      }

      add_euler_decomposition(2 * best_nbasis, 0);
      add_euler_decomposition(2 * best_nbasis + 1, 1);

      return gates;
    }

  private:
    [[nodiscard]] std::vector<matrix2x2>
    decomp0_inner(const TwoQubitWeylDecomposition& target) const {
      return {
          target.K1r * target.K2r,
          target.K1l * target.K2l,
      };
    }

    [[nodiscard]] std::vector<matrix2x2>
    decomp1_inner(const TwoQubitWeylDecomposition& target) const {
      // FIXME: fix for z!=0 and c!=0 using closest reflection (not always in
      // the Weyl chamber)
      return {
          basis_decomposer.K2r.transpose().conjugate() * target.K2r,
          basis_decomposer.K2l.transpose().conjugate() * target.K2l,
          target.K1r * basis_decomposer.K1r.transpose().conjugate(),
          target.K1l * basis_decomposer.K1l.transpose().conjugate(),
      };
    }

    [[nodiscard]] std::vector<matrix2x2> decomp2_supercontrolled_inner(
        const TwoQubitWeylDecomposition& target) const {
      return {
          q2r * target.K2r,
          q2l * target.K2l,
          q1ra * rz_matrix(2. * target.b) * q1rb,
          q1la * rz_matrix(-2. * target.a) * q1lb,
          target.K1r * q0r,
          target.K1l * q0l,
      };
    }

    [[nodiscard]] std::vector<matrix2x2> decomp3_supercontrolled_inner(
        const TwoQubitWeylDecomposition& target) const {
      return {
          u3r * target.K2r,
          u3l * target.K2l,
          u2ra * rz_matrix(2. * target.b) * u2rb,
          u2la * rz_matrix(-2. * target.a) * u2lb,
          u1ra * rz_matrix(-2. * target.c) * u1rb,
          u1l,
          target.K1r * u0r,
          target.K1l * u0l,
      };
    }

    matrix4x4 compute_unitary(const TwoQubitGateSequence& sequence,
                              fp global_phase) {
      auto phase = std::exp(std::complex<fp>{0, global_phase});
      matrix4x4 matrix{};
      matrix.diagonal().setConstant(phase);

      for (auto&& gate : sequence.gates) {
        matrix4x4 gate_matrix = getTwoQubitMatrix(gate);

        matrix = gate_matrix * matrix;
      }
      return matrix;
    }

    [[nodiscard]] std::array<qfp, 4>
    traces(TwoQubitWeylDecomposition target) const {
      return {
          static_cast<fp>(4.) *
              qfp(std::cos(target.a) * std::cos(target.b) * std::cos(target.c),
                  std::sin(target.a) * std::sin(target.b) * std::sin(target.c)),
          static_cast<fp>(4.) *
              qfp(std::cos(qc::PI_4 - target.a) *
                      std::cos(basis_decomposer.b - target.b) *
                      std::cos(target.c),
                  std::sin(qc::PI_4 - target.a) *
                      std::sin(basis_decomposer.b - target.b) *
                      std::sin(target.c)),
          qfp(4. * std::cos(target.c), 0.),
          qfp(4., 0.),
      };
    }

    OneQubitGateSequence generate_circuit(EulerBasis target_basis,
                                          const matrix2x2& unitaryMatrix,
                                          bool simplify,
                                          std::optional<fp> atol) {
      auto [theta, phi, lambda, phase] =
          angles_from_unitary(unitaryMatrix, target_basis);

      switch (target_basis) {
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

    OneQubitGateSequence unitary_to_gate_sequence_inner(
        matrix2x2 unitary_mat, const std::vector<EulerBasis>& target_basis_list,
        std::size_t qubit,
        const std::vector<std::unordered_map<std::string, fp>>&
            error_map, // TODO: remove error_map+qubit for platform
                       // independence
        bool simplify, std::optional<fp> atol) {
      auto calculateError = [](const OneQubitGateSequence& sequence) {
        return sequence.gates.size();
      };

      auto minError = std::numeric_limits<fp>::max();
      OneQubitGateSequence bestCircuit;
      for (std::size_t i = 0; i < target_basis_list.size(); ++i) {
        auto& target_basis = target_basis_list[i];
        auto circuit =
            generate_circuit(target_basis, unitary_mat, simplify, atol);
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
      if (std::abs(theta) < angleZeroEpsilon) {
        lambda += phi;
        lambda = mod2pi(lambda);
        if (std::abs(lambda) > angleZeroEpsilon) {
          gates.push_back({kGate, {lambda}});
          globalPhase += lambda / 2.0;
        }
        return {gates, globalPhase};
      }

      if (std::abs(theta - qc::PI) < angleZeroEpsilon) {
        globalPhase += phi;
        lambda -= phi;
        phi = 0.0;
      }
      if (std::abs(mod2pi(lambda + qc::PI)) < angleZeroEpsilon ||
          std::abs(mod2pi(phi + qc::PI)) < angleZeroEpsilon) {
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
  }; // namespace mqt::ir::opt
};

const matrix2x2 GateDecompositionPattern::identityGate = matrix2x2::Identity();
const matrix2x2 GateDecompositionPattern::hGate{{1.0 / sqrt2, 1.0 / sqrt2},
                                                {1.0 / sqrt2, -1.0 / sqrt2}};
const matrix2x2 GateDecompositionPattern::IPZ{{IM, C_ZERO}, {C_ZERO, M_IM}};
const matrix2x2 GateDecompositionPattern::IPY{{C_ZERO, C_ONE},
                                              {C_M_ONE, C_ZERO}};
const matrix2x2 GateDecompositionPattern::IPX{{C_ZERO, IM}, {IM, C_ZERO}};

/**
 * @brief Populates the given pattern set with patterns for gate
 * decomposition.
 */
void populateGateDecompositionPatterns(mlir::RewritePatternSet& patterns) {
  patterns.add<GateDecompositionPattern>(patterns.getContext());
}

} // namespace mqt::ir::opt

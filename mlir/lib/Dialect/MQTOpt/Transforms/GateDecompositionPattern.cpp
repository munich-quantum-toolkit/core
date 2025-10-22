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
#include "d.h"
#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h"

#include <array>
#include <cstddef>
#include <functional>
#include <llvm/ADT/STLExtras.h>
#include <map>
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
    auto series = getTwoQubitSeries(op);
    llvm::errs() << "SERIES SIZE: " << series.size() << '\n';

    if (series.size() <= 3) {
      return mlir::failure();
    }

    matrix4x4 unitaryMatrix =
        helpers::kroneckerProduct(identityGate, identityGate);
    for (auto&& gate : series) {
      auto gateMatrix = getTwoQubitMatrix({.type = helpers::getQcType(gate),
                                           .parameter = {/*TODO*/},
                                           .qubit_id = {0, 1}});
      unitaryMatrix = helpers::multiply(unitaryMatrix, gateMatrix);
    }

    auto decomposer = TwoQubitBasisDecomposer::new_inner();
    auto sequence = decomposer.twoQubitDecompose(
        unitaryMatrix, DEFAULT_FIDELITY, true, std::nullopt);
    if (!sequence) {
      return mlir::failure();
    }

    applySeries(rewriter, series, *sequence);

    return mlir::success();
  }

  [[nodiscard]] static llvm::SmallVector<UnitaryInterface>
  getTwoQubitSeries(UnitaryInterface op) {
    llvm::SmallVector<mlir::Value, 2> qubits(2);
    llvm::SmallVector<UnitaryInterface> result;

    if (helpers::isSingleQubitOperation(op)) {
      qubits = {op->getResult(0), mlir::Value{}};
    } else if (helpers::isTwoQubitOperation(op)) {
      qubits = op->getResults();
    } else {
      return result;
    }
    result.push_back(op);

    auto findNextInSeries = [&](mlir::Operation* user) {
      auto userUnitary = mlir::dyn_cast<UnitaryInterface>(user);
      if (!userUnitary) {
        return false;
      }

      if (helpers::isSingleQubitOperation(userUnitary)) {
        auto&& operand = userUnitary->getOperand(0);
        auto* it = llvm::find(qubits, operand);
        if (it == qubits.end()) {
          throw std::logic_error{"Operand of single-qubit op and user of "
                                 "qubit is not in qubits"};
        }
        *it = userUnitary->getResult(0);

        result.push_back(userUnitary);
        return true;
      }
      if (helpers::isTwoQubitOperation(userUnitary)) {
        auto&& firstOperand = userUnitary->getOperand(0);
        auto&& secondOperand = userUnitary->getOperand(1);
        auto* firstQubitIt = llvm::find(qubits, firstOperand);
        auto* secondQubitIt = llvm::find(qubits, secondOperand);
        if (firstQubitIt == qubits.end() || secondQubitIt == qubits.end()) {
          return false;
        }
        *firstQubitIt = userUnitary->getResult(0);
        *secondQubitIt = userUnitary->getResult(1);

        result.push_back(userUnitary);
        return true;
      }
      return false;
    };

    while (true) {
      assert(qubits[0].hasOneUse());
      assert(qubits[1].hasOneUse());
      bool isSeriesOngoing = findNextInSeries(*qubits[0].getUsers().begin()) ||
                             findNextInSeries(*qubits[1].getUsers().begin());
      if (!isSeriesOngoing) {
        return result;
      }
    }
  }

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

  struct QubitGateSequence {
    struct Gate {
      qc::OpType type{qc::I};
      std::vector<fp> parameter;
      std::vector<std::size_t> qubit_id = {0};
    };
    std::vector<Gate> gates;
    fp globalPhase;
  };
  using OneQubitGateSequence = QubitGateSequence;
  using TwoQubitGateSequence = QubitGateSequence;

  static void applySeries(mlir::PatternRewriter& rewriter,
                          const llvm::SmallVector<UnitaryInterface>& series,
                          const TwoQubitGateSequence& sequence) {
    if (sequence.globalPhase != 0.0) {
      createOneParameterGate<GPhaseOp>(rewriter, series[0]->getLoc(),
                                       sequence.globalPhase, {});
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
  static constexpr matrix2x2 identityGate = {1, 0, 0, 1};
  static constexpr matrix2x2 hGate = {1.0 / sqrt2, 1.0 / sqrt2, 1.0 / sqrt2,
                                      -1.0 / sqrt2};

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

  // GPT generated
  void tridiagonalize_inplace(rmatrix4x4& A) {
    auto dot3 = [](auto&& a, auto&& b) {
      return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    };
    // Householder for column 0
    {
      auto x = std::array{A[1 * 4 + 0], A[8 + 0],
                          A[12 + 0]}; // Elements [1,0], [2,0], [3,0]
      auto norm_x = std::sqrt(dot3(x, x));
      if (norm_x > 1e-12) {
        auto sign = (x[0] >= 0) ? 1.0 : -1.0;
        auto u0 = x[0] + sign * norm_x;
        auto denom = std::sqrt(u0 * u0 + x[1] * x[1] + x[2] * x[2]);
        auto v = std::array{u0 / denom, x[1] / denom, x[2] / denom};

        // Apply H = I - 2vvᵀ to A from both sides: A := H A H
        // Only affect rows and columns 1-3

        // temp = A * v
        std::array<fp, 3> temp = {0, 0, 0};
        for (int i = 1; i < 4; ++i) {
          for (int j = 1; j < 4; ++j) {
            temp[i - 1] += A[i * 4 + j] * v[j - 1];
          }
        }

        // w = 2 * (A*v - (vᵀ*A*v)*v)
        auto alpha = dot3(v, temp);
        std::array<fp, 3> w;
        for (int i = 0; i < 3; ++i)
          w[i] = 2.0 * (temp[i] - alpha * v[i]);

        // Update A = A - v wᵀ - w vᵀ
        for (int i = 1; i < 4; ++i) {
          for (int j = i; j < 4; ++j) {
            auto delta = v[i - 1] * w[j - 1] + w[i - 1] * v[j - 1];
            A[i * 4 + j] -= delta;
            if (i != j)
              A[j * 4 + i] -= delta; // symmetry
          }
        }
      }
    }

    // Householder for column 1 (submatrix 2x2 at bottom right)
    {
      auto x = std::array{A[10], A[15]}; // Elements [2,1], [3,1]
      auto norm_x = std::sqrt(x[0] * x[0] + x[1] * x[1]);
      if (norm_x > 1e-12) {
        auto sign = (x[0] >= 0) ? 1.0 : -1.0;
        auto u0 = x[0] + sign * norm_x;
        auto denom = std::sqrt(u0 * u0 + x[1] * x[1]);
        auto v = std::array{u0 / denom, x[1] / denom};

        // temp = A * v
        std::array<fp, 2> temp = {0, 0};
        for (int i = 2; i < 4; ++i) {
          for (int j = 2; j < 4; ++j) {
            temp[i - 2] += A[i * 4 + j] * v[j - 2];
          }
        }

        // w = 2 * (A*v - (vᵀ*A*v)*v)
        auto alpha = v[0] * temp[0] + v[1] * temp[1];
        std::array<fp, 2> w;
        for (int i = 0; i < 2; ++i)
          w[i] = 2.0 * (temp[i] - alpha * v[i]);

        // Update A = A - v wᵀ - w vᵀ
        for (int i = 2; i < 4; ++i) {
          for (int j = i; j < 4; ++j) {
            auto delta = v[i - 2] * w[j - 2] + w[i - 2] * v[j - 2];
            A[i * 4 + j] -= delta;
            if (i != j)
              A[j * 4 + i] -= delta;
          }
        }
      }
    }

    // Now A is tridiagonal
  }
  // https://docs.rs/faer/latest/faer/mat/generic/struct.Mat.html#method.self_adjoint_eigen
  static std::pair<rmatrix4x4, rdiagonal4x4>
  self_adjoint_eigen_lower(rmatrix4x4 A) {
    // rdiagonal4x4 S;
    // auto U = self_adjoint_evd(A, S);

    // rmatrix4x4 U;
    // jacobi_eigen_decomposition(A, U, S);

    auto [U, S] = self_adjoint_evd(A);

    return std::make_pair(U, S);
  }

  static std::tuple<matrix2x2, matrix2x2, fp>
  decompose_two_qubit_product_gate(matrix4x4 special_unitary) {
    using helpers::determinant;
    using helpers::dot;
    using helpers::kroneckerProduct;
    using helpers::transpose_conjugate;
    // first quadrant
    matrix2x2 r = {special_unitary[0 * 4 + 0], special_unitary[0 * 4 + 1],
                   special_unitary[1 * 4 + 0], special_unitary[1 * 4 + 1]};
    auto det_r = determinant(r);
    if (std::abs(det_r) < 0.1) {
      // third quadrant
      r = {special_unitary[2 * 4 + 0], special_unitary[2 * 4 + 1],
           special_unitary[3 * 4 + 0], special_unitary[3 * 4 + 1]};
      det_r = determinant(r);
    }
    if (std::abs(det_r) < 0.1) {
      throw std::runtime_error{
          "decompose_two_qubit_product_gate: unable to decompose: det_r < 0.1"};
    }
    llvm::transform(r, r.begin(),
                    [&](auto&& x) { return x / std::sqrt(det_r); });
    // transpose with complex conjugate of each element
    matrix2x2 r_t_conj = transpose_conjugate(r);

    auto temp = kroneckerProduct(identityGate, r_t_conj);
    temp = dot(special_unitary, temp);

    // [[a, b, c, d],
    //  [e, f, g, h], => [[a, c],
    //  [i, j, k, l],     [i, k]]
    //  [m, n, o, p]]
    matrix2x2 l = {temp[0 * 4 + 0], temp[0 * 4 + 2], temp[2 * 4 + 0],
                   temp[2 * 4 + 2]};
    auto det_l = determinant(l);
    if (std::abs(det_l) < 0.9) {
      throw std::runtime_error{
          "decompose_two_qubit_product_gate: unable to decompose: detL < 0.9"};
    }
    llvm::transform(l, l.begin(),
                    [&](auto&& x) { return x / std::sqrt(det_l); });
    auto phase = std::arg(det_l) / 2.;

    return {l, r, phase};
  }

  static matrix4x4 magic_basis_transform(const matrix4x4& unitary,
                                         MagicBasisTransform direction) {
    using helpers::dot;
    constexpr matrix4x4 B_NON_NORMALIZED = {
        C_ONE,  IM,     C_ZERO, C_ZERO,  C_ZERO, C_ZERO, IM,     C_ONE,
        C_ZERO, C_ZERO, IM,     C_M_ONE, C_ONE,  M_IM,   C_ZERO, C_ZERO,
    };

    constexpr matrix4x4 B_NON_NORMALIZED_DAGGER = {
        qfp(0.5, 0.),  C_ZERO,        C_ZERO,        qfp(0.5, 0.),
        qfp(0., -0.5), C_ZERO,        C_ZERO,        qfp(0., 0.5),
        C_ZERO,        qfp(0., -0.5), qfp(0., -0.5), C_ZERO,
        C_ZERO,        qfp(0.5, 0.),  qfp(-0.5, 0.), C_ZERO,
    };
    if (direction == MagicBasisTransform::OutOf) {
      return dot(dot(B_NON_NORMALIZED_DAGGER, unitary), B_NON_NORMALIZED);
    }
    if (direction == MagicBasisTransform::Into) {
      return dot(dot(B_NON_NORMALIZED, unitary), B_NON_NORMALIZED_DAGGER);
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
    return {cos, isin, isin, cos};
  }

  static matrix2x2 ry_matrix(fp theta) {
    auto half_theta = theta / 2.;
    auto cos = qfp(std::cos(half_theta), 0.);
    auto sin = qfp(std::sin(half_theta), 0.);
    return {cos, -sin, sin, cos};
  }

  static matrix2x2 rz_matrix(fp theta) {
    auto ilam2 = qfp(0., 0.5 * theta);
    return {std::exp(-ilam2), C_ZERO, C_ZERO, std::exp(ilam2)};
  }

  static std::array<fp, 4> angles_from_unitary(const matrix2x2& matrix,
                                               EulerBasis basis) {
    if (basis == EulerBasis::XYX) {
      return params_xyx_inner(matrix);
    }
    throw std::invalid_argument{"Unknown EulerBasis for angles_from_unitary"};
  }

  static std::array<fp, 4> params_zyz_inner(const matrix2x2& matrix) {
    auto getIndex = [](auto x, auto y) { return (y * 2) + x; };
    auto determinant = [getIndex](auto&& matrix) {
      return (matrix.at(getIndex(0, 0)) * matrix.at(getIndex(1, 1))) -
             (matrix.at(getIndex(1, 0)) * matrix.at(getIndex(0, 1)));
    };

    auto detArg = std::arg(determinant(matrix));
    auto phase = 0.5 * detArg;
    auto theta = 2. * std::atan2(std::abs(matrix.at(getIndex(1, 0))),
                                 std::abs(matrix.at(getIndex(0, 0))));
    auto ang1 = std::arg(matrix.at(getIndex(1, 1)));
    auto ang2 = std::arg(matrix.at(getIndex(1, 0)));
    auto phi = ang1 + ang2 - detArg;
    auto lam = ang1 - ang2;
    return {theta, phi, lam, phase};
  }

  static std::array<fp, 4> params_xyx_inner(const matrix2x2& matrix) {
    auto mat_zyz = std::array{
        static_cast<fp>(0.5) *
            (matrix.at(0 * 2 + 0) + matrix.at(0 * 2 + 1 * 2 * 2) +
             matrix.at(1 * 2 + 0) + matrix.at(1 * 2 + 1)),
        static_cast<fp>(0.5) *
            (matrix.at(0 * 2 + 0) - matrix.at(0 * 2 + 1 * 2 * 2) +
             matrix.at(1 * 2 + 0) - matrix.at(1 * 2 + 1)),
        static_cast<fp>(0.5) *
            (matrix.at(0 * 2 + 0) + matrix.at(0 * 2 + 1 * 2 * 2) -
             matrix.at(1 * 2 + 0) - matrix.at(1 * 2 + 1)),
        static_cast<fp>(0.5) *
            (matrix.at(0 * 2 + 0) - matrix.at(0 * 2 + 1 * 2 * 2) -
             matrix.at(1 * 2 + 0) + matrix.at(1 * 2 + 1)),
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

  static constexpr std::array<qfp, 4> IPZ = {IM, C_ZERO, C_ZERO, M_IM};
  static constexpr std::array<qfp, 4> IPY = {C_ZERO, C_ONE, C_M_ONE, C_ZERO};
  static constexpr std::array<qfp, 4> IPX = {C_ZERO, IM, IM, C_ZERO};

  static matrix2x2 getSingleQubitMatrix(const QubitGateSequence::Gate& gate) {
    if (gate.type == qc::SX) {
      return {qfp{0.5, 0.5}, qfp{0.5, -0.5}, qfp{0.5, -0.5}, qfp{0.5, 0.5}};
    }
    if (gate.type == qc::RZ) {
      return rz_matrix(gate.parameter.at(0));
    }
    if (gate.type == qc::X) {
      return {0, 1, 1, 0};
    }
    throw std::invalid_argument{
        "unsupported gate type for single qubit matrix"};
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
      throw std::logic_error{"Invalid qubit ID in compute_unitary"};
    }
    if (gate.qubit_id.size() == 2) {
      if (gate.type == qc::X) {
        // controlled X (CX)
        if (gate.qubit_id == std::vector<std::size_t>{0, 1}) {
          return {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0};
        }
        if (gate.qubit_id == std::vector<std::size_t>{1, 0}) {
          return {1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0};
        }
      }
      throw std::invalid_argument{"unsupported gate type for two qubit matrix"};
    }
    throw std::logic_error{"Invalid number of qubit IDs in compute_unitary"};
  }

  struct TwoQubitWeylDecomposition {
    fp a;
    fp b;
    fp c;
    fp global_phase;
    std::array<qfp, 4> K1l;
    std::array<qfp, 4> K2l;
    std::array<qfp, 4> K1r;
    std::array<qfp, 4> K2r;
    Specialization specialization;
    EulerBasis default_euler_basis;
    std::optional<fp> requested_fidelity;
    fp calculated_fidelity;
    matrix4x4 unitary_matrix;

    static TwoQubitWeylDecomposition
    new_inner(matrix4x4 unitary_matrix, std::optional<fp> fidelity,
              std::optional<Specialization> _specialization) {
      using helpers::determinant;
      using helpers::diagonal;
      using helpers::dot;
      using helpers::transpose;
      auto& u = unitary_matrix;
      auto det_u = determinant(u);
      auto det_pow = std::pow(det_u, static_cast<fp>(-0.25));
      llvm::transform(u, u.begin(), [&](auto&& x) { return x * det_pow; });
      llvm::errs() << "===== U =====\n";
      helpers::print(u);
      auto global_phase = std::arg(det_u) / 4.;
      auto u_p = magic_basis_transform(u, MagicBasisTransform::OutOf);
      llvm::errs() << "===== U_P =====\n";
      helpers::print(u_p);
      auto m2 = dot(transpose(u_p), u_p);
      auto default_euler_basis = EulerBasis::ZYZ;
      llvm::errs() << "===== M2 =====\n";
      helpers::print(m2);

      arma::Mat<qfp> U(4, 4);
      for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
          U.at(j, i) = u_p[j * 4 + i];
        }
      }
      auto x = U.st() * U;
      std::cerr << "ARMA\n" << U.t() << "\n\n" << U << "\n\n" << x << std::endl;

      // M2 is a symmetric complex matrix. We need to decompose it as M2 = P D
      // P^T where P ∈ SO(4), D is diagonal with unit-magnitude elements.
      //
      // We can't use raw `eig` directly because it isn't guaranteed to give us
      // real or orthogonal eigenvectors. Instead, since `M2` is
      // complex-symmetric,
      //   M2 = A + iB
      // for real-symmetric `A` and `B`, and as
      //   M2^+ @ M2 = A^2 + B^2 + i [A, B] = 1
      // we must have `A` and `B` commute, and consequently they are
      // simultaneously diagonalizable. Mixing them together _should_ account
      // for any degeneracy problems, but it's not guaranteed, so we repeat it a
      // little bit.  The fixed seed is to make failures deterministic; the
      // value is not important.
      auto state = std::mt19937{2023};
      std::normal_distribution<fp> dist;
      auto found = false;
      diagonal4x4 d{{C_ZERO}};
      matrix4x4 p{{C_ZERO}};

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
        std::array<fp, 16> m2_real;
        llvm::transform(m2, m2_real.begin(), [&](const qfp& val) {
          return rand_a * val.real() + rand_b * val.imag();
        });
        rmatrix4x4 p_inner_real = self_adjoint_eigen_lower(m2_real).first;
        matrix4x4 p_inner;
        llvm::transform(p_inner_real, p_inner.begin(),
                        [](auto&& x) { return qfp(x, 0.0); });
        auto d_inner = diagonal(dot(dot(transpose(p_inner), m2), p_inner));

        llvm::errs() << "===== D_INNER =====\n";
        helpers::print(d_inner);
        llvm::errs() << "===== P_INNER =====\n";
        helpers::print(p_inner);
        matrix4x4 diag_d{}; // zero initialization
        diag_d[0 * 4 + 0] = d_inner[0];
        diag_d[1 * 4 + 1] = d_inner[1];
        diag_d[2 * 4 + 2] = d_inner[2];
        diag_d[3 * 4 + 3] = d_inner[3];

        auto compare = dot(dot(p_inner, diag_d), transpose(p_inner));
        llvm::errs() << "===== COMPARE =====\n";
        helpers::print(compare);
        found = llvm::all_of_zip(compare, m2, [](auto&& a, auto&& b) {
          return std::abs(a - b) <= 1.0e-13;
        });
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
      std::array<fp, d.size()> d_real;
      llvm::transform(d, d_real.begin(),
                      [](auto&& x) { return -std::arg(x) / 2.0; });
      helpers::print(d_real, "D_REAL");
      d_real[3] = -d_real[0] - d_real[1] - d_real[2];
      std::array<fp, 3> cs;
      for (std::size_t i = 0; i < cs.size(); ++i) {
        assert(i < d_real.size());
        cs[i] = remEuclid((d_real[i] + d_real[3]) / 2.0, qc::PI_2);
      }
      decltype(cs) cstemp;
      llvm::transform(cs, cstemp.begin(), [](auto&& x) {
        auto tmp = remEuclid(x, qc::PI_2);
        return std::min(tmp, qc::PI_2 - tmp);
      });
      std::array<std::size_t, cstemp.size()> order{0, 1, 2};
      llvm::stable_sort(order,
                        [&](auto a, auto b) { return cstemp[a] < cstemp[b]; });
      std::tie(order[0], order[1], order[2]) = {order[1], order[2], order[0]};
      std::tie(cs[0], cs[1], cs[2]) = {cs[order[0]], cs[order[1]],
                                       cs[order[2]]};
      std::tie(d_real[0], d_real[1], d_real[2]) = {
          d_real[order[0]], d_real[order[1]], d_real[order[2]]};

      // swap columns of p according to order
      constexpr auto P_ROW_LENGTH = 4;
      auto p_orig = p;
      for (std::size_t i = 0; i < order.size(); ++i) {
        for (std::size_t row = 0; row < P_ROW_LENGTH; ++row) {
          std::swap(p[row * 3 + i], p_orig[row * 3 + order[i]]);
        }
      }

      if (determinant(p).real() < 0.0) {
        // negate last column
        for (int i = 0; i < P_ROW_LENGTH; ++i) {
          auto& x = p[i * P_ROW_LENGTH + P_ROW_LENGTH - 1];
          x = -x;
        }
      }

      matrix4x4 temp{};
      temp[0 * 4 + 0] = std::exp(IM * d_real[0]);
      temp[1 * 4 + 1] = std::exp(IM * d_real[1]);
      temp[2 * 4 + 2] = std::exp(IM * d_real[2]);
      temp[3 * 4 + 3] = std::exp(IM * d_real[3]);
      helpers::print(temp, "TEMP");
      auto k1 = magic_basis_transform(dot(dot(u_p, p), temp),
                                      MagicBasisTransform::Into);
      auto k2 = magic_basis_transform(transpose(p), MagicBasisTransform::Into);

      auto [K1l, K1r, phase_l] = decompose_two_qubit_product_gate(k1);
      auto [K2l, K2r, phase_r] = decompose_two_qubit_product_gate(k2);
      global_phase += phase_l + phase_r;

      // Flip into Weyl chamber
      if (cs[0] > qc::PI_2) {
        cs[0] -= 3.0 * qc::PI_2;
        K1l = dot(K1l, IPY);
        K1r = dot(K1r, IPY);
        global_phase += qc::PI_2;
      }
      if (cs[1] > qc::PI_2) {
        cs[1] -= 3.0 * qc::PI_2;
        K1l = dot(K1l, IPX);
        K1r = dot(K1r, IPX);
        global_phase += qc::PI_2;
      }
      auto conjs = 0;
      if (cs[0] > qc::PI_4) {
        cs[0] = qc::PI_2 - cs[0];
        K1l = dot(K1l, IPY);
        K2r = dot(IPY, K2r);
        conjs += 1;
        global_phase -= qc::PI_2;
      }
      if (cs[1] > qc::PI_4) {
        cs[1] = qc::PI_2 - cs[1];
        K1l = dot(K1l, IPX);
        K2r = dot(IPX, K2r);
        conjs += 1;
        global_phase += qc::PI_2;
        if (conjs == 1) {
          global_phase -= qc::PI;
        }
      }
      if (cs[2] > qc::PI_2) {
        cs[2] -= 3.0 * qc::PI_2;
        K1l = dot(K1l, IPZ);
        K1r = dot(K1r, IPZ);
        global_phase += qc::PI_2;
        if (conjs == 1) {
          global_phase -= qc::PI;
        }
      }
      if (conjs == 1) {
        cs[2] = qc::PI_2 - cs[2];
        K1l = dot(K1l, IPZ);
        K2r = dot(IPZ, K2r);
        global_phase += qc::PI_2;
      }
      if (cs[2] > qc::PI_4) {
        cs[2] -= qc::PI_2;
        K1l = dot(K1l, IPZ);
        K1r = dot(K1r, IPZ);
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
          a,
          b,
          c,
          global_phase,
          K1l,
          K2l,
          K1r,
          K2r,
          Specialization::General,
          default_euler_basis,
          fidelity,
          -1.0,
          unitary_matrix,
      };
      auto get_specialized_decomposition = [&]() {
        // :math:`U \sim U_d(0,0,0) \sim Id`
        //
        // This gate binds 0 parameters, we make it canonical by
        // setting
        // :math:`K2_l = Id` , :math:`K2_r = Id`.
        if (specialization == Specialization::IdEquiv) {
          return TwoQubitWeylDecomposition{
              0.,
              0.,
              0.,
              general.global_phase,
              dot(general.K1l, general.K2l),
              identityGate,
              dot(general.K1r, general.K2r),
              identityGate,
              specialization,
              general.default_euler_basis,
              general.requested_fidelity,
              general.calculated_fidelity,
              general.unitary_matrix,
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
                qc::PI_4,
                qc::PI_4,
                qc::PI_4,
                general.global_phase,
                dot(general.K1l, general.K2r),
                identityGate,
                dot(general.K1r, general.K2l),
                identityGate,
                specialization,
                general.default_euler_basis,
                general.requested_fidelity,
                general.calculated_fidelity,
                general.unitary_matrix,
            };
          } else {
            flipped_from_original = true;
            return TwoQubitWeylDecomposition{
                qc::PI_4,
                qc::PI_4,
                qc::PI_4,
                global_phase + qc::PI_2,
                dot(dot(general.K1l, IPZ), general.K2r),
                identityGate,
                dot(dot(general.K1r, IPZ), general.K2l),
                identityGate,
                specialization,
                general.default_euler_basis,
                general.requested_fidelity,
                general.calculated_fidelity,
                general.unitary_matrix,
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
          auto k2l_dag = transpose(general.K2l);
          llvm::transform(k2l_dag, k2l_dag.begin(),
                          [](auto&& x) { return std::conj(x); });

          return TwoQubitWeylDecomposition{
              closest,
              closest,
              closest,
              general.global_phase,
              dot(general.K1l, general.K2l),
              identityGate,
              dot(general.K1r, general.K2l),
              dot(k2l_dag, general.K2r),
              specialization,
              general.default_euler_basis,
              general.requested_fidelity,
              general.calculated_fidelity,
              general.unitary_matrix,
          };
        }
        // :math:`U \sim U_d(\alpha\pi/4, \alpha\pi/4, -\alpha\pi/4) \sim
        // \text{SWAP}^\alpha`
        //
        // (a non-equivalent root of SWAP from the TwoQubitWeylPartialSWAPEquiv
        // similar to how :math:`x = (\pm \sqrt(x))^2`)
        //
        // This gate binds 3 parameters, we make it canonical by setting:
        //
        // :math:`K2_l = Id`
        if (specialization == Specialization::PartialSWAPFlipEquiv) {
          auto closest = closest_partial_swap(a, b, -c);
          auto k2l_dag = transpose(general.K2l);
          llvm::transform(k2l_dag, k2l_dag.begin(),
                          [](auto&& x) { return std::conj(x); });

          return TwoQubitWeylDecomposition{
              closest,
              closest,
              -closest,
              general.global_phase,
              dot(general.K1l, general.K2l),
              identityGate,
              dot(dot(dot(general.K1r, IPZ), general.K2l), IPZ),
              dot(dot(dot(IPZ, k2l_dag), IPZ), general.K2r),
              specialization,
              general.default_euler_basis,
              general.requested_fidelity,
              general.calculated_fidelity,
              general.unitary_matrix,
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
              a,
              0.,
              0.,
              global_phase + k2lphase + k2rphase,
              dot(general.K1l, rx_matrix(k2lphi)),
              dot(ry_matrix(k2ltheta), rx_matrix(k2llambda)),
              dot(general.K1r, rx_matrix(k2rphi)),
              dot(ry_matrix(k2rtheta), rx_matrix(k2rlambda)),
              specialization,
              euler_basis,
              general.requested_fidelity,
              general.calculated_fidelity,
              general.unitary_matrix,
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
          TwoQubitWeylDecomposition{
              qc::PI_4,
              qc::PI_4,
              c,
              global_phase + k2lphase + k2rphase,
              dot(general.K1l, rz_matrix(k2rphi)),
              dot(ry_matrix(k2ltheta), rz_matrix(k2llambda)),
              dot(general.K1r, rz_matrix(k2lphi)),
              dot(ry_matrix(k2rtheta), rz_matrix(k2rlambda)),
              specialization,
              general.default_euler_basis,
              general.requested_fidelity,
              general.calculated_fidelity,
              general.unitary_matrix,
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
              (a + b) / 2.,
              (a + b) / 2.,
              c,
              global_phase + k2lphase,
              dot(general.K1l, rz_matrix(k2lphi)),
              dot(ry_matrix(k2ltheta), rz_matrix(k2llambda)),
              dot(general.K1r, rz_matrix(k2lphi)),
              dot(rz_matrix(-k2lphi), general.K2r),
              specialization,
              general.default_euler_basis,
              general.requested_fidelity,
              general.calculated_fidelity,
              general.unitary_matrix,
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
              a,
              (b + c) / 2.,
              (b + c) / 2.,
              global_phase + k2lphase,
              dot(general.K1l, rx_matrix(k2lphi)),
              dot(ry_matrix(k2ltheta), rx_matrix(k2llambda)),
              dot(general.K1r, rx_matrix(k2lphi)),
              dot(rx_matrix(-k2lphi), general.K2r),
              specialization,
              euler_basis,
              general.requested_fidelity,
              general.calculated_fidelity,
              general.unitary_matrix,
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
              a,
              (b - c) / 2.,
              -((b - c) / 2.),
              global_phase + k2lphase,
              dot(general.K1l, rx_matrix(k2lphi)),
              dot(ry_matrix(k2ltheta), rx_matrix(k2llambda)),
              dot(dot(dot(general.K1r, IPZ), rx_matrix(k2lphi)), IPZ),
              dot(dot(dot(IPZ, rx_matrix(-k2lphi)), IPZ), general.K2r),
              specialization,
              euler_basis,
              general.requested_fidelity,
              general.calculated_fidelity,
              general.unitary_matrix,
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
    qc::OpType gate;
    std::vector<fp> gate_params;
    fp basis_fidelity;
    EulerBasis euler_basis;
    std::optional<bool> pulse_optimize;
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
    new_inner(qc::OpType gate = qc::X, // CX
              const std::vector<fp>& gate_params = {},
              matrix4x4 gate_matrix = {1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1,
                                       0, 0}, // CX matrix
              fp basis_fidelity = 1.0, EulerBasis euler_basis = EulerBasis::ZYZ,
              std::optional<bool> pulse_optimize = std::nullopt) {
      using helpers::dot;
      using helpers::transpose_conjugate;

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
      constexpr auto K12R_ARR = std::array{
          qfp(0., FRAC_1_SQRT_2),
          qfp(FRAC_1_SQRT_2, 0.),
          qfp(-FRAC_1_SQRT_2, 0.),
          qfp(0., -FRAC_1_SQRT_2),
      };
      constexpr auto K12L_ARR = std::array{
          qfp(0.5, 0.5),
          qfp(0.5, 0.5),
          qfp(-0.5, 0.5),
          qfp(0.5, -0.5),
      };

      auto basis_decomposer = TwoQubitWeylDecomposition::new_inner(
          gate_matrix, DEFAULT_FIDELITY, std::nullopt);
      auto super_controlled =
          relative_eq(basis_decomposer.a, qc::PI_4,
                      std::numeric_limits<fp>::epsilon(), 1e-09) &&
          relative_eq(basis_decomposer.c, 0.0,
                      std::numeric_limits<fp>::epsilon(), 1e-09);

      // Create some useful matrices U1, U2, U3 are equivalent to the basis,
      // expand as Ui = Ki1.Ubasis.Ki2
      auto b = basis_decomposer.b;
      auto temp = qfp(0.5, -0.5);
      auto k11l = std::array{
          temp * (M_IM * std::exp(qfp(0., -b))), temp * std::exp(qfp(0., -b)),
          temp * (M_IM * std::exp(qfp(0., b))), temp * -(std::exp(qfp(0., b)))};
      auto k11r = std::array{FRAC_1_SQRT_2 * std::exp((IM * qfp(0., -b))),
                             FRAC_1_SQRT_2 * -std::exp(qfp(0., -b)),
                             FRAC_1_SQRT_2 * std::exp(qfp(0., b)),
                             FRAC_1_SQRT_2 * (M_IM * std::exp(qfp(0., b)))};
      auto k32l_k21l = std::array{FRAC_1_SQRT_2 * std::cos(qfp(1., (2. * b))),
                                  FRAC_1_SQRT_2 * (IM * std::sin((2. * b))),
                                  FRAC_1_SQRT_2 * (IM * std::sin(2. * b)),
                                  FRAC_1_SQRT_2 * qfp(1., -std::cos(2. * b))};
      temp = qfp(0.5, 0.5);
      auto k21r = std::array{
          temp * (M_IM * std::exp(qfp(0., -2. * b))),
          temp * std::exp(qfp(0., -2. * b)),
          temp * (IM * std::exp(qfp(0., 2. * b))),
          temp * std::exp(qfp(0., 2. * b)),
      };
      constexpr auto K22L_ARR = std::array{
          qfp(FRAC_1_SQRT_2, 0.),
          qfp(-FRAC_1_SQRT_2, 0.),
          qfp(FRAC_1_SQRT_2, 0.),
          qfp(FRAC_1_SQRT_2, 0.),
      };
      constexpr auto K22R_ARR = std::array{C_ZERO, C_ONE, C_M_ONE, C_ZERO};
      auto k31l = std::array{
          FRAC_1_SQRT_2 * std::exp(qfp(0., -b)),
          FRAC_1_SQRT_2 * std::exp(qfp(0., -b)),
          FRAC_1_SQRT_2 * -std::exp(qfp(0., b)),
          FRAC_1_SQRT_2 * std::exp(qfp(0., b)),
      };
      auto k31r = std::array{
          IM * std::exp(qfp(0., b)),
          C_ZERO,
          C_ZERO,
          M_IM * std::exp(qfp(0., -b)),
      };
      temp = qfp(0.5, 0.5);
      auto k32r = std::array{
          temp * std::exp(qfp(0., b)),
          temp * -std::exp(qfp(0., -b)),
          temp * (M_IM * std::exp(qfp(0., b))),
          temp * (M_IM * std::exp(qfp(0., -b))),
      };
      auto k1ld = transpose_conjugate(basis_decomposer.K1l);
      auto k1rd = transpose_conjugate(basis_decomposer.K1r);
      auto k2ld = transpose_conjugate(basis_decomposer.K2l);
      auto k2rd = transpose_conjugate(basis_decomposer.K2r);
      // Pre-build the fixed parts of the matrices used in 3-part decomposition
      auto u0l = dot(k31l, k1ld);
      auto u0r = dot(k31r, k1rd);
      auto u1l = dot(dot(k2ld, k32l_k21l), k1ld);
      auto u1ra = dot(k2rd, k32r);
      auto u1rb = dot(k21r, k1rd);
      auto u2la = dot(k2ld, K22L_ARR);
      auto u2lb = dot(k11l, k1ld);
      auto u2ra = dot(k2rd, K22R_ARR);
      auto u2rb = dot(k11r, k1rd);
      auto u3l = dot(k2ld, K12L_ARR);
      auto u3r = dot(k2rd, K12R_ARR);
      // Pre-build the fixed parts of the matrices used in the 2-part
      // decomposition
      auto q0l = dot(transpose_conjugate(K12L_ARR), k1ld);
      auto q0r = dot(dot(transpose_conjugate(K12R_ARR), IPZ), k1rd);
      auto q1la = dot(k2ld, transpose_conjugate(k11l));
      auto q1lb = dot(k11l, k1ld);
      auto q1ra = dot(dot(k2rd, IPZ), transpose_conjugate(k11r));
      auto q1rb = dot(k11r, k1rd);
      auto q2l = dot(k2ld, K12L_ARR);
      auto q2r = dot(k2rd, K12R_ARR);

      return TwoQubitBasisDecomposer{
          gate,
          gate_params,
          basis_fidelity,
          euler_basis,
          pulse_optimize,
          basis_decomposer,
          super_controlled,
          u0l,
          u0r,
          u1l,
          u1ra,
          u1rb,
          u2la,
          u2lb,
          u2ra,
          u2rb,
          u3l,
          u3r,
          q0l,
          q0r,
          q1la,
          q1lb,
          q1ra,
          q1rb,
          q2l,
          q2r,
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
        auto minValue = std::numeric_limits<fp>::max();
        auto minIndex = -1;
        for (std::size_t i = 0; i < traces.size(); ++i) {
          auto value = trace_to_fid(traces[i]) * std::pow(basis_fidelity, i);
          if (value < minValue) {
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
      if (pulse_optimize.value_or(true)) {
        return pulse_optimal_chooser(best_nbasis, decomposition,
                                     target_decomposed);
      }
      std::vector<EulerBasis> target_1q_basis_list;
      target_1q_basis_list.push_back(euler_basis);
      llvm::SmallVector<std::optional<TwoQubitGateSequence>, 8>
          euler_decompositions;
      for (auto&& decomp : decomposition) {
        auto euler_decomp = unitary_to_gate_sequence_inner(
            decomp, target_1q_basis_list, 0, {}, true, std::nullopt);
        euler_decompositions.push_back(euler_decomp);
      }
      TwoQubitGateSequence gates{.globalPhase = target_decomposed.global_phase};
      // Worst case length is 5x 1q gates for each 1q decomposition + 1x 2q gate
      // We might overallocate a bit if the euler basis is different but
      // the worst case is just 16 extra elements with just a String and 2
      // smallvecs each. This is only transient though as the circuit sequences
      // aren't long lived and are just used to create a QuantumCircuit or
      // DAGCircuit when we return to Python space.
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

        gates.gates.push_back(
            {.type = gate, .parameter = gate_params, .qubit_id = {0, 1}});
      }

      add_euler_decomposition(2 * best_nbasis, 0);
      add_euler_decomposition(2 * best_nbasis + 1, 1);

      return gates;
    }

  private:
    [[nodiscard]] std::vector<matrix2x2>
    decomp0_inner(const TwoQubitWeylDecomposition& target) const {
      using helpers::dot;
      return {
          dot(target.K1r, target.K2r),
          dot(target.K1l, target.K2l),
      };
    }

    [[nodiscard]] std::vector<matrix2x2>
    decomp1_inner(const TwoQubitWeylDecomposition& target) const {
      using helpers::dot;
      using helpers::transpose_conjugate;
      // FIXME: fix for z!=0 and c!=0 using closest reflection (not always in
      // the Weyl chamber)
      return {
          dot(transpose_conjugate(basis_decomposer.K2r), target.K2r),
          dot(transpose_conjugate(basis_decomposer.K2l), target.K2l),
          dot(target.K1r, transpose_conjugate(basis_decomposer.K1r)),
          dot(target.K1l, transpose_conjugate(basis_decomposer.K1l)),
      };
    }

    [[nodiscard]] std::vector<matrix2x2> decomp2_supercontrolled_inner(
        const TwoQubitWeylDecomposition& target) const {
      using helpers::dot;
      return {
          dot(q2r, target.K2r),
          dot(q2l, target.K2l),
          dot(dot(q1ra, rz_matrix(2. * target.b)), q1rb),
          dot(dot(q1la, rz_matrix(-2. * target.a)), q1lb),
          dot(target.K1r, q0r),
          dot(target.K1l, q0l),
      };
    }

    [[nodiscard]] std::vector<matrix2x2> decomp3_supercontrolled_inner(
        const TwoQubitWeylDecomposition& target) const {
      using helpers::dot;
      return {
          dot(u3r, target.K2r),
          dot(u3l, target.K2l),
          dot(dot(u2ra, rz_matrix(2. * target.b)), u2rb),
          dot(dot(u2la, rz_matrix(-2. * target.a)), u2lb),
          dot(dot(u1ra, rz_matrix(-2. * target.c)), u1rb),
          u1l,
          dot(target.K1r, u0r),
          dot(target.K1l, u0l),
      };
    }

    std::optional<TwoQubitGateSequence>
    pulse_optimal_chooser(std::uint8_t best_nbasis,
                          std::vector<matrix2x2> decomposition,
                          const TwoQubitWeylDecomposition& target_decomposed) {
      if (pulse_optimize.has_value() &&
          (best_nbasis == 0 || best_nbasis == 1 || best_nbasis > 3)) {
        return std::nullopt;
      }
      if (euler_basis != EulerBasis::ZSX && euler_basis != EulerBasis::ZSXX) {
        if (pulse_optimize.has_value()) {
          throw std::runtime_error{
              "'pulse_optimize' currently only works with ZSX basis"};
        }
        return std::nullopt;
      }

      if (gate != qc::X) { // CX
        if (pulse_optimize.has_value()) {
          throw std::runtime_error{
              "pulse_optimizer currently only works with CNOT entangling gate"};
        }
        return std::nullopt;
      }
      std::optional<TwoQubitGateSequence> result;
      if (best_nbasis == 3) {
        result =
            get_sx_vz_3cx_efficient_euler(decomposition, target_decomposed);
      } else if (best_nbasis == 2) {
        result =
            get_sx_vz_2cx_efficient_euler(decomposition, target_decomposed);
      }
      if (pulse_optimize.has_value() && !result.has_value()) {
        throw std::runtime_error{
            "Failed to compute requested pulse optimal decomposition"};
      }
      return result;
    }

    ///
    /// Decomposition of SU(4) gate for device with SX, virtual RZ, and CNOT
    /// gates assuming two CNOT gates are needed.
    ///
    /// This first decomposes each unitary from the KAK decomposition into ZXZ
    /// on the source qubit of the CNOTs and XZX on the targets in order to
    /// commute operators to beginning and end of decomposition. The beginning
    /// and ending single qubit gates are then collapsed and re-decomposed with
    /// the single qubit decomposer. This last step could be avoided if
    /// performance is a concern.
    TwoQubitGateSequence get_sx_vz_2cx_efficient_euler(
        const std::vector<matrix2x2>& decomposition,
        const TwoQubitWeylDecomposition& target_decomposed) {
      using helpers::dot;
      TwoQubitGateSequence gates{.globalPhase = target_decomposed.global_phase};
      gates.globalPhase -= 2. * basis_decomposer.global_phase;

      auto get_euler_angles = [&](std::size_t startIndex, EulerBasis basis) {
        std::vector<std::array<fp, 3>> result;
        for (std::size_t i = startIndex; i < decomposition.size(); i += 2) {
          auto euler_angles = angles_from_unitary(decomposition[i], basis);
          gates.globalPhase += euler_angles[3];
          result.push_back({euler_angles[2], euler_angles[0], euler_angles[1]});
        }
        return result;
      };

      auto euler_q0 = get_euler_angles(0, EulerBasis::ZXZ);
      auto euler_q1 = get_euler_angles(1, EulerBasis::XZX);

      auto euler_matrix_q0 =
          dot(rx_matrix(euler_q0[0][1]), rz_matrix(euler_q0[0][0]));
      euler_matrix_q0 =
          dot(rz_matrix(euler_q0[0][2] + euler_q0[1][0] + qc::PI_2),
              euler_matrix_q0);
      append_1q_sequence(gates, euler_matrix_q0, 0);
      auto euler_matrix_q1 =
          dot(rz_matrix(euler_q1[0][1]), rx_matrix(euler_q1[0][0]));
      euler_matrix_q1 =
          dot(rx_matrix(euler_q1[0][2] + euler_q1[1][0]), euler_matrix_q1);
      append_1q_sequence(gates, euler_matrix_q1, 1);
      gates.gates.push_back(
          {.type = qc::X, .qubit_id = {0, 1}}); // TODO: mark CX somehow?
      gates.gates.push_back({.type = qc::SX, .qubit_id = {0}});
      gates.gates.push_back({.type = qc::RZ,
                             .parameter = {euler_q0[1][1] - qc::PI},
                             .qubit_id = {0}});
      gates.gates.push_back({.type = qc::SX, .qubit_id = {0}});
      gates.gates.push_back(
          {.type = qc::RZ, .parameter = {euler_q1[1][1]}, .qubit_id = {1}});
      gates.globalPhase += qc::PI_2;
      gates.gates.push_back(
          {.type = qc::X, .qubit_id = {0, 1}}); // TODO: mark CX somehow?
      euler_matrix_q0 =
          dot(rx_matrix(euler_q0[2][1]),
              rz_matrix(euler_q0[1][2] + euler_q0[2][0] + qc::PI_2));
      euler_matrix_q0 = dot(rz_matrix(euler_q0[2][2]), euler_matrix_q0);
      append_1q_sequence(gates, euler_matrix_q0, 0);
      euler_matrix_q1 = dot(rz_matrix(euler_q1[2][1]),
                            rx_matrix(euler_q1[1][2] + euler_q1[2][0]));
      euler_matrix_q1 = dot(rx_matrix(euler_q1[2][2]), euler_matrix_q1);
      append_1q_sequence(gates, euler_matrix_q1, 1);
      return gates;
    }

    /// Decomposition of SU(4) gate for device with SX, virtual RZ, and CNOT
    /// gates assuming three CNOT gates are needed.
    ///
    /// This first decomposes each unitary from the KAK decomposition into ZXZ
    /// on the source qubit of the CNOTs and XZX on the targets in order commute
    /// operators to beginning and end of decomposition. Inserting Hadamards
    /// reverses the direction of the CNOTs and transforms a variable Rx ->
    /// variable virtual Rz. The beginning and ending single qubit gates are
    /// then collapsed and re-decomposed with the single qubit decomposer. This
    /// last step could be avoided if performance is a concern.
    std::optional<TwoQubitGateSequence> get_sx_vz_3cx_efficient_euler(
        const std::vector<matrix2x2>& decomposition,
        const TwoQubitWeylDecomposition& target_decomposed) {
      using helpers::dot;
      TwoQubitGateSequence gates{.globalPhase = target_decomposed.global_phase};
      gates.globalPhase -= 3. * basis_decomposer.global_phase;
      gates.globalPhase = remEuclid(gates.globalPhase, qc::TAU);
      auto atol = 1e-10; // absolute tolerance for floats
                         // Decompose source unitaries to zxz

      auto get_euler_angles = [&](std::size_t startIndex, EulerBasis basis) {
        std::vector<std::array<fp, 3>> result;
        for (std::size_t i = startIndex; i < decomposition.size(); i += 2) {
          auto euler_angles = angles_from_unitary(decomposition[i], basis);
          gates.globalPhase += euler_angles[3];
          result.push_back({euler_angles[2], euler_angles[0], euler_angles[1]});
        }
        return result;
      };

      auto euler_q0 = get_euler_angles(0, EulerBasis::ZXZ);
      // Decompose target unitaries to xzx
      auto euler_q1 = get_euler_angles(1, EulerBasis::XZX);

      auto x12 = euler_q0[1][2] + euler_q0[2][0];
      auto x12_is_non_zero = std::abs(x12) < atol;
      auto x12_is_old_mult = std::abs(std::cos(x12) - 1.0) < atol;
      auto x12_phase = 0.;
      auto x12_is_pi_mult = std::abs(std::sin(x12)) < atol;
      if (x12_is_pi_mult) {
        x12_phase = qc::PI * std::cos(x12);
      }
      auto x02_add = x12 - euler_q0[1][0];
      auto x12_is_half_pi = std::abs(x12 - qc::PI_2) < atol;

      auto euler_matrix_q0 =
          dot(rx_matrix(euler_q0[0][1]), rz_matrix(euler_q0[0][0]));
      if (x12_is_non_zero && x12_is_pi_mult) {
        euler_matrix_q0 =
            dot(rz_matrix(euler_q0[0][2] - x02_add), euler_matrix_q0);
      } else {
        euler_matrix_q0 =
            dot(rz_matrix(euler_q0[0][2] + euler_q0[1][0]), euler_matrix_q0);
      }
      euler_matrix_q0 = dot(hGate, euler_matrix_q0);
      append_1q_sequence(gates, euler_matrix_q0, 0);

      auto rx_0 = rx_matrix(euler_q1[0][0]);
      auto rz = rz_matrix(euler_q1[0][1]);
      auto rx_1 = rx_matrix(euler_q1[0][2] + euler_q1[1][0]);
      auto euler_matrix_q1 = dot(rz, rx_0);
      euler_matrix_q1 = dot(rx_1, euler_matrix_q1);
      euler_matrix_q1 = dot(hGate, euler_matrix_q1);
      append_1q_sequence(gates, euler_matrix_q1, 1);

      gates.gates.push_back({.type = qc::X, .qubit_id = {1, 0}});

      if (x12_is_pi_mult) {
        // even or odd multiple
        if (x12_is_non_zero) {
          gates.globalPhase += x12_phase;
        }
        if (x12_is_non_zero && x12_is_old_mult) {
          gates.gates.push_back({.type = qc::RZ,
                                 .parameter = {-euler_q0[1][1]},
                                 .qubit_id = {0}});
        } else {
          gates.gates.push_back(
              {.type = qc::RZ, .parameter = {euler_q0[1][1]}, .qubit_id = {0}});
          gates.globalPhase += qc::PI;
        }
      }
      if (x12_is_half_pi) {
        gates.gates.push_back({.type = qc::SX, .qubit_id = {0}});
        gates.globalPhase -= qc::PI_4;
      } else if (x12_is_non_zero && !x12_is_pi_mult) {
        if (!pulse_optimize.has_value()) {
          append_1q_sequence(gates, rx_matrix(x12), 0);
        } else {
          return std::nullopt;
        }
      }
      if (std::abs(euler_q1[1][1] - qc::PI_2) < atol) {
        gates.gates.push_back({.type = qc::SX, .qubit_id = {1}});
        gates.globalPhase -= qc::PI_4;
      } else if (!pulse_optimize.has_value()) {
        append_1q_sequence(gates, rx_matrix(euler_q1[1][1]), 1);
      } else {
        return std::nullopt;
      }
      gates.gates.push_back({.type = qc::RZ,
                             .parameter = {euler_q1[1][2] + euler_q1[2][0]},
                             .qubit_id = {1}});
      gates.gates.push_back({.type = qc::X, .qubit_id = {1, 0}});
      gates.gates.push_back(
          {.type = qc::RZ, .parameter = {euler_q1[2][1]}, .qubit_id = {0}});
      if (std::abs(euler_q1[2][1] - qc::PI_2) < atol) {
        gates.gates.push_back({.type = qc::SX, .qubit_id = {1}});
        gates.globalPhase -= qc::PI_4;
      } else if (!pulse_optimize.has_value()) {
        append_1q_sequence(gates, rx_matrix(euler_q1[2][1]), 1);
      } else {
        return std::nullopt;
      }
      gates.gates.push_back({.type = qc::X, .qubit_id = {1, 0}});
      auto eulerMatrix = dot(rz_matrix(euler_q0[2][2] + euler_q0[3][0]), hGate);
      eulerMatrix = dot(rx_matrix(euler_q0[3][1]), eulerMatrix);
      eulerMatrix = dot(rz_matrix(euler_q0[3][2]), eulerMatrix);
      append_1q_sequence(gates, eulerMatrix, 0);

      eulerMatrix = dot(rx_matrix(euler_q1[2][2] + euler_q1[3][0]), hGate);
      eulerMatrix = dot(rz_matrix(euler_q1[3][1]), eulerMatrix);
      eulerMatrix = dot(rx_matrix(euler_q1[3][2]), eulerMatrix);
      append_1q_sequence(gates, eulerMatrix, 1);

      auto out_unitary = compute_unitary(gates, gates.globalPhase);
      // TODO: fix the sign problem to avoid correction here
      if (std::abs(target_decomposed.unitary_matrix.at(0 * 4 + 0) -
                   (-out_unitary.at(0 * 4 + 0))) < atol) {
        gates.globalPhase += qc::PI;
      }
      return gates;
    }

    matrix4x4 compute_unitary(const TwoQubitGateSequence& sequence,
                              fp global_phase) {
      using helpers::dot;
      auto phase = std::exp(std::complex<fp>{0, global_phase});
      matrix4x4 matrix;
      matrix[0 * 4 + 0] = phase;
      matrix[1 * 4 + 1] = phase;
      matrix[2 * 4 + 2] = phase;
      matrix[3 * 4 + 3] = phase;

      for (auto&& gate : sequence.gates) {
        matrix4x4 gate_matrix = getTwoQubitMatrix(gate);

        matrix = dot(gate_matrix, matrix);
      }
      return matrix;
    }

    void append_1q_sequence(TwoQubitGateSequence& twoQubitSequence,
                            matrix2x2 unitary, std::uint8_t qubit) {
      std::vector<EulerBasis> target_1q_basis_list;
      target_1q_basis_list.push_back(euler_basis);
      auto sequence = unitary_to_gate_sequence_inner(
          unitary, target_1q_basis_list, qubit, {}, true, std::nullopt);
      twoQubitSequence.globalPhase += sequence.globalPhase;
      for (auto&& gate : sequence.gates) {
        twoQubitSequence.gates.push_back({.type = gate.type,
                                          .parameter = gate.parameter,
                                          .qubit_id = {qubit}});
      }
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
            error_map, // TODO: remove error_map+qubit for platform independence
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
     *       Any modifications or derivative works of this code must retain this
     *       copyright notice, and modified files need to carry a notice
     *       indicating that they have been altered from the originals.
     */
    [[nodiscard]] static OneQubitGateSequence
    calculateRotationGates(fp theta, fp phi, fp lambda, fp phase,
                           qc::OpType kGate, qc::OpType aGate, bool simplify,
                           std::optional<fp> atol) {
      fp angleZeroEpsilon = atol.value_or(1e-12);
      if (simplify) {
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

/**
 * @brief Populates the given pattern set with patterns for gate
 * decomposition.
 */
void populateGateDecompositionPatterns(mlir::RewritePatternSet& patterns) {
  patterns.add<GateDecompositionPattern>(patterns.getContext());
}

} // namespace mqt::ir::opt

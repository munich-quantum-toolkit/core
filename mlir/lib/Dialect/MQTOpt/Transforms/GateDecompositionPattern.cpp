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

  dd::TwoQubitGateMatrix twoQubitIdentity = {
      {{1, 0, 0, 0}, {0, 0, 1, 0}, {0, 1, 0, 0}, {0, 0, 0, 1}}};

  mlir::LogicalResult
  matchAndRewrite(UnitaryInterface op,
                  mlir::PatternRewriter& rewriter) const override {
    auto series = getTwoQubitSeries(op);
    if (series.size() <= 3) {
      return mlir::failure();
    }

    dd::TwoQubitGateMatrix unitaryMatrix = dd::opToTwoQubitGateMatrix(qc::I);
    for (auto&& gate : series) {
      if (auto gateMatrix = helpers::getUnitaryMatrix(gate)) {
        unitaryMatrix = helpers::multiply(unitaryMatrix, *gateMatrix);
      }
    }

    twoQubitDecompose(unitaryMatrix);

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
    while (true) {
      for (auto&& user : op->getUsers()) {
        auto userUnitary = llvm::cast<UnitaryInterface>(user);
        if (helpers::isSingleQubitOperation(userUnitary)) {
          auto&& operand = userUnitary->getOperand(0);
          auto* it = llvm::find(qubits, operand);
          if (it == qubits.end()) {
            return result;
          }
          *it = userUnitary->getResult(0);

          result.push_back(userUnitary);
        } else if (helpers::isTwoQubitOperation(userUnitary)) {
          auto&& firstOperand = userUnitary->getOperand(0);
          auto&& secondOperand = userUnitary->getOperand(1);
          auto* firstQubitIt = llvm::find(qubits, firstOperand);
          auto* secondQubitIt = llvm::find(qubits, secondOperand);
          if (firstQubitIt == qubits.end() || secondQubitIt == qubits.end()) {
            return result;
          }
          *firstQubitIt = userUnitary->getResult(0);
          *secondQubitIt = userUnitary->getResult(1);

          result.push_back(userUnitary);
        } else {
          return result;
        }
      }
      return result;
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

  using qfp = std::complex<qc::fp>;
  using diagonal4x4 = std::array<qfp, 4>;
  using vector2d = std::vector<qfp>;
  using matrix2x2 = std::array<qfp, 4>;
  using matrix4x4 = std::array<qfp, 16>;

  static constexpr matrix2x2 identityGate = {1, 0, 0, 1};

  static qc::fp remEuclid(qc::fp a, qc::fp b) {
    auto r = std::fmod(a, b);
    return (r < 0.0) ? r + std::abs(b) : r;
  }

  static matrix2x2 dot(const matrix2x2& lhs, const matrix2x2& rhs) {
    return lhs;
  }
  static matrix4x4 dot(const matrix4x4& lhs, const matrix4x4& rhs) {
    return lhs;
  }

  static matrix2x2 transpose(const matrix2x2& x) { return x; }
  static matrix4x4 transpose(const matrix4x4& x) { return x; }

  static qfp determinant(const matrix2x2& x) { return 0.0; };
  static qfp determinant(const matrix4x4& x) { return 0.0; };

  static matrix2x2 multiply(qfp factor, matrix2x2 matrix) {
    llvm::transform(matrix, matrix.begin(),
                    [&](auto&& x) { return factor * x; });
    return matrix;
  }

  static matrix4x4 kroneckerProduct(const matrix2x2& lhs,
                                    const matrix2x2& rhs) {
    return from(multiply(lhs[0 * 2 + 0], rhs), multiply(lhs[0 * 2 + 1], rhs),
                multiply(lhs[1 * 2 + 0], rhs), multiply(lhs[1 * 2 + 1], rhs));
  }

  static matrix4x4 from(const matrix2x2& first_quadrant,
                        const matrix2x2& second_quadrant,
                        const matrix2x2& third_quadrant,
                        const matrix2x2& fourth_quadrant) {
    return {
        first_quadrant[0 * 2 + 0],  first_quadrant[0 * 2 + 1],
        second_quadrant[0 * 2 + 0], second_quadrant[0 * 2 + 1],
        first_quadrant[1 * 2 + 0],  first_quadrant[1 * 2 + 1],
        second_quadrant[1 * 2 + 0], second_quadrant[1 * 2 + 1],
        third_quadrant[0 * 2 + 0],  first_quadrant[0 * 2 + 1],
        fourth_quadrant[0 * 2 + 0], fourth_quadrant[0 * 2 + 1],
        third_quadrant[1 * 2 + 0],  first_quadrant[1 * 2 + 1],
        fourth_quadrant[1 * 2 + 0], fourth_quadrant[1 * 2 + 1],
    };
  }

  // https://docs.rs/faer/latest/faer/mat/generic/struct.Mat.html#method.self_adjoint_eigen
  static matrix4x4 self_adjoint_eigen_lower(const matrix4x4& x) { return x; }

  static std::tuple<matrix2x2, matrix2x2, qc::fp>
  decompose_two_qubit_product_gate(matrix4x4 special_unitary) {
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
    matrix2x2 r_t_conj;
    llvm::transform(transpose(r), r_t_conj.begin(),
                    [](auto&& x) { return std::conj(x); });

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

  static diagonal4x4 diagonal(const matrix4x4& matrix) {
    return {matrix[0 * 4 + 0], matrix[1 * 4 + 1], matrix[2 * 4 + 2],
            matrix[3 * 4 + 3]};
  }

  static matrix4x4 magic_basis_transform(const matrix4x4& unitary,
                                         MagicBasisTransform direction) {
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

  static qc::fp trace_to_fid(const qfp& x) {
    auto x_abs = std::abs(x);
    return (4.0 + x_abs * x_abs) / 20.0;
  }

  static qc::fp closest_partial_swap(qc::fp a, qc::fp b, qc::fp c) {
    auto m = (a + b + c) / 3.;
    auto [am, bm, cm] = std::array{a - m, b - m, c - m};
    auto [ab, bc, ca] = std::array{a - b, b - c, c - a};
    return m + am * bm * cm * (6. + ab * ab + bc * bc + ca * ca) / 18.;
  }

  static matrix2x2 rx_matrix(qc::fp theta) {
    auto half_theta = theta / 2.;
    auto cos = qfp(std::cos(half_theta), 0.);
    auto isin = qfp(0., -std::sin(half_theta));
    return {cos, isin, isin, cos};
  }

  static matrix2x2 ry_matrix(qc::fp theta) {
    auto half_theta = theta / 2.;
    auto cos = qfp(std::cos(half_theta), 0.);
    auto sin = qfp(std::sin(half_theta), 0.);
    return {cos, -sin, sin, cos};
  }

  static matrix2x2 rz_matrix(qc::fp theta) {
    auto ilam2 = qfp(0., 0.5 * theta);
    return {std::exp(-ilam2), C_ZERO, C_ZERO, std::exp(ilam2)};
  }

  static std::array<qc::fp, 4> angles_from_unitary(const matrix2x2& matrix,
                                                   EulerBasis basis) {
    if (basis == EulerBasis::XYX) {
      return params_xyx_inner(matrix);
    }
    throw std::invalid_argument{"Unknown EulerBasis for angles_from_unitary"};
  }

  static std::array<qc::fp, 4> params_xyx_inner(const matrix2x2& matrix) {}

  static constexpr std::complex<qc::fp> C_ZERO{0., 0.};
  static constexpr std::complex<qc::fp> C_ONE{1., 0.};
  static constexpr std::complex<qc::fp> C_M_ONE{-1., 0.};
  static constexpr std::complex<qc::fp> IM{0., 1.};
  static constexpr std::complex<qc::fp> M_IM{0., -1.};

  struct TwoQubitWeylDecomposition {
    qc::fp a;
    qc::fp b;
    qc::fp c;
    qc::fp global_phase;
    std::array<std::complex<qc::fp>, 4> K1l;
    std::array<std::complex<qc::fp>, 4> K2l;
    std::array<std::complex<qc::fp>, 4> K1r;
    std::array<std::complex<qc::fp>, 4> K2r;
    Specialization specialization;
    EulerBasis default_euler_basis;
    std::optional<qc::fp> requested_fidelity;
    qc::fp calculated_fidelity;
    matrix4x4 unitary_matrix;

    static TwoQubitWeylDecomposition
    new_inner(matrix4x4 unitary_matrix,

              std::optional<qc::fp> fidelity,
              std::optional<Specialization> _specialization) {
      constexpr std::array<std::complex<qc::fp>, 4> IPZ = {IM, C_ZERO, C_ZERO,
                                                           M_IM};
      constexpr std::array<std::complex<qc::fp>, 4> IPY = {C_ZERO, C_ONE,
                                                           C_M_ONE, C_ZERO};
      constexpr std::array<std::complex<qc::fp>, 4> IPX = {C_ZERO, IM, IM,
                                                           C_ZERO};

      auto& u = unitary_matrix;
      auto det_u = determinant(u);
      auto det_pow = std::pow(det_u, static_cast<qc::fp>(-0.25));
      llvm::transform(u, u.begin(), [&](auto&& x) { return x * det_pow; });
      auto global_phase = std::arg(det_u) / 4.;
      auto u_p = magic_basis_transform(u, MagicBasisTransform::OutOf);
      auto m2 = dot(transpose(u_p), u_p);
      auto default_euler_basis = EulerBasis::ZYZ;

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
      std::normal_distribution<qc::fp> dist;
      auto found = false;
      diagonal4x4 d;
      matrix4x4 p;
      for (int i = 0; i < 100; ++i) {
        qc::fp rand_a;
        qc::fp rand_b;
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
        matrix4x4 m2_real;
        llvm::transform(m2, m2_real.begin(), [&](const qfp& val) {
          return rand_a * val.real() + rand_b * val.imag();
        });
        matrix4x4 p_inner = self_adjoint_eigen_lower(m2_real);
        auto d_inner = diagonal(dot(dot(transpose(p_inner), m2), p_inner));
        matrix4x4 diag_d{}; // zero initialization
        diag_d[0 * 4 + 0] = d_inner[0];
        diag_d[1 * 4 + 1] = d_inner[1];
        diag_d[2 * 4 + 2] = d_inner[2];
        diag_d[3 * 4 + 3] = d_inner[3];

        auto compare = dot(dot(p_inner, diag_d), transpose(p_inner));
        found = llvm::all_of_zip(compare, m2, [](auto&& a, auto&& b) {
          return std::abs(a - b) < 1.0e-13;
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
      std::array<qc::fp, d.size()> d_real;
      llvm::transform(d, d_real.begin(),
                      [](auto&& x) { return -std::arg(x) / 2.0; });
      d_real[3] = -d_real[0] - d_real[1] - d_real[2];
      std::array<qc::fp, 3> cs;
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
      auto is_close = [&](qc::fp ap, qc::fp bp, qc::fp cp) -> bool {
        auto da = a - ap;
        auto db = b - bp;
        auto dc = c - cp;
        auto tr = 4. * qfp(std::cos(da) * std::cos(db) * std::cos(dc),
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
          K1r,
          K2l,
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
      };

      TwoQubitWeylDecomposition specialized = get_specialized_decomposition();

      auto get_tr = [&]() {
        if (flipped_from_original) {
          auto [da, db, dc] = std::array{
              qc::PI_2 - a - specialized.a,
              b - specialized.b,
              -c - specialized.c,
          };
          return 4. * qfp(std::cos(da) * std::cos(db) * std::cos(dc),
                          std::sin(da) * std::sin(db) * std::sin(dc));
        } else {
          auto [da, db, dc] = std::array{a - specialized.a, b - specialized.b,
                                         c - specialized.c};
          return 4. * qfp(std::cos(da) * std::cos(db) * std::cos(dc),
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
    } // namespace mqt::ir::opt
  };

  static constexpr auto DEFAULT_FIDELITY = 1.0 - 1.0e-9;

  struct TwoQubitBasisDecomposer {
    std::string gate;
    qc::fp basis_fidelity;
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

    void twoQubitDecompose(const matrix4x4& unitaryMatrix,
                           std::optional<qc::fp> _basis_fidelity,
                           bool approximate,
                           std::optional<std::uint8_t> _num_basis_uses) {
      auto get_basis_fidelity = [&]() {
        if (approximate) {
          return _basis_fidelity.value_or(this->basis_fidelity);
        }
        return 1.0;
      };
      qc::fp basis_fidelity = get_basis_fidelity();
      auto target_decomposed = TwoQubitWeylDecomposition::new_inner(
          unitaryMatrix, DEFAULT_FIDELITY, std::nullopt);
      auto traces = this->traces(target_decomposed);
      auto get_default_nbasis = [&]() {
        auto minValue = std::numeric_limits<qc::fp>::max();
        auto minIndex = -1;
        for(std::size_t i = 0; i < traces.size(); ++i) {
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
          return decomp0_inner(&target_decomposed);
        }
        if (best_nbasis == 1) {
          return decomp1_inner(&target_decomposed);
        }
        if (best_nbasis == 2) {
          return decomp2_supercontrolled_inner(&target_decomposed);
        }
        if (best_nbasis == 3) {
          return decomp3_supercontrolled_inner(&target_decomposed);
        }
        throw std::logic_error{"Invalid basis to use"};
      };
      auto pulse_optimize = this->pulse_optimize.unwrap_or(true);
      auto sequence = if (pulse_optimize) {
        self.pulse_optimal_chooser(best_nbasis, &decomposition,
                                   &target_decomposed)
            ?
      }
      else {None};
      if let
        Some(seq) = sequence { return Ok(seq); }
      auto target_1q_basis_list = EulerBasisSet::new ();
      target_1q_basis_list.add_basis(self.euler_basis);
      let euler_decompositions
          : SmallVec<[Option<OneQubitGateSequence>; 8]> =
                decomposition.iter()
                    .map(| decomp |
                         {unitary_to_gate_sequence_inner(
                             decomp.view(), &target_1q_basis_list, 0, None,
                             true, None, )})
                    .collect();
      auto gates = Vec::with_capacity(TWO_QUBIT_SEQUENCE_DEFAULT_CAPACITY);
      auto global_phase = target_decomposed.global_phase;
      global_phase -= best_nbasis as f64 * self.basis_decomposer.global_phase;
      if best_nbasis
        == 2 { global_phase += PI; }
    for
      i in 0..best_nbasis as usize {
        if let
          Some(euler_decomp) = &euler_decompositions[2 * i] {
            for
              gate in& euler_decomp.gates {
                gates.push((gate.0.into(), gate.1.clone(), smallvec ![0]));
              }
            global_phase += euler_decomp.global_phase
          }
        if let
          Some(euler_decomp) = &euler_decompositions[2 * i + 1] {
            for
              gate in& euler_decomp.gates {
                gates.push((gate.0.into(), gate.1.clone(), smallvec ![1]));
              }
            global_phase += euler_decomp.global_phase
          }
        gates.push(
            (self.gate.clone(), self.gate_params.clone(), smallvec ![ 0, 1 ]));
      }
    if let
      Some(euler_decomp) = &euler_decompositions[2 * best_nbasis as usize] {
        for
          gate in& euler_decomp.gates {
            gates.push((gate.0.into(), gate.1.clone(), smallvec ![0]));
          }
        global_phase += euler_decomp.global_phase
      }
    if let
      Some(euler_decomp) = &euler_decompositions[2 * best_nbasis as usize + 1] {
        for
          gate in& euler_decomp.gates {
            gates.push((gate.0.into(), gate.1.clone(), smallvec ![1]));
          }
        global_phase += euler_decomp.global_phase
      }
    Ok(TwoQubitGateSequence{
        gates,
        global_phase,
    })
    }

    std::array<qfp, 4> traces(TwoQubitWeylDecomposition target) {
      return {
          4. * qfp(std::cos(target.a) * std::cos(target.b) * std::cos(target.c),
                   std::sin(target.a) * std::sin(target.b) *
                       std::sin(target.c), ),
          4. * qfp(std::cos(qc::PI_4 - target.a) *
                       std::cos(self.basis_decomposer.b - target.b) *
                       std::cos(target.c),
                   std::sin(qc::PI_4 - target.a) *
                       std::sin(self.basis_decomposer.b - target.b) *
                       std::sin(target.c)),
          qfp(4. * std::cos(target.c), 0.),
          qfp(4., 0.),
      };
    }
  };
};

/**
 * @brief Populates the given pattern set with patterns for gate
 * decomposition.
 */
void populateGateDecompositionPatterns(mlir::RewritePatternSet& patterns) {
  patterns.add<GateDecompositionPattern>(patterns.getContext());
}

} // namespace mqt::ir::opt

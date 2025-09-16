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
#include <iterator>
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
    //EulerBasis default_euler_basis; // TODO: simply use ZYZ?
    std::optional<qc::fp> requested_fidelity;
    qc::fp calculated_fidelity;
    dd::TwoQubitGateMatrix unitary_matrix;

    TwoQubitWeylDecomposition new_inner(
        std::array<std::complex<qc::fp>, 4> unitary_matrix,

        std::optional<qc::fp> fidelity,
        std::optional<Specialization> _specialization
    ) {
        constexpr std::array<std::complex<qc::fp>, 4> IPZ = [[IM, C_ZERO], [C_ZERO, M_IM]];
        constexpr std::array<std::complex<qc::fp>, 4> IPY  =[[C_ZERO, C_ONE], [C_M_ONE, C_ZERO]];
        constexpr std::array<std::complex<qc::fp>, 4> IPX = [[C_ZERO, IM], [IM, C_ZERO]];


        auto u = unitary_matrix;
        let det_u = u.view().into_faer().determinant();
        let det_pow = det_u.powf(-0.25);
        u.mapv_inplace(|x| x * det_pow);
        let mut global_phase = det_u.arg() / 4.;
        let u_p = magic_basis_transform(u.view(), MagicBasisTransform::OutOf);
        let m2 = u_p.t().dot(&u_p);
        let default_euler_basis = EulerBasis::ZYZ;

        // M2 is a symmetric complex matrix. We need to decompose it as M2 = P D P^T where
        // P ∈ SO(4), D is diagonal with unit-magnitude elements.
        //
        // We can't use raw `eig` directly because it isn't guaranteed to give us real or orthogonal
        // eigenvectors. Instead, since `M2` is complex-symmetric,
        //   M2 = A + iB
        // for real-symmetric `A` and `B`, and as
        //   M2^+ @ M2 = A^2 + B^2 + i [A, B] = 1
        // we must have `A` and `B` commute, and consequently they are simultaneously diagonalizable.
        // Mixing them together _should_ account for any degeneracy problems, but it's not
        // guaranteed, so we repeat it a little bit.  The fixed seed is to make failures
        // deterministic; the value is not important.
        let mut state = Pcg64Mcg::seed_from_u64(2023);
        let mut found = false;
        let mut d: Array1<Complex64> = Array1::zeros(0);
        let mut p: Array2<Complex64> = Array2::zeros((0, 0));
        for i in 0..100 {
            let rand_a: f64;
            let rand_b: f64;
            // For debugging the algorithm use the same RNG values from the
            // previous Python implementation for the first random trial.
            // In most cases this loop only executes a single iteration and
            // using the same rng values rules out possible RNG differences
            // as the root cause of a test failure
            if i == 0 {
                rand_a = 1.2602066112249388;
                rand_b = 0.22317849046722027;
            } else {
                rand_a = state.sample(StandardNormal);
                rand_b = state.sample(StandardNormal);
            }
            let m2_real = m2.mapv(|val| rand_a * val.re + rand_b * val.im);
            let p_inner = m2_real
                .view()
                .into_faer()
                .self_adjoint_eigen(Lower)
                .map_err(|e| QiskitError::new_err(format!("{e:?}")))?
                .U()
                .into_ndarray()
                .mapv(Complex64::from);
            let d_inner = p_inner.t().dot(&m2).dot(&p_inner).diag().to_owned();
            let mut diag_d: Array2<Complex64> = Array2::zeros((4, 4));
            diag_d
                .diag_mut()
                .iter_mut()
                .enumerate()
                .for_each(|(index, x)| *x = d_inner[index]);

            let compare = p_inner.dot(&diag_d).dot(&p_inner.t());
            found = abs_diff_eq!(compare.view(), m2, epsilon = 1.0e-13);
            if found {
                p = p_inner;
                d = d_inner;
                break;
            }
        }
        if !found {
            return Err(QiskitError::new_err(format!(
                "TwoQubitWeylDecomposition: failed to diagonalize M2. Please report this at https://github.com/Qiskit/qiskit-terra/issues/4159. Input: {unitary_matrix:?}"
            )));
        }
        let mut d = -d.map(|x| x.arg() / 2.);
        d[3] = -d[0] - d[1] - d[2];
        let mut cs: SmallVec<[f64; 3]> = (0..3)
            .map(|i| ((d[i] + d[3]) / 2.0).rem_euclid(TWO_PI))
            .collect();
        let cstemp: SmallVec<[f64; 3]> = cs
            .iter()
            .map(|x| x.rem_euclid(PI2))
            .map(|x| x.min(PI2 - x))
            .collect();
        let mut order = arg_sort(&cstemp);
        (order[0], order[1], order[2]) = (order[1], order[2], order[0]);
        (cs[0], cs[1], cs[2]) = (cs[order[0]], cs[order[1]], cs[order[2]]);
        (d[0], d[1], d[2]) = (d[order[0]], d[order[1]], d[order[2]]);
        let mut p_orig = p.clone();
        for (i, item) in order.iter().enumerate().take(3) {
            let slice_a = p.slice_mut(s![.., i]);
            let slice_b = p_orig.slice_mut(s![.., *item]);
            Zip::from(slice_a).and(slice_b).for_each(::std::mem::swap);
        }
        if p.view().into_faer().determinant().re < 0. {
            p.slice_mut(s![.., -1]).mapv_inplace(|x| -x);
        }
        let mut temp: Array2<Complex64> = Array2::zeros((4, 4));
        temp.diag_mut()
            .iter_mut()
            .enumerate()
            .for_each(|(index, x)| *x = (IM * d[index]).exp());
        let k1 = magic_basis_transform(u_p.dot(&p).dot(&temp).view(), MagicBasisTransform::Into);
        let k2 = magic_basis_transform(p.t(), MagicBasisTransform::Into);

        #[allow(non_snake_case)]
        let (mut K1l, mut K1r, phase_l) = decompose_two_qubit_product_gate(k1.view())?;
        #[allow(non_snake_case)]
        let (K2l, mut K2r, phase_r) = decompose_two_qubit_product_gate(k2.view())?;
        global_phase += phase_l + phase_r;

        // Flip into Weyl chamber
        if cs[0] > PI2 {
            cs[0] -= PI32;
            K1l = K1l.dot(&ipy);
            K1r = K1r.dot(&ipy);
            global_phase += PI2;
        }
        if cs[1] > PI2 {
            cs[1] -= PI32;
            K1l = K1l.dot(&ipx);
            K1r = K1r.dot(&ipx);
            global_phase += PI2;
        }
        let mut conjs = 0;
        if cs[0] > PI4 {
            cs[0] = PI2 - cs[0];
            K1l = K1l.dot(&ipy);
            K2r = ipy.dot(&K2r);
            conjs += 1;
            global_phase -= PI2;
        }
        if cs[1] > PI4 {
            cs[1] = PI2 - cs[1];
            K1l = K1l.dot(&ipx);
            K2r = ipx.dot(&K2r);
            conjs += 1;
            global_phase += PI2;
            if conjs == 1 {
                global_phase -= PI;
            }
        }
        if cs[2] > PI2 {
            cs[2] -= PI32;
            K1l = K1l.dot(&ipz);
            K1r = K1r.dot(&ipz);
            global_phase += PI2;
            if conjs == 1 {
                global_phase -= PI;
            }
        }
        if conjs == 1 {
            cs[2] = PI2 - cs[2];
            K1l = K1l.dot(&ipz);
            K2r = ipz.dot(&K2r);
            global_phase += PI2;
        }
        if cs[2] > PI4 {
            cs[2] -= PI2;
            K1l = K1l.dot(&ipz);
            K1r = K1r.dot(&ipz);
            global_phase -= PI2;
        }
        let [a, b, c] = [cs[1], cs[0], cs[2]];
        let is_close = |ap: f64, bp: f64, cp: f64| -> bool {
            let [da, db, dc] = [a - ap, b - bp, c - cp];
            let tr = 4.
                * c64(
                    da.cos() * db.cos() * dc.cos(),
                    da.sin() * db.sin() * dc.sin(),
                );
            match fidelity {
                Some(fid) => tr.trace_to_fid() >= fid,
                // Set to false here to default to general specialization in the absence of a
                // fidelity and provided specialization.
                None => false,
            }
        };

        let closest_abc = closest_partial_swap(a, b, c);
        let closest_ab_minus_c = closest_partial_swap(a, b, -c);
        let mut flipped_from_original = false;
        let specialization = match _specialization {
            Some(specialization) => specialization,
            None => {
                if is_close(0., 0., 0.) {
                    Specialization::IdEquiv
                } else if is_close(PI4, PI4, PI4) || is_close(PI4, PI4, -PI4) {
                    Specialization::SWAPEquiv
                } else if is_close(closest_abc, closest_abc, closest_abc) {
                    Specialization::PartialSWAPEquiv
                } else if is_close(closest_ab_minus_c, closest_ab_minus_c, -closest_ab_minus_c) {
                    Specialization::PartialSWAPFlipEquiv
                } else if is_close(a, 0., 0.) {
                    Specialization::ControlledEquiv
                } else if is_close(PI4, PI4, c) {
                    Specialization::MirrorControlledEquiv
                } else if is_close((a + b) / 2., (a + b) / 2., c) {
                    Specialization::fSimaabEquiv
                } else if is_close(a, (b + c) / 2., (b + c) / 2.) {
                    Specialization::fSimabbEquiv
                } else if is_close(a, (b - c) / 2., (c - b) / 2.) {
                    Specialization::fSimabmbEquiv
                } else {
                    Specialization::General
                }
            }
        };
        let general = TwoQubitWeylDecomposition {
            a,
            b,
            c,
            global_phase,
            K1l,
            K1r,
            K2l,
            K2r,
            specialization: Specialization::General,
            default_euler_basis,
            requested_fidelity: fidelity,
            calculated_fidelity: -1.0,
            unitary_matrix,
        };
        let mut specialized: TwoQubitWeylDecomposition = match specialization {
            // :math:`U \sim U_d(0,0,0) \sim Id`
            //
            // This gate binds 0 parameters, we make it canonical by setting
            // :math:`K2_l = Id` , :math:`K2_r = Id`.
            Specialization::IdEquiv => TwoQubitWeylDecomposition {
                specialization,
                a: 0.,
                b: 0.,
                c: 0.,
                K1l: general.K1l.dot(&general.K2l),
                K1r: general.K1r.dot(&general.K2r),
                K2l: Array2::eye(2),
                K2r: Array2::eye(2),
                ..general
            },
            // :math:`U \sim U_d(\pi/4, \pi/4, \pi/4) \sim U(\pi/4, \pi/4, -\pi/4) \sim \text{SWAP}`
            //
            // This gate binds 0 parameters, we make it canonical by setting
            // :math:`K2_l = Id` , :math:`K2_r = Id`.
            Specialization::SWAPEquiv => {
                if c > 0. {
                    TwoQubitWeylDecomposition {
                        specialization,
                        a: PI4,
                        b: PI4,
                        c: PI4,
                        K1l: general.K1l.dot(&general.K2r),
                        K1r: general.K1r.dot(&general.K2l),
                        K2l: Array2::eye(2),
                        K2r: Array2::eye(2),
                        ..general
                    }
                } else {
                    flipped_from_original = true;
                    TwoQubitWeylDecomposition {
                        specialization,
                        a: PI4,
                        b: PI4,
                        c: PI4,
                        global_phase: global_phase + PI2,
                        K1l: general.K1l.dot(&ipz).dot(&general.K2r),
                        K1r: general.K1r.dot(&ipz).dot(&general.K2l),
                        K2l: Array2::eye(2),
                        K2r: Array2::eye(2),
                        ..general
                    }
                }
            }
            // :math:`U \sim U_d(\alpha\pi/4, \alpha\pi/4, \alpha\pi/4) \sim \text{SWAP}^\alpha`
            //
            // This gate binds 3 parameters, we make it canonical by setting:
            //
            // :math:`K2_l = Id`.
            Specialization::PartialSWAPEquiv => {
                let closest = closest_partial_swap(a, b, c);
                let mut k2l_dag = general.K2l.t().to_owned();
                k2l_dag.view_mut().mapv_inplace(|x| x.conj());
                TwoQubitWeylDecomposition {
                    specialization,
                    a: closest,
                    b: closest,
                    c: closest,
                    K1l: general.K1l.dot(&general.K2l),
                    K1r: general.K1r.dot(&general.K2l),
                    K2r: k2l_dag.dot(&general.K2r),
                    K2l: Array2::eye(2),
                    ..general
                }
            }
            // :math:`U \sim U_d(\alpha\pi/4, \alpha\pi/4, -\alpha\pi/4) \sim \text{SWAP}^\alpha`
            //
            // (a non-equivalent root of SWAP from the TwoQubitWeylPartialSWAPEquiv
            // similar to how :math:`x = (\pm \sqrt(x))^2`)
            //
            // This gate binds 3 parameters, we make it canonical by setting:
            //
            // :math:`K2_l = Id`
            Specialization::PartialSWAPFlipEquiv => {
                let closest = closest_partial_swap(a, b, -c);
                let mut k2l_dag = general.K2l.t().to_owned();
                k2l_dag.mapv_inplace(|x| x.conj());
                TwoQubitWeylDecomposition {
                    specialization,
                    a: closest,
                    b: closest,
                    c: -closest,
                    K1l: general.K1l.dot(&general.K2l),
                    K1r: general.K1r.dot(&ipz).dot(&general.K2l).dot(&ipz),
                    K2r: ipz.dot(&k2l_dag).dot(&ipz).dot(&general.K2r),
                    K2l: Array2::eye(2),
                    ..general
                }
            }
            // :math:`U \sim U_d(\alpha, 0, 0) \sim \text{Ctrl-U}`
            //
            // This gate binds 4 parameters, we make it canonical by setting:
            //
            //      :math:`K2_l = Ry(\theta_l) Rx(\lambda_l)` ,
            //      :math:`K2_r = Ry(\theta_r) Rx(\lambda_r)` .
            Specialization::ControlledEquiv => {
                let euler_basis = EulerBasis::XYX;
                let [k2ltheta, k2lphi, k2llambda, k2lphase] =
                    angles_from_unitary(general.K2l.view(), euler_basis);
                let [k2rtheta, k2rphi, k2rlambda, k2rphase] =
                    angles_from_unitary(general.K2r.view(), euler_basis);
                TwoQubitWeylDecomposition {
                    specialization,
                    a,
                    b: 0.,
                    c: 0.,
                    global_phase: global_phase + k2lphase + k2rphase,
                    K1l: general.K1l.dot(&rx_matrix(k2lphi)),
                    K1r: general.K1r.dot(&rx_matrix(k2rphi)),
                    K2l: ry_matrix(k2ltheta).dot(&rx_matrix(k2llambda)),
                    K2r: ry_matrix(k2rtheta).dot(&rx_matrix(k2rlambda)),
                    default_euler_basis: euler_basis,
                    ..general
                }
            }
            // :math:`U \sim U_d(\pi/4, \pi/4, \alpha) \sim \text{SWAP} \cdot \text{Ctrl-U}`
            //
            // This gate binds 4 parameters, we make it canonical by setting:
            //
            // :math:`K2_l = Ry(\theta_l)\cdot Rz(\lambda_l)` , :math:`K2_r = Ry(\theta_r)\cdot Rz(\lambda_r)`
            Specialization::MirrorControlledEquiv => {
                let [k2ltheta, k2lphi, k2llambda, k2lphase] =
                    angles_from_unitary(general.K2l.view(), EulerBasis::ZYZ);
                let [k2rtheta, k2rphi, k2rlambda, k2rphase] =
                    angles_from_unitary(general.K2r.view(), EulerBasis::ZYZ);
                TwoQubitWeylDecomposition {
                    specialization,
                    a: PI4,
                    b: PI4,
                    c,
                    global_phase: global_phase + k2lphase + k2rphase,
                    K1l: general.K1l.dot(&rz_matrix(k2rphi)),
                    K1r: general.K1r.dot(&rz_matrix(k2lphi)),
                    K2l: ry_matrix(k2ltheta).dot(&rz_matrix(k2llambda)),
                    K2r: ry_matrix(k2rtheta).dot(&rz_matrix(k2rlambda)),
                    ..general
                }
            }
            // :math:`U \sim U_d(\alpha, \alpha, \beta), \alpha \geq |\beta|`
            //
            // This gate binds 5 parameters, we make it canonical by setting:
            //
            // :math:`K2_l = Ry(\theta_l)\cdot Rz(\lambda_l)`.
            Specialization::fSimaabEquiv => {
                let [k2ltheta, k2lphi, k2llambda, k2lphase] =
                    angles_from_unitary(general.K2l.view(), EulerBasis::ZYZ);
                TwoQubitWeylDecomposition {
                    specialization,
                    a: (a + b) / 2.,
                    b: (a + b) / 2.,
                    c,
                    global_phase: global_phase + k2lphase,
                    K1r: general.K1r.dot(&rz_matrix(k2lphi)),
                    K1l: general.K1l.dot(&rz_matrix(k2lphi)),
                    K2l: ry_matrix(k2ltheta).dot(&rz_matrix(k2llambda)),
                    K2r: rz_matrix(-k2lphi).dot(&general.K2r),
                    ..general
                }
            }
            // :math:`U \sim U_d(\alpha, \beta, -\beta), \alpha \geq \beta \geq 0`
            //
            // This gate binds 5 parameters, we make it canonical by setting:
            //
            // :math:`K2_l = Ry(\theta_l)Rx(\lambda_l)`
            Specialization::fSimabbEquiv => {
                let euler_basis = EulerBasis::XYX;
                let [k2ltheta, k2lphi, k2llambda, k2lphase] =
                    angles_from_unitary(general.K2l.view(), euler_basis);
                TwoQubitWeylDecomposition {
                    specialization,
                    a,
                    b: (b + c) / 2.,
                    c: (b + c) / 2.,
                    global_phase: global_phase + k2lphase,
                    K1r: general.K1r.dot(&rx_matrix(k2lphi)),
                    K1l: general.K1l.dot(&rx_matrix(k2lphi)),
                    K2l: ry_matrix(k2ltheta).dot(&rx_matrix(k2llambda)),
                    K2r: rx_matrix(-k2lphi).dot(&general.K2r),
                    default_euler_basis: euler_basis,
                    ..general
                }
            }
            // :math:`U \sim U_d(\alpha, \beta, -\beta), \alpha \geq \beta \geq 0`
            //
            // This gate binds 5 parameters, we make it canonical by setting:
            //
            // :math:`K2_l = Ry(\theta_l)Rx(\lambda_l)`
            Specialization::fSimabmbEquiv => {
                let euler_basis = EulerBasis::XYX;
                let [k2ltheta, k2lphi, k2llambda, k2lphase] =
                    angles_from_unitary(general.K2l.view(), euler_basis);
                TwoQubitWeylDecomposition {
                    specialization,
                    a,
                    b: (b - c) / 2.,
                    c: -((b - c) / 2.),
                    global_phase: global_phase + k2lphase,
                    K1l: general.K1l.dot(&rx_matrix(k2lphi)),
                    K1r: general.K1r.dot(&ipz).dot(&rx_matrix(k2lphi)).dot(&ipz),
                    K2l: ry_matrix(k2ltheta).dot(&rx_matrix(k2llambda)),
                    K2r: ipz.dot(&rx_matrix(-k2lphi)).dot(&ipz).dot(&general.K2r),
                    default_euler_basis: euler_basis,
                    ..general
                }
            }
            // U has no special symmetry.
            //
            // This gate binds all 6 possible parameters, so there is no need to make the single-qubit
            // pre-/post-gates canonical.
            Specialization::General => general,
        };

        let tr = if flipped_from_original {
            let [da, db, dc] = [
                PI2 - a - specialized.a,
                b - specialized.b,
                -c - specialized.c,
            ];
            4. * c64(
                da.cos() * db.cos() * dc.cos(),
                da.sin() * db.sin() * dc.sin(),
            )
        } else {
            let [da, db, dc] = [a - specialized.a, b - specialized.b, c - specialized.c];
            4. * c64(
                da.cos() * db.cos() * dc.cos(),
                da.sin() * db.sin() * dc.sin(),
            )
        };
        specialized.calculated_fidelity = tr.trace_to_fid();
        if let Some(fid) = specialized.requested_fidelity {
            if specialized.calculated_fidelity + 1.0e-13 < fid {
                return Err(QiskitError::new_err(format!(
                    "Specialization: {:?} calculated fidelity: {} is worse than requested fidelity: {}",
                    specialized.specialization,
                    specialized.calculated_fidelity,
                    fid
                )));
            }
        }
        specialized.global_phase += tr.arg();
        Ok(specialized)
  };

  void twoQubitDecompose(dd::TwoQubitGateMatrix unitaryMatrix) {
    qc::fp basis_fidelity = 1.0;
    let target_decomposed = TwoQubitWeylDecomposition::new_inner(
        unitary, Some(DEFAULT_FIDELITY), None)
        ? ;
    let traces = self.traces(&target_decomposed);
    let best_nbasis = _num_basis_uses.unwrap_or_else(
        ||
        {traces.into_iter()
             .enumerate()
             .map(| (idx, trace) |
                  (idx, trace.trace_to_fid() * basis_fidelity.powi(idx as i32)))
             .min_by(| (_idx1, fid1),
                     (_idx2, fid2) | fid2.partial_cmp(fid1).unwrap())
             .unwrap() .0 as u8});
    let decomposition = match best_nbasis{
        0 = > decomp0_inner(&target_decomposed),
        1 = > self.decomp1_inner(&target_decomposed),
        2 = > self.decomp2_supercontrolled_inner(&target_decomposed),
        3 = > self.decomp3_supercontrolled_inner(&target_decomposed),
        _ = > unreachable !("Invalid basis to use"),
    };
    let pulse_optimize = self.pulse_optimize.unwrap_or(true);
    let sequence = if pulse_optimize {
      self.pulse_optimal_chooser(best_nbasis, &decomposition,
                                 &target_decomposed)
          ?
    }
    else {None};
    if let
      Some(seq) = sequence { return Ok(seq); }
    let mut target_1q_basis_list = EulerBasisSet::new ();
    target_1q_basis_list.add_basis(self.euler_basis);
    let euler_decompositions
        : SmallVec<[Option<OneQubitGateSequence>; 8]> =
              decomposition.iter()
                  .map(| decomp |
                       {unitary_to_gate_sequence_inner(decomp.view(),
                                                       &target_1q_basis_list, 0,
                                                       None, true, None, )})
                  .collect();
    let mut gates = Vec::with_capacity(TWO_QUBIT_SEQUENCE_DEFAULT_CAPACITY);
    let mut global_phase = target_decomposed.global_phase;
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

  std::array<std::complex<qc::fp>, 4> traces(TwoQubitWeylDecomposition target) {
    return {
        4. * std::complex<qc::fp>(
                 target.a.cos() * target.b.cos() * target.c.cos(),
                 target.a.sin() * target.b.sin() * target.c.sin(), ),
        4. *
            c64((PI4 - target.a).cos() *
                    (self.basis_decomposer.b - target.b).cos() * target.c.cos(),
                (PI4 - target.a).sin() *
                    (self.basis_decomposer.b - target.b).sin() *
                    target.c.sin(), ),
        c64(4. * target.c.cos(), 0.),
        c64(4., 0.),
    };
  }
};

/**
 * @brief Populates the given pattern set with patterns for gate
 * decomposition.
 */
void populateGateDecompositionPatterns(mlir::RewritePatternSet& patterns) {
  patterns.add<GateDecompositionPattern>(patterns.getContext());
}

} // namespace mqt::ir::opt

#pragma once

#include "Helpers.h"

namespace mqt::ir::opt {
// return Q, R such that A = Q * R
static void qrDecomposition(const rmatrix4x4& A, rmatrix4x4& Q, rmatrix4x4& R) {
  // array of factor Q1, Q2, ... Qm
  std::vector<rmatrix4x4> qv(4);

  // temp array
  auto z(A);
  rmatrix4x4 z1;

  auto vmadd = [](const auto& a, const auto& b, double s, auto& c) {
    for (int i = 0; i < 4; i++)
      c[i] = a[i] + s * b[i];
  };

  auto compute_householder_factor = [](rmatrix4x4& mat, const rdiagonal4x4& v) {
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++)
        mat[i + 4 * j] = -2.0 * v[i] * v[j];
    for (int i = 0; i < 4; i++)
      mat[i * 4 + i] += 1;
  };

  // take c-th column of a matrix, put results in Vector v
  auto extract_column = [](const rmatrix4x4& m, rdiagonal4x4& v, int c) {
    for (int i = 0; i < 4; i++)
      v[i] = m[i + 4 * c];
  };

  auto compute_minor = [](rmatrix4x4& lhs, const rmatrix4x4& rhs, int d) {
    for (int i = 0; i < d; i++)
      lhs[i * 4 + i] = 1.0;
    for (int i = d; i < 4; i++)
      for (int j = d; j < 4; j++)
        lhs[i + 4 * j] = rhs[i + 4 * j];
  };

  auto norm = [](auto&& m) {
    double sum = 0;
    for (int i = 0; i < m.size(); i++)
      sum += m[i] * m[i];
    return sqrt(sum);
  };

  auto rescale_unit = [&](auto& m) {
    auto factor = norm(m);
    for (int i = 0; i < m.size(); i++)
      m[i] /= factor;
  };

  for (int k = 0; k < 4 && k < 4 - 1; k++) {

    rdiagonal4x4 e{}, x{};
    double a{};

    // compute minor
    compute_minor(z1, z, k);

    // extract k-th column into x
    extract_column(z1, x, k);

    a = norm(x);
    if (A[k * 4 + k] > 0)
      a = -a;

    for (int i = 0; i < 4; i++)
      e[i] = (i == k) ? 1 : 0;

    // e = x + a*e
    vmadd(x, e, a, e);

    // e = e / ||e||
    rescale_unit(e);

    // qv[k] = I - 2 *e*e^T
    compute_householder_factor(qv[k], e);

    // z = qv[k] * z1
    z = helpers::multiply(qv[k], z1);
  }

  Q = qv[0];

  // after this loop, we will obtain Q (up to a transpose operation)
  for (int i = 1; i < 4 && i < 4 - 1; i++) {

    z1 = helpers::multiply(qv[i], Q);
    Q = z1;
  }

  R = helpers::multiply(Q, A);
  Q = helpers::transpose(Q);
}

// Function to perform self-adjoint eigenvalue decomposition (4x4 matrix)
static rmatrix4x4                // eigenvectors (4x4)
self_adjoint_evd(rmatrix4x4 A,   // input symmetric matrix (4x4)
                 rdiagonal4x4& s // eigenvalues
) {
  rmatrix4x4 U = {1, 0, 0, 0, 0, 1, 0, 0,
                  0, 0, 1, 0, 0, 0, 0, 1}; // Start with identity

  auto isConverged = [](const rmatrix4x4& A, double tol = 1e-10) -> bool {
    double sum = 0.0;
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        if (i != j)
          sum += A[i + 4 * j] * A[i + 4 * j];
      }
    }
    return std::sqrt(sum) < tol;
  };

  rmatrix4x4 Q{};
  rmatrix4x4 R{};

  constexpr auto maxIters = 100;
  for (int iter = 0; iter < maxIters; ++iter) {
    qrDecomposition(A, Q, R);

    // A = R * Q
    A = helpers::multiply(R, Q);

    // eigenVectors = eigenVectors * Q
    U = helpers::multiply(U, Q);

    if (isConverged(A)) {
      break;
    }
  }

  for (int i = 0; i < 4; ++i) {
    s[i] = A[i * 4 + i];
  }
  return U;
}

} // namespace mqt::ir::opt

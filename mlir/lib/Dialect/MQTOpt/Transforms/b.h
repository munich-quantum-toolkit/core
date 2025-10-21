#pragma once

#include "Helpers.h"

namespace mqt::ir::opt {
void tridiagonalization_inplace(rmatrix4x4& mat, rdiagonal4x4& hCoeffs) {
  auto n = 4;

  auto makeHouseholder = [](llvm::SmallVector<fp, 4>& essential, fp& tau,
                            fp& beta) {
    std::vector<fp> tail{essential.begin(), essential.end()};

    auto squaredNorm = [](auto&& v) {
      qfp sum{};
      for (auto&& x : v) {
        sum += qfp(std::real(x) * std::real(x), std::imag(x) * std::imag(x));
      }
      return sum.real() + sum.imag();
    };

    auto tailSqNorm = essential.size() == 1 ? 0.0 : squaredNorm(tail);
    fp c0 = essential[0];
    const fp tol = (std::numeric_limits<fp>::min)();

    if (tailSqNorm <= tol && std::norm(std::imag(c0)) <= tol) {
      tau = 0;
      beta = std::real(c0);
      llvm::fill(essential, 0);
    } else {
      beta = std::sqrt(std::norm(c0) + tailSqNorm);
      if (std::real(c0) >= 0.0) {
        beta = -beta;
      }
      for (std::size_t i = 0; i < essential.size(); ++i) {
        essential[i] = tail[i] / (c0 - beta);
      }
      tau = helpers::conj((beta - c0) / beta);
    }
  };

  auto lowerSelfadjointView = [](auto matrix) {
    const int n = std::sqrt(matrix.size());
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j <= i; ++j) {
        matrix[j * n + i] = matrix[i * n + j];
      }
    }
    return matrix;
  };

  auto bottomRightCorner = [](auto&& matrix, int rows, int columns) {
    const int n = std::sqrt(matrix.size());
    return helpers::submatrix(matrix, n - rows, n - columns, rows, columns);
  };

  auto getColumn = [](const rmatrix4x4& matrix, int column) {
    rdiagonal4x4 result;
    for (int j = 0; j < 4; ++j) {
      result[j] = matrix[j * 4 + column];
    }
    return result;
  };

  auto getTail = [](auto&& array, int size) {
    std::vector<fp> result(size);
    for (int i = 0; i < size; ++i) {
      result[i] = array[array.size() - size + i];
    }
    return result;
  };

  auto rankUpdate = [](auto matrix, auto&& u, auto&& v, auto&& alpha) {
    rmatrix4x4 add1{};
    for (int i = 0; i < u.size(); ++i) {
      for (int j = 0; j < v.size(); ++j) {
        add1[j * 4 + i] += alpha * u[i] * helpers::conj(v[j]);
      }
    }
    rmatrix4x4 add2{};
    for (int i = 0; i < v.size(); ++i) {
      for (int j = 0; j < u.size(); ++j) {
        add2[j * 4 + i] = helpers::conj(alpha) * v[i] * helpers::conj(u[j]);
      }
    }
    for (int i = 0; i < matrix.size(); ++i) {
      matrix[i] += add1[i] + add2[i];
    }
    return matrix;
  };

  for (auto i = 0; i < n - 1; ++i) {
    auto remainingSize = n - i - 1;
    fp beta;
    fp h;

    // matA.col(i).tail(remainingSize).makeHouseholderInPlace(h, beta);
    llvm::SmallVector<fp, 4> tmp;
    for (int j = n - remainingSize; j < n; ++j) {
      tmp.push_back(mat[j * n + i]);
    }
    makeHouseholder(tmp, h, beta);

    // Apply similarity transformation to remaining columns,
    // i.e., A = H A H' where H = I - h v v' and v = matA.col(i).tail(n-i-1)
    // matA.col(i).coeffRef(i + 1) = fp(1);
    mat[(i + 1) * n + i] = 1.0;

    // hCoeffs.tail(n - i - 1).noalias() =
    //     (matA.bottomRightCorner(remainingSize, remainingSize).template
    //     selfadjointView<Lower>() *
    //      (conj(h) * matA.col(i).tail(remainingSize)));
    auto tmp2 = helpers::multiply(
        lowerSelfadjointView(
            bottomRightCorner(mat, remainingSize, remainingSize)),
        helpers::multiply(helpers::conj(h),
                          getTail(getColumn(mat, i), remainingSize)),
        remainingSize);
    for (int a = 0; a < remainingSize; ++a) {
      hCoeffs[i + 1 + a] = tmp2[a];
    }

    // hCoeffs.tail(n - i - 1) +=
    //     (conj(h) * RealScalar(-0.5) *
    //     (hCoeffs.tail(remainingSize).dot(matA.col(i).tail(remainingSize))))
    //     * matA.col(i).tail(n - i - 1);
    auto tmpFactor =
        helpers::conj(h) * static_cast<fp>(-0.5) *
        helpers::vectorsDot(getTail(hCoeffs, remainingSize),
                            getTail(getColumn(mat, i), remainingSize));
    tmp2 = helpers::multiply(tmpFactor,
                             helpers::submatrix(mat, i + 1, i, n - i - 1, 1));
    for (int a = 0; a < remainingSize; ++a) {
      hCoeffs[i + 1 + a] += tmp2[a];
    }

    // matA.bottomRightCorner(remainingSize, remainingSize)
    //     .template selfadjointView<Lower>()
    //     .rankUpdate(matA.col(i).tail(remainingSize),
    //     hCoeffs.tail(remainingSize), Scalar(-1));
    auto updatedMatrix = bottomRightCorner(mat, remainingSize, remainingSize);
    updatedMatrix = lowerSelfadjointView(updatedMatrix);
    updatedMatrix = rankUpdate(
        updatedMatrix, helpers::submatrix(mat, n - remainingSize, i, remainingSize, 1),
        std::vector<fp>{hCoeffs.begin() + (n - remainingSize), hCoeffs.end()},
        -1.0);
    // update bottom right corner
    helpers::assignSubmatrix(mat, updatedMatrix, n - remainingSize,
                             n - remainingSize, remainingSize, remainingSize);

    // matA.col(i).coeffRef(i + 1) = beta;
    mat[(i + 1) * n + i] = beta;
    // hCoeffs.coeffRef(i) = h;
    hCoeffs[i] = h;
  }
}

void tridiagonal_qr_step(rdiagonal4x4& diag, std::vector<fp>& subdiag,
                         int start, int end, rmatrix4x4& matrixQ) {
  // Wilkinson Shift.
  auto td = (diag[end - 1] - diag[end]) * static_cast<fp>(0.5);
  auto e = subdiag[end - 1];
  // Note that thanks to scaling, e^2 or td^2 cannot overflow, however they can
  // still underflow thus leading to inf/NaN values when using the following
  // commented code:
  //   RealScalar e2 = numext::abs2(subdiag[end-1]);
  //   RealScalar mu = diag[end] - e2 / (td + (td>0 ? 1 : -1) * sqrt(td*td +
  //   e2));
  // This explain the following, somewhat more complicated, version:
  auto mu = diag[end];
  if (td == 0.0) {
    mu -= std::abs(e);
  } else if (e != 0.0) {
    const auto e2 = std::norm(e);
    const auto h = std::hypot(td, e);
    if (e2 == 0.0) {
      mu -= e / ((td + (td > static_cast<fp>(0) ? h : -h)) / e);
    } else {
      mu -= e2 / (td + (td > static_cast<fp>(0) ? h : -h));
    }
  }

  auto x = diag[start] - mu;
  auto z = subdiag[start];
  // If z ever becomes zero, the Givens rotation will be the identity and
  // z will stay zero for all future iterations.
  for (int k = start; k < end && z != 0.0; ++k) {
    struct JacobiRotation {
      fp c;
      fp s;

      void makeGivens(fp p, fp q) {
        if (q == 0.0) {
          c = p < 0 ? -1 : 1;
          s = 0;
        } else if (p == 0.0) {
          c = 0;
          s = q < 0 ? 1 : -1;
        } else if (std::abs(p) > std::abs(q)) {
          auto t = q / p;
          auto u = std::sqrt(1.0 + std::norm(t));
          if (p < 0.0)
            u = -u;
          c = 1.0 / u;
          s = -t * c;
        } else {
          auto t = p / q;
          auto u = std::sqrt(1.0 + std::norm(t));
          if (q < 0)
            u = -u;
          s = -1.0 / u;
          c = -t * s;
        }
      }

      void applyOnTheRight(rmatrix4x4& matrix, int p, int q) {
        const int n = std::sqrt(matrix.size());
        auto x = helpers::submatrix(matrix, 0, p, n, 1);
        auto y = helpers::submatrix(matrix, 0, q, n, 1);
        auto j = *this;
        j.transpose();

        if (j.c == 0.0 && j.s == 0.0) {
          return;
        }

        for (int i = 0; i < n; ++i) {
          auto xi = x[i];
          auto yi = y[i];
          x[i] = j.c * xi + helpers::conj(j.s) * yi;
          y[i] = -j.s * xi + helpers::conj(j.c) * yi;
        }

        helpers::assignSubmatrix(matrix, x, 0, p, n, 1);
        helpers::assignSubmatrix(matrix, y, 0, q, n, 1);
      }

      void transpose() { s = -helpers::conj(s); }
    } rot;
    rot.makeGivens(x, z);

    // do T = G' T G
    auto sdk = rot.s * diag[k] + rot.c * subdiag[k];
    auto dkp1 = rot.s * subdiag[k] + rot.c * diag[k + 1];

    diag[k] = rot.c * (rot.c * diag[k] - rot.s * subdiag[k]) -
              rot.s * (rot.c * subdiag[k] - rot.s * diag[k + 1]);
    diag[k + 1] = rot.s * sdk + rot.c * dkp1;
    subdiag[k] = rot.c * sdk - rot.s * dkp1;

    if (k > start)
      subdiag[k - 1] = rot.c * subdiag[k - 1] - rot.s * z;

    // "Chasing the bulge" to return to triangular form.
    x = subdiag[k];
    if (k < end - 1) {
      z = -rot.s * subdiag[k + 1];
      subdiag[k + 1] = rot.c * subdiag[k + 1];
    }

    // apply the givens rotation to the unit matrix Q = Q * G
    rot.applyOnTheRight(matrixQ, k, k + 1);
  }
}

void computeFromTridiagonal_impl(rdiagonal4x4& diag, std::vector<fp>& subdiag,
                                 const int maxIterations,
                                 bool computeEigenvectors, rmatrix4x4& eivec) {
  auto n = diag.size();
  auto end = n - 1;
  int start = 0;
  int iter = 0; // total number of iterations

  constexpr auto considerAsZero = (std::numeric_limits<fp>::min)();
  const auto precision_inv =
      static_cast<fp>(1.0) / std::numeric_limits<fp>::epsilon();
  while (end > 0) {
    for (int i = start; i < end; ++i) {
      if (std::abs(subdiag[i]) < considerAsZero) {
        subdiag[i] = static_cast<fp>(0);
      } else {
        // abs(subdiag[i]) <= epsilon * sqrt(abs(diag[i]) + abs(diag[i+1]))
        // Scaled to prevent underflows.
        const auto scaled_subdiag = precision_inv * subdiag[i];
        if (scaled_subdiag * scaled_subdiag <=
            (std::abs(diag[i]) + std::abs(diag[i + 1]))) {
          subdiag[i] = static_cast<fp>(0);
        }
      }
    }

    // find the largest unreduced block at the end of the matrix.
    while (end > 0 && subdiag[end - 1] == 0.0) {
      end--;
    }
    if (end <= 0) {
      break;
    }

    // if we spent too many iterations, we give up
    iter++;
    if (iter > maxIterations * n) {
      break;
    }

    start = end - 1;
    while (start > 0 && subdiag[start - 1] != 0.0) {
      start--;
    }

    tridiagonal_qr_step(diag, subdiag, start, end, eivec);
  }
  // Sort eigenvalues and corresponding vectors.
  // TODO make the sort optional ?
  // TODO use a better sort algorithm !!
  if (iter > maxIterations * n) {
    throw std::runtime_error{"No convergence for eigenvalue decomposition!"};
  }
  for (int i = 0; i < n - 1; ++i) {
    // diag.segment(i, n - i).minCoeff(&k);
    int k = std::distance(diag.begin() + i,
                          std::min_element(diag.begin() + i, diag.end()));
    if (k > 0 && k < n - i) {
      std::swap(diag[i], diag[k + i]);
      if (computeEigenvectors) {
        for (int j = 0; j < n; ++j) {
          std::swap(eivec[j * n + i], eivec[j * n + (k + i)]);
        }
      }
    }
  }
}

void householderSequenceEval(rmatrix4x4& m_vectors,
                             const rdiagonal4x4& m_coeffs, int length,
                             int shift) {
  auto essentialVector = [&](const rmatrix4x4& vectors, int k) {
    int start = k + 1 + shift;
    std::vector<fp> result;
    result.reserve(4 - start);
    for (int j = start; j < 4; ++j) {
      result.push_back(vectors[j * 4 + k]);
    }
    return result;
  };

  auto applyThisOnTheLeft = [&](auto&& vectors, auto& dst) {
    const auto n = std::sqrt(dst.size());
    for (int k = 0; k < length; ++k) {
      int actual_k = n - k - 1;
      int dstRows = n - shift - actual_k;

      // TODO
      // auto sub_dst = dst.bottomRows(dstRows);
      // sub_dst.applyHouseholderOnTheLeft(essentialVector(vectors, actual_k),
      //                                   m_coeffs[actual_k]);
    }
  };

  auto applyHouseholderOnTheLeft = [](std::vector<fp>& matrix,
                                      const std::vector<fp>& essential,
                                      const fp& tau) {
    const int n = std::sqrt(matrix.size());
    if (tau != 0.0) {
      auto firstRow = helpers::submatrix(matrix, 0, 0, 1, n);
      auto bottom = helpers::submatrix(matrix, 1, 0, n - 1, n);

      auto tmp = helpers::multiply(helpers::transpose_conjugate(essential),
                                   bottom, essential.size());
      tmp = helpers::add(tmp, firstRow);

      auto tmp2 = helpers::add(
          firstRow, helpers::multiply(-1.0, helpers::multiply(tau, tmp)));
      // insert first row (first 4 elements)
      llvm::copy(tmp2, matrix.begin());

      tmp2 = helpers::add(
          bottom, helpers::multiply(-tau, helpers::multiply(essential, tmp,
                                                            essential.size())));
      // insert all rows except first row
      llvm::copy(tmp2, matrix.begin() + n);
    }
  };

  auto applyHouseholderOnTheRight = [](std::vector<fp>& matrix,
                                       const std::vector<fp>& essential,
                                       const fp& tau) {
    const int n = std::sqrt(matrix.size());
    if (tau != 0.0) {
      auto firstColumn = helpers::submatrix(matrix, 0, 0, n, 1);
      auto right = helpers::submatrix(matrix, 0, 1, n, n - 1);

      auto tmp = helpers::multiply(right, essential, n - 1);
      tmp = helpers::add(tmp, firstColumn);

      auto tmp2 = helpers::add(firstColumn, helpers::multiply(-tau, tmp));
      auto tmp3 = helpers::add(
          right,
          helpers::multiply(
              -tau, helpers::multiply(
                        tmp, helpers::transpose_conjugate(essential), 1)));
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
          if (i == 0) {
            // insert first column (first 4 elements)
            matrix[j * n + i] = tmp2[j];
          } else {
            // insert all except first column
            matrix[j * n + i] = tmp3[j * (n - 1) + i];
          }
        }
      }
    }
  };

  rmatrix4x4 dst;
  const int n = std::sqrt(m_vectors.size());
  const int vecs = length;
  dst = helpers::kroneckerProduct(std::array{1.0, 0.0, 0.0, 1.0},
                                  {1.0, 0.0, 0.0, 1.0});
  for (int k = vecs - 1; k >= 0; --k) {
    int cornerSize = n - k - shift;
    auto bottomRightCorner = helpers::submatrix(
        dst, n - cornerSize, n - cornerSize, cornerSize, cornerSize);
    applyHouseholderOnTheLeft(bottomRightCorner, essentialVector(m_vectors, k),
                              m_coeffs[k]);
    // update bottom right corner
    helpers::assignSubmatrix(dst, bottomRightCorner, n - cornerSize,
                             n - cornerSize, cornerSize, cornerSize);
  }
}

// https://docs.rs/faer/latest/faer/mat/generic/struct.Mat.html#method.self_adjoint_eigen
static std::pair<rmatrix4x4, rdiagonal4x4> self_adjoint_evd(rmatrix4x4 matrix) {

  auto lowerTriangularView = [](auto matrix) {
    const int n = std::sqrt(matrix.size());
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j <= i; ++j) {
        matrix[j * n + i] = 0.0;
      }
    }
    return matrix;
  };

  rdiagonal4x4 eigenvalues{1};
  rmatrix4x4 eigenvectors;
  int n = 4;

  // map the matrix coefficients to [-1:1] to avoid over- and underflow.
  eigenvectors = lowerTriangularView(matrix);
  fp scale = *llvm::max_element(eigenvalues, [](auto&& a, auto&& b) {
    return std::abs(a) < std::abs(b);
  });
  if (scale == 0.0)
    scale = static_cast<fp>(1.0);
  for (auto& eigenvector : eigenvectors) {
    eigenvector /= scale;
  }
  rdiagonal4x4 hCoeffs;
  tridiagonalization_inplace(eigenvectors, hCoeffs);
  eigenvalues = helpers::diagonal(eigenvectors);
  auto subdiag = helpers::diagonal<-1>(eigenvectors);
  householderSequenceEval(eigenvectors, helpers::conjugate(hCoeffs), n - 1, 1);

  computeFromTridiagonal_impl(eigenvalues, subdiag, 1000, true, eigenvectors);

  // scale back the eigen values
  for (auto& eigenvalue : eigenvalues) {
    eigenvalue *= scale;
  }

  return {eigenvectors, eigenvalues};
}
} // namespace mqt::ir::opt

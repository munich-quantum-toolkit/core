/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/Utils/Eigensolver.h"

#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Support/LLVM.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <tuple>
#include <utility>

namespace mlir::qco {
// Adapted from John Burkardt's MIT-licensed EISPACK C port (`tred2` / `tql2`):
// https://people.sc.fsu.edu/~jburkardt/c_src/eispack/eispack.c
// Original Fortran: https://netlib.org/eispack/tred2.f,
// https://netlib.org/eispack/tql2.f
// Specialized to `n = 4`; input is row-major, accumulator `z` is column-major.

/// EISPACK `tred2` for `n = 4` (column-major `z[row + col*n]`).
static void symmetricTred24(const std::array<double, 16>& input,
                            std::array<double, 16>& z,
                            std::array<double, 4>& diag,
                            std::array<double, 4>& subdiag) {
  constexpr std::size_t n = 4;
  const auto zAt = [&z](const std::size_t row,
                        const std::size_t col) -> double& {
    return z[row + (col * n)];
  };
  double h = 0.0;

  for (std::size_t col = 0; col < n; ++col) {
    for (std::size_t row = col; row < n; ++row) {
      zAt(row, col) = input[(row * n) + col];
    }
    diag[col] = input[((n - 1) * n) + col];
  }

  for (int i = static_cast<int>(n) - 1; i >= 1; --i) {
    const auto ui = static_cast<std::size_t>(i);
    const std::size_t l = ui - 1;
    h = 0.0;
    double scale = 0.0;
    for (std::size_t k = 0; k <= l; ++k) {
      scale += std::abs(diag[k]);
    }
    if (scale == 0.0) {
      subdiag[ui] = diag[l];
      for (std::size_t j = 0; j <= l; ++j) {
        diag[j] = zAt(l, j);
        zAt(ui, j) = 0.0;
        zAt(j, ui) = 0.0;
      }
      diag[ui] = 0.0;
      continue;
    }
    for (std::size_t k = 0; k <= l; ++k) {
      diag[k] /= scale;
    }
    for (std::size_t k = 0; k <= l; ++k) {
      h += diag[k] * diag[k];
    }
    const double f = diag[l];
    const double g = -std::sqrt(h) * std::copysign(1.0, f);
    subdiag[ui] = scale * g;
    h -= f * g;
    diag[l] = f - g;

    for (std::size_t k = 0; k <= l; ++k) {
      subdiag[k] = 0.0;
    }
    for (std::size_t j = 0; j <= l; ++j) {
      const double fj = diag[j];
      zAt(j, ui) = fj;
      double gj = subdiag[j] + (zAt(j, j) * fj);
      for (std::size_t k = j + 1; k <= l; ++k) {
        gj += zAt(k, j) * diag[k];
        subdiag[k] += zAt(k, j) * fj;
      }
      subdiag[j] = gj;
    }
    double ff = 0.0;
    for (std::size_t k = 0; k <= l; ++k) {
      subdiag[k] /= h;
      ff += subdiag[k] * diag[k];
    }
    const double hh = 0.5 * ff / h;
    for (std::size_t k = 0; k <= l; ++k) {
      subdiag[k] -= hh * diag[k];
    }
    for (std::size_t j = 0; j <= l; ++j) {
      const double fj = diag[j];
      const double gj = subdiag[j];
      for (std::size_t k = j; k <= l; ++k) {
        zAt(k, j) -= (fj * subdiag[k]) + (gj * diag[k]);
      }
      diag[j] = zAt(l, j);
      zAt(ui, j) = 0.0;
    }
    diag[ui] = h;
  }

  for (std::size_t i = 1; i < n; ++i) {
    const std::size_t l = i - 1;
    zAt(n - 1, l) = zAt(l, l);
    zAt(l, l) = 1.0;
    h = diag[i];
    if (h != 0.0) {
      for (std::size_t k = 0; k <= l; ++k) {
        diag[k] = zAt(k, i) / h;
      }
      for (std::size_t j = 0; j <= l; ++j) {
        double g = 0.0;
        for (std::size_t k = 0; k <= l; ++k) {
          g += zAt(k, i) * zAt(k, j);
        }
        for (std::size_t k = 0; k <= l; ++k) {
          zAt(k, j) -= g * diag[k];
        }
      }
    }
    for (std::size_t k = 0; k <= l; ++k) {
      zAt(k, i) = 0.0;
    }
  }

  for (std::size_t j = 0; j < n; ++j) {
    diag[j] = zAt(n - 1, j);
  }
  for (std::size_t j = 0; j < n - 1; ++j) {
    zAt(n - 1, j) = 0.0;
  }
  zAt(n - 1, n - 1) = 1.0;
  subdiag[0] = 0.0;
}

/// EISPACK `tql2` for `n = 4` (column-major `z[row + col*n]`).
static void symmetricTql24(std::array<double, 4>& diag,
                           std::array<double, 4>& subdiag,
                           std::array<double, 16>& z) {
  constexpr std::size_t n = 4;
  const auto zAt = [&z](const std::size_t row,
                        const std::size_t col) -> double& {
    return z[row + (col * n)];
  };

  for (std::size_t i = 1; i < n; ++i) {
    subdiag[i - 1] = subdiag[i];
  }
  double f = 0.0;
  double tst1 = 0.0;
  subdiag[n - 1] = 0.0;

  for (std::size_t l = 0; l < n; ++l) {
    int j = 0;
    const double h = std::abs(diag[l]) + std::abs(subdiag[l]);
    tst1 = std::max(tst1, h);

    std::size_t m = l;
    for (; m < n; ++m) {
      const double tst2 = tst1 + std::abs(subdiag[m]);
      if (tst2 == tst1) {
        break;
      }
    }

    if (m != l) {
      while (true) {
        if (j == 30) {
          llvm::report_fatal_error("symmetricTql2_4: failed to converge");
        }
        ++j;

        const std::size_t l1 = l + 1;
        const std::size_t l2 = l1 + 1;
        const double g = diag[l];
        const double p = (diag[l1] - g) / (2.0 * subdiag[l]);
        const double r = std::hypot(p, 1.0);
        diag[l] = subdiag[l] / (p + std::copysign(std::abs(r), p));
        diag[l1] = subdiag[l] * (p + std::copysign(std::abs(r), p));
        const double dl1 = diag[l1];
        const double hh = g - diag[l];
        for (std::size_t i = l2; i < n; ++i) {
          diag[i] -= hh;
        }
        f += hh;

        double pv = diag[m];
        double c = 1.0;
        double c2 = c;
        const double el1 = subdiag[l1];
        double s = 0.0;
        double c3 = 1.0;
        double s2 = 0.0;
        const std::size_t mml = m - l;
        for (std::size_t ii = 1; ii <= mml; ++ii) {
          c3 = c2;
          c2 = c;
          s2 = s;
          const std::size_t i = m - ii;
          const double gi = c * subdiag[i];
          const double hi = c * pv;
          const double ri = std::hypot(pv, subdiag[i]);
          subdiag[i + 1] = s * ri;
          s = subdiag[i] / ri;
          c = pv / ri;
          pv = (c * diag[i]) - (s * gi);
          diag[i + 1] = hi + (s * ((c * gi) + (s * diag[i])));
          for (std::size_t k = 0; k < n; ++k) {
            const double zkI1 = zAt(k, i + 1);
            zAt(k, i + 1) = (s * zAt(k, i)) + (c * zkI1);
            zAt(k, i) = (c * zAt(k, i)) - (s * zkI1);
          }
        }
        pv = -s * s2 * c3 * el1 * subdiag[l] / dl1;
        subdiag[l] = s * pv;
        diag[l] = c * pv;
        const double tst2 = tst1 + std::abs(subdiag[l]);
        if (tst2 > tst1) {
          continue;
        }
        break;
      }
    }
    diag[l] += f;
  }

  for (std::size_t ii = 1; ii < n; ++ii) {
    const std::size_t i = ii - 1;
    std::size_t k = i;
    double p = diag[i];
    for (std::size_t j = ii; j < n; ++j) {
      if (diag[j] < p) {
        k = j;
        p = diag[j];
      }
    }
    if (k == i) {
      continue;
    }
    diag[k] = diag[i];
    diag[i] = p;
    for (std::size_t j = 0; j < n; ++j) {
      const double tmp = zAt(j, i);
      zAt(j, i) = zAt(j, k);
      zAt(j, k) = tmp;
    }
  }
}

[[nodiscard]] SymmetricEigen4
decomposeSymmetricEigen4(const std::array<double, 16>& symmetric) {
  constexpr std::size_t n = 4;

  SymmetricEigen4 result;
  std::array<double, 16> z{};
  std::array<double, 4> subdiag{};
  symmetricTred24(symmetric, z, result.eigenvalues, subdiag);
  symmetricTql24(result.eigenvalues, subdiag, z);

  for (std::size_t col = 0; col < n; ++col) {
    for (std::size_t row = 0; row < n; ++row) {
      result.eigenvectors(row, col) = z[row + (col * n)];
    }
  }
  return result;
}

[[nodiscard]] SymmetricEigen4
decomposeSymmetricEigen4(const Matrix4x4& matrix) {
  return decomposeSymmetricEigen4(matrix.realPart());
}

[[nodiscard]] static bool isFiniteComplex(const Complex& value) {
  return std::isfinite(value.real()) && std::isfinite(value.imag());
}

static void normalizeInPlace(const llvm::MutableArrayRef<Complex> values) {
  double sumSq = 0.0;
  for (const Complex& value : values) {
    sumSq += std::norm(value);
  }
  const double norm = std::sqrt(sumSq);
  if (norm <= MATRIX_TOLERANCE) {
    return;
  }
  for (Complex& value : values) {
    value /= norm;
  }
}

// Complex EISPACK helpers:
// - `pythag` and `csroot`: John Burkardt's MIT-licensed C port
//   https://people.sc.fsu.edu/~jburkardt/c_src/eispack/eispack.c
// - `cdiv`, `corth`, `comqr2`: NETLIB EISPACK Fortran
//   https://netlib.org/eispack/cdiv.f
//   https://netlib.org/eispack/corth.f
//   https://netlib.org/eispack/comqr2.f
// Local names: complexEigenStableHypot (pythag), complexEigenComplexSqrt
// (csroot), complexEigenComplexDivide (cdiv), complexEigenReduceToHessenberg
// (corth), complexEigenQrSolve (comqr2).

namespace {

/// Row-major `ld x n` matrix view for EISPACK storage (`values[row + col *
/// ld]`).
class EispackMatrixView {
public:
  EispackMatrixView(MutableArrayRef<double> values, const int ld)
      : values_(values), ld_(ld) {}

  [[nodiscard]] static std::size_t rowMajorIndex(const int row, const int col,
                                                 const int ld) {
    return static_cast<std::size_t>(row) +
           (static_cast<std::size_t>(col) * static_cast<std::size_t>(ld));
  }

  [[nodiscard]] double& at(const int row, const int col) {
    return values_[rowMajorIndex(row, col, ld_)];
  }

  [[nodiscard]] const double& at(const int row, const int col) const {
    return values_[rowMajorIndex(row, col, ld_)];
  }

private:
  MutableArrayRef<double> values_;
  int ld_;
};

} // namespace

[[nodiscard]] static double complexEigenStableHypot(const double a,
                                                    const double b) {
  double p = std::max(std::abs(a), std::abs(b));
  if (p != 0.0) {
    double r = std::min(std::abs(a), std::abs(b)) / p;
    r = r * r;
    while (true) {
      const double t = 4.0 + r;
      if (t == 4.0) {
        break;
      }
      const double s = r / t;
      const double u = 1.0 + (2.0 * s);
      p = u * p;
      r = (s / u) * (s / u) * r;
    }
  }
  return p;
}

[[nodiscard]] static std::pair<double, double>
complexEigenComplexSqrt(const double xr, const double xi) {
  const double inputReal = xr;
  const double inputImag = xi;
  const double s =
      std::sqrt(0.5 * (complexEigenStableHypot(inputReal, inputImag) +
                       std::abs(inputReal)));

  double yr = 0.0;
  double yi = 0.0;
  if (0.0 <= inputReal) {
    yr = s;
  }

  double sSign = s;
  if (inputImag < 0.0) {
    sSign = -s;
  }

  if (inputReal <= 0.0) {
    yi = sSign;
  }

  if (inputReal < 0.0) {
    yr = 0.5 * (inputImag / yi);
  } else if (0.0 < inputReal) {
    yi = 0.5 * (inputImag / yr);
  }
  return {yr, yi};
}

[[nodiscard]] static std::pair<double, double>
complexEigenComplexDivide(const double dividendReal, const double dividendImag,
                          const double divisorReal, const double divisorImag) {
  const double s = std::abs(divisorReal) + std::abs(divisorImag);
  const double dividendRealScaled = dividendReal / s;
  const double dividendImagScaled = dividendImag / s;
  const double divisorRealScaled = divisorReal / s;
  const double divisorImagScaled = divisorImag / s;
  const double denom = (divisorRealScaled * divisorRealScaled) +
                       (divisorImagScaled * divisorImagScaled);
  return {((dividendRealScaled * divisorRealScaled) +
           (dividendImagScaled * divisorImagScaled)) /
              denom,
          ((dividendImagScaled * divisorRealScaled) -
           (dividendRealScaled * divisorImagScaled)) /
              denom};
}

static void
complexEigenReduceToHessenberg(const int leadingDim, const int order,
                               const int rowLow, const int rowHigh,
                               MutableArrayRef<double> matrixRealBuf,
                               MutableArrayRef<double> matrixImagBuf,
                               MutableArrayRef<double> householderRealBuf,
                               MutableArrayRef<double> householderImagBuf) {
  EispackMatrixView matrixReal(matrixRealBuf, leadingDim);
  EispackMatrixView matrixImag(matrixImagBuf, leadingDim);
  const auto householderRealAt =
      [&householderRealBuf](const int index) -> double& {
    return householderRealBuf[static_cast<std::size_t>(index)];
  };
  const auto householderImagAt =
      [&householderImagBuf](const int index) -> double& {
    return householderImagBuf[static_cast<std::size_t>(index)];
  };

  const int kp1 = rowLow + 1;
  const int la = rowHigh - 1;
  if (la < kp1) {
    return;
  }

  for (int m = kp1; m <= la; ++m) {
    double h = 0.0;
    householderRealAt(m) = 0.0;
    householderImagAt(m) = 0.0;
    double scale = 0.0;
    const int subCol = m - 1;
    for (int i = m; i <= rowHigh; ++i) {
      scale += std::abs(matrixReal.at(i, subCol)) +
               std::abs(matrixImag.at(i, subCol));
    }

    if (scale == 0.0) {
      continue;
    }

    const int mp = m + rowHigh;
    for (int ii = m; ii <= rowHigh; ++ii) {
      const int i = mp - ii;
      householderRealAt(i) = matrixReal.at(i, subCol) / scale;
      householderImagAt(i) = matrixImag.at(i, subCol) / scale;
      h += (householderRealAt(i) * householderRealAt(i)) +
           (householderImagAt(i) * householderImagAt(i));
    }

    double g = std::sqrt(h);
    const double f =
        complexEigenStableHypot(householderRealAt(m), householderImagAt(m));
    if (f == 0.0) {
      householderRealAt(m) = g;
      matrixReal.at(m, subCol) = scale;
    } else {
      h += f * g;
      g /= f;
      householderRealAt(m) = (1.0 + g) * householderRealAt(m);
      householderImagAt(m) = (1.0 + g) * householderImagAt(m);
    }

    for (int j = m; j < order; ++j) {
      double fr = 0.0;
      double fi = 0.0;
      for (int ii = m; ii <= rowHigh; ++ii) {
        const int i = mp - ii;
        fr += (householderRealAt(i) * matrixReal.at(i, j)) +
              (householderImagAt(i) * matrixImag.at(i, j));
        fi += (householderRealAt(i) * matrixImag.at(i, j)) -
              (householderImagAt(i) * matrixReal.at(i, j));
      }
      fr /= h;
      fi /= h;
      for (int i = m; i <= rowHigh; ++i) {
        matrixReal.at(i, j) -=
            (fr * householderRealAt(i)) - (fi * householderImagAt(i));
        matrixImag.at(i, j) -=
            (fr * householderImagAt(i)) + (fi * householderRealAt(i));
      }
    }

    for (int i = 0; i <= rowHigh; ++i) {
      double fr = 0.0;
      double fi = 0.0;
      for (int jj = m; jj <= rowHigh; ++jj) {
        const int j = mp - jj;
        fr += (householderRealAt(j) * matrixReal.at(i, j)) -
              (householderImagAt(j) * matrixImag.at(i, j));
        fi += (householderRealAt(j) * matrixImag.at(i, j)) +
              (householderImagAt(j) * matrixReal.at(i, j));
      }
      fr /= h;
      fi /= h;
      for (int j = m; j <= rowHigh; ++j) {
        matrixReal.at(i, j) -=
            (fr * householderRealAt(j)) + (fi * householderImagAt(j));
        matrixImag.at(i, j) +=
            (fr * householderImagAt(j)) - (fi * householderRealAt(j));
      }
    }

    householderRealAt(m) = scale * householderRealAt(m);
    householderImagAt(m) = scale * householderImagAt(m);
    matrixReal.at(m, subCol) = -g * matrixReal.at(m, subCol);
    matrixImag.at(m, subCol) = -g * matrixImag.at(m, subCol);
  }
}

[[nodiscard]] static int
complexEigenQrSolve(const int leadingDim, const int order, const int rowLow,
                    const int rowHigh,
                    MutableArrayRef<double> householderRealBuf,
                    MutableArrayRef<double> householderImagBuf,
                    MutableArrayRef<double> hessenbergRealBuf,
                    MutableArrayRef<double> hessenbergImagBuf,
                    MutableArrayRef<double> eigenvalueRealBuf,
                    MutableArrayRef<double> eigenvalueImagBuf,
                    MutableArrayRef<double> eigenvectorRealBuf,
                    MutableArrayRef<double> eigenvectorImagBuf) {
  EispackMatrixView hessenbergReal(hessenbergRealBuf, leadingDim);
  EispackMatrixView hessenbergImag(hessenbergImagBuf, leadingDim);
  EispackMatrixView eigenvectorReal(eigenvectorRealBuf, leadingDim);
  EispackMatrixView eigenvectorImag(eigenvectorImagBuf, leadingDim);
  const auto householderRealAt =
      [&householderRealBuf](const int index) -> double& {
    return householderRealBuf[static_cast<std::size_t>(index)];
  };
  const auto householderImagAt =
      [&householderImagBuf](const int index) -> double& {
    return householderImagBuf[static_cast<std::size_t>(index)];
  };
  const auto eigenvalueRealAt =
      [&eigenvalueRealBuf](const int index) -> double& {
    return eigenvalueRealBuf[static_cast<std::size_t>(index)];
  };
  const auto eigenvalueImagAt =
      [&eigenvalueImagBuf](const int index) -> double& {
    return eigenvalueImagBuf[static_cast<std::size_t>(index)];
  };

  for (int j = 0; j < order; ++j) {
    for (int i = 0; i < order; ++i) {
      eigenvectorReal.at(i, j) = 0.0;
      eigenvectorImag.at(i, j) = 0.0;
    }
    eigenvectorReal.at(j, j) = 1.0;
  }

  const int numHouseholderReflections = rowHigh - rowLow - 1;
  if (numHouseholderReflections > 0) {
    for (int ii = 1; ii <= numHouseholderReflections; ++ii) {
      const int i = rowHigh - ii;
      if (householderRealAt(i) == 0.0 && householderImagAt(i) == 0.0) {
        continue;
      }
      if (hessenbergReal.at(i, i - 1) == 0.0 &&
          hessenbergImag.at(i, i - 1) == 0.0) {
        continue;
      }
      const double householderNorm =
          (hessenbergReal.at(i, i - 1) * householderRealAt(i)) +
          (hessenbergImag.at(i, i - 1) * householderImagAt(i));
      const int ip1 = i + 1;
      for (int k = ip1; k <= rowHigh; ++k) {
        householderRealAt(k) = hessenbergReal.at(k, i - 1);
        householderImagAt(k) = hessenbergImag.at(k, i - 1);
      }
      for (int j = i; j <= rowHigh; ++j) {
        double sr = 0.0;
        double si = 0.0;
        for (int k = i; k <= rowHigh; ++k) {
          sr += (householderRealAt(k) * eigenvectorReal.at(k, j)) +
                (householderImagAt(k) * eigenvectorImag.at(k, j));
          si += (householderRealAt(k) * eigenvectorImag.at(k, j)) -
                (householderImagAt(k) * eigenvectorReal.at(k, j));
        }
        sr /= householderNorm;
        si /= householderNorm;
        for (int k = i; k <= rowHigh; ++k) {
          eigenvectorReal.at(k, j) +=
              (sr * householderRealAt(k)) - (si * householderImagAt(k));
          eigenvectorImag.at(k, j) +=
              (sr * householderImagAt(k)) + (si * householderRealAt(k));
        }
      }
    }
  }

  if (numHouseholderReflections >= 0) {
    const int hessLow = rowLow + 1;
    for (int i = hessLow; i <= rowHigh; ++i) {
      const int ll = std::min(i + 1, rowHigh);
      if (hessenbergImag.at(i, i - 1) == 0.0) {
        continue;
      }
      const double subdiagonalNorm = complexEigenStableHypot(
          hessenbergReal.at(i, i - 1), hessenbergImag.at(i, i - 1));
      const double yr = hessenbergReal.at(i, i - 1) / subdiagonalNorm;
      const double yi = hessenbergImag.at(i, i - 1) / subdiagonalNorm;
      hessenbergReal.at(i, i - 1) = subdiagonalNorm;
      hessenbergImag.at(i, i - 1) = 0.0;
      for (int j = i; j < order; ++j) {
        const double si =
            (yr * hessenbergImag.at(i, j)) - (yi * hessenbergReal.at(i, j));
        hessenbergReal.at(i, j) =
            (yr * hessenbergReal.at(i, j)) + (yi * hessenbergImag.at(i, j));
        hessenbergImag.at(i, j) = si;
      }
      for (int j = 0; j <= ll; ++j) {
        const double si =
            (yr * hessenbergImag.at(j, i)) + (yi * hessenbergReal.at(j, i));
        hessenbergReal.at(j, i) =
            (yr * hessenbergReal.at(j, i)) - (yi * hessenbergImag.at(j, i));
        hessenbergImag.at(j, i) = si;
      }
      for (int j = rowLow; j <= rowHigh; ++j) {
        const double si =
            (yr * eigenvectorImag.at(j, i)) + (yi * eigenvectorReal.at(j, i));
        eigenvectorReal.at(j, i) =
            (yr * eigenvectorReal.at(j, i)) - (yi * eigenvectorImag.at(j, i));
        eigenvectorImag.at(j, i) = si;
      }
    }
  }

  for (int i = 0; i < order; ++i) {
    if (i >= rowLow && i <= rowHigh) {
      continue;
    }
    eigenvalueRealAt(i) = hessenbergReal.at(i, i);
    eigenvalueImagAt(i) = hessenbergImag.at(i, i);
  }

  int activeEigenIndex = rowHigh;
  double eigenSumReal = 0.0;
  double eigenSumImag = 0.0;
  int qrIterationBudget = 30 * order;

  while (activeEigenIndex >= rowLow) {
    int qrIterationStep = 0;
    const int activeEigenIndexMinus1 = activeEigenIndex - 1;

    while (true) {
      int l = rowLow;
      for (int ll = rowLow; ll <= activeEigenIndex; ++ll) {
        l = activeEigenIndex + rowLow - ll;
        if (l == rowLow) {
          break;
        }
        const double tst1Local = std::abs(hessenbergReal.at((l - 1), l - 1)) +
                                 std::abs(hessenbergImag.at((l - 1), l - 1)) +
                                 std::abs(hessenbergReal.at(l, l)) +
                                 std::abs(hessenbergImag.at(l, l));
        const double tst2Local =
            tst1Local + std::abs(hessenbergReal.at(l, l - 1));
        if (tst2Local == tst1Local) {
          break;
        }
      }

      if (l == activeEigenIndex) {
        break;
      }
      if (qrIterationBudget == 0) {
        return activeEigenIndex;
      }

      double sr = hessenbergReal.at(activeEigenIndex, activeEigenIndex);
      double si = hessenbergImag.at(activeEigenIndex, activeEigenIndex);
      if (qrIterationStep == 10 || qrIterationStep == 20) {
        sr = std::abs(
            hessenbergReal.at(activeEigenIndex, activeEigenIndexMinus1));
        if (activeEigenIndex >= rowLow + 2) {
          sr += std::abs(
              hessenbergReal.at(activeEigenIndexMinus1, activeEigenIndex - 2));
        }
        si = 0.0;
      } else {
        double xr =
            hessenbergReal.at(activeEigenIndexMinus1, activeEigenIndex) *
            hessenbergReal.at(activeEigenIndex, activeEigenIndexMinus1);
        double xi =
            hessenbergImag.at(activeEigenIndexMinus1, activeEigenIndex) *
            hessenbergReal.at(activeEigenIndex, activeEigenIndexMinus1);
        if (xr != 0.0 || xi != 0.0) {
          const double yr = (hessenbergReal.at(activeEigenIndexMinus1,
                                               activeEigenIndexMinus1) -
                             sr) /
                            2.0;
          const double yi = (hessenbergImag.at(activeEigenIndexMinus1,
                                               activeEigenIndexMinus1) -
                             si) /
                            2.0;
          auto [zzr, zzi] = complexEigenComplexSqrt((yr * yr) - (yi * yi) + xr,
                                                    (2.0 * yr * yi) + xi);
          if ((yr * zzr) + (yi * zzi) < 0.0) {
            zzr = -zzr;
            zzi = -zzi;
          }
          std::tie(xr, xi) =
              complexEigenComplexDivide(xr, xi, yr + zzr, yi + zzi);
          sr -= xr;
          si -= xi;
        }
      }

      for (int i = rowLow; i <= activeEigenIndex; ++i) {
        hessenbergReal.at(i, i) -= sr;
        hessenbergImag.at(i, i) -= si;
      }
      eigenSumReal += sr;
      eigenSumImag += si;
      ++qrIterationStep;
      --qrIterationBudget;

      {
        const int lp1 = l + 1;
        for (int i = lp1; i <= activeEigenIndex; ++i) {
          sr = hessenbergReal.at(i, i - 1);
          hessenbergReal.at(i, i - 1) = 0.0;
          const double stepNorm = complexEigenStableHypot(
              complexEigenStableHypot(hessenbergReal.at((i - 1), i - 1),
                                      hessenbergImag.at((i - 1), i - 1)),
              sr);
          const double xr = hessenbergReal.at((i - 1), i - 1) / stepNorm;
          eigenvalueRealAt(i - 1) = xr;
          const double xi = hessenbergImag.at((i - 1), i - 1) / stepNorm;
          eigenvalueImagAt(i - 1) = xi;
          hessenbergReal.at((i - 1), i - 1) = stepNorm;
          hessenbergImag.at((i - 1), i - 1) = 0.0;
          hessenbergImag.at(i, i - 1) = sr / stepNorm;

          for (int j = i; j < order; ++j) {
            const double yr = hessenbergReal.at((i - 1), j);
            const double yi = hessenbergImag.at((i - 1), j);
            const double zzr = hessenbergReal.at(i, j);
            const double zzi = hessenbergImag.at(i, j);
            hessenbergReal.at((i - 1), j) =
                (xr * yr) + (xi * yi) + (hessenbergImag.at(i, i - 1) * zzr);
            hessenbergImag.at((i - 1), j) =
                (xr * yi) - (xi * yr) + (hessenbergImag.at(i, i - 1) * zzi);
            hessenbergReal.at(i, j) =
                (xr * zzr) - (xi * zzi) - (hessenbergImag.at(i, i - 1) * yr);
            hessenbergImag.at(i, j) =
                (xr * zzi) + (xi * zzr) - (hessenbergImag.at(i, i - 1) * yi);
          }
        }
      }

      si = hessenbergImag.at(activeEigenIndex, activeEigenIndex);
      if (si != 0.0) {
        const double stepNorm = complexEigenStableHypot(
            hessenbergReal.at(activeEigenIndex, activeEigenIndex), si);
        sr = hessenbergReal.at(activeEigenIndex, activeEigenIndex) / stepNorm;
        si /= stepNorm;
        hessenbergReal.at(activeEigenIndex, activeEigenIndex) = stepNorm;
        hessenbergImag.at(activeEigenIndex, activeEigenIndex) = 0.0;
        if (activeEigenIndex != order - 1) {
          const int ip1 = activeEigenIndex + 1;
          for (int j = ip1; j < order; ++j) {
            const double yr = hessenbergReal.at(activeEigenIndex, j);
            const double yi = hessenbergImag.at(activeEigenIndex, j);
            hessenbergReal.at(activeEigenIndex, j) = (sr * yr) + (si * yi);
            hessenbergImag.at(activeEigenIndex, j) = (sr * yi) - (si * yr);
          }
        }
      }

      {
        const int lp1 = l + 1;
        for (int j = lp1; j <= activeEigenIndex; ++j) {
          const double xr = eigenvalueRealAt(j - 1);
          const double xi = eigenvalueImagAt(j - 1);
          for (int i = 0; i <= j; ++i) {
            const double yr = hessenbergReal.at(i, j - 1);
            double yi = 0.0;
            const double zzr = hessenbergReal.at(i, j);
            const double zzi = hessenbergImag.at(i, j);
            if (i != j) {
              yi = hessenbergImag.at(i, j - 1);
              hessenbergImag.at(i, j - 1) =
                  (xr * yi) + (xi * yr) + (hessenbergImag.at(j, j - 1) * zzi);
            }
            hessenbergReal.at(i, j - 1) =
                (xr * yr) - (xi * yi) + (hessenbergImag.at(j, j - 1) * zzr);
            hessenbergReal.at(i, j) =
                (xr * zzr) + (xi * zzi) - (hessenbergImag.at(j, j - 1) * yr);
            hessenbergImag.at(i, j) =
                (xr * zzi) - (xi * zzr) - (hessenbergImag.at(j, j - 1) * yi);
          }
          for (int i = rowLow; i <= rowHigh; ++i) {
            const double yr = eigenvectorReal.at(i, j - 1);
            const double yi = eigenvectorImag.at(i, j - 1);
            const double zzr = eigenvectorReal.at(i, j);
            const double zzi = eigenvectorImag.at(i, j);
            eigenvectorReal.at(i, j - 1) =
                (xr * yr) - (xi * yi) + (hessenbergImag.at(j, j - 1) * zzr);
            eigenvectorImag.at(i, j - 1) =
                (xr * yi) + (xi * yr) + (hessenbergImag.at(j, j - 1) * zzi);
            eigenvectorReal.at(i, j) =
                (xr * zzr) + (xi * zzi) - (hessenbergImag.at(j, j - 1) * yr);
            eigenvectorImag.at(i, j) =
                (xr * zzi) - (xi * zzr) - (hessenbergImag.at(j, j - 1) * yi);
          }
        }
      }

      if (si != 0.0) {
        for (int i = 0; i <= activeEigenIndex; ++i) {
          const double yr = hessenbergReal.at(i, activeEigenIndex);
          const double yi = hessenbergImag.at(i, activeEigenIndex);
          hessenbergReal.at(i, activeEigenIndex) = (sr * yr) - (si * yi);
          hessenbergImag.at(i, activeEigenIndex) = (sr * yi) + (si * yr);
        }
        for (int i = rowLow; i <= rowHigh; ++i) {
          const double yr = eigenvectorReal.at(i, activeEigenIndex);
          const double yi = eigenvectorImag.at(i, activeEigenIndex);
          eigenvectorReal.at(i, activeEigenIndex) = (sr * yr) - (si * yi);
          eigenvectorImag.at(i, activeEigenIndex) = (sr * yi) + (si * yr);
        }
      }
    }

    hessenbergReal.at(activeEigenIndex, activeEigenIndex) += eigenSumReal;
    eigenvalueRealAt(activeEigenIndex) =
        hessenbergReal.at(activeEigenIndex, activeEigenIndex);
    hessenbergImag.at(activeEigenIndex, activeEigenIndex) += eigenSumImag;
    eigenvalueImagAt(activeEigenIndex) =
        hessenbergImag.at(activeEigenIndex, activeEigenIndex);
    activeEigenIndex = activeEigenIndexMinus1;
  }

  double norm = 0.0;
  for (int i = 0; i < order; ++i) {
    for (int j = i; j < order; ++j) {
      const double matrixNorm =
          std::abs(hessenbergReal.at(i, j)) + std::abs(hessenbergImag.at(i, j));
      norm = std::max(norm, matrixNorm);
    }
  }

  if (order != 1 && norm != 0.0) {
    for (int nn = 2; nn <= order; ++nn) {
      activeEigenIndex = order + 1 - nn;
      double xr = eigenvalueRealAt(activeEigenIndex);
      double xi = eigenvalueImagAt(activeEigenIndex);
      hessenbergReal.at(activeEigenIndex, activeEigenIndex) = 1.0;
      hessenbergImag.at(activeEigenIndex, activeEigenIndex) = 0.0;
      for (int ii = 1; ii <= activeEigenIndex; ++ii) {
        const int i = activeEigenIndex - ii;
        double zzr = 0.0;
        double zzi = 0.0;
        const int ip1 = i + 1;
        for (int j = ip1; j <= activeEigenIndex; ++j) {
          zzr += (hessenbergReal.at(i, j) *
                  hessenbergReal.at(j, activeEigenIndex)) -
                 (hessenbergImag.at(i, j) *
                  hessenbergImag.at(j, activeEigenIndex));
          zzi += (hessenbergReal.at(i, j) *
                  hessenbergImag.at(j, activeEigenIndex)) +
                 (hessenbergImag.at(i, j) *
                  hessenbergReal.at(j, activeEigenIndex));
        }
        double yr = xr - eigenvalueRealAt(i);
        double yi = xi - eigenvalueImagAt(i);
        if (yr == 0.0 && yi == 0.0) {
          double tst1 = norm;
          yr = tst1;
          double tst2 = 0.0;
          do {
            yr = 0.01 * yr;
            tst2 = norm + yr;
          } while (tst2 <= tst1);
        }
        auto [divReal, divImag] = complexEigenComplexDivide(zzr, zzi, yr, yi);
        hessenbergReal.at(i, activeEigenIndex) = divReal;
        hessenbergImag.at(i, activeEigenIndex) = divImag;
        const double trLocal =
            std::abs(hessenbergReal.at(i, activeEigenIndex)) +
            std::abs(hessenbergImag.at(i, activeEigenIndex));
        if (trLocal == 0.0) {
          continue;
        }
        const double tst1 = trLocal;
        const double tst2 = tst1 + (1.0 / tst1);
        if (tst2 > tst1) {
          continue;
        }
        for (int j = i; j <= activeEigenIndex; ++j) {
          hessenbergReal.at(j, activeEigenIndex) /= trLocal;
          hessenbergImag.at(j, activeEigenIndex) /= trLocal;
        }
      }
    }

    for (int i = 0; i < order; ++i) {
      if (i >= rowLow && i <= rowHigh) {
        continue;
      }
      for (int j = i; j < order; ++j) {
        eigenvectorReal.at(i, j) = hessenbergReal.at(i, j);
        eigenvectorImag.at(i, j) = hessenbergImag.at(i, j);
      }
    }

    for (int jj = rowLow; jj <= rowHigh; ++jj) {
      const int j = rowHigh + rowLow - jj;
      const int m = std::min(j, rowHigh);
      for (int i = rowLow; i <= rowHigh; ++i) {
        double zzr = 0.0;
        double zzi = 0.0;
        for (int k = rowLow; k <= m; ++k) {
          zzr += (eigenvectorReal.at(i, k) * hessenbergReal.at(k, j)) -
                 (eigenvectorImag.at(i, k) * hessenbergImag.at(k, j));
          zzi += (eigenvectorReal.at(i, k) * hessenbergImag.at(k, j)) +
                 (eigenvectorImag.at(i, k) * hessenbergReal.at(k, j));
        }
        eigenvectorReal.at(i, j) = zzr;
        eigenvectorImag.at(i, j) = zzi;
      }
    }
  }

  return 0;
}

constexpr int K_COMPLEX_EIGEN4_SIZE = 4;
constexpr int K_COMPLEX_EIGEN4_LD = K_COMPLEX_EIGEN4_SIZE;
// EISPACK `comqr2` backsubstitution uses column/row index `n` in an `nm`-stride
// layout.
constexpr int K_COMPLEX_EIGEN4_MATRIX_STORAGE =
    (K_COMPLEX_EIGEN4_LD * K_COMPLEX_EIGEN4_SIZE) + K_COMPLEX_EIGEN4_SIZE + 1;
constexpr int K_COMPLEX_EIGEN4_EIGENVALUE_STORAGE = K_COMPLEX_EIGEN4_SIZE + 1;

[[nodiscard]] static double
eigenvectorColumnNorm(const int order, const int col, const int leadingDim,
                      const ArrayRef<double> eigenvectorReal,
                      const ArrayRef<double> eigenvectorImag) {
  double normSq = 0.0;
  for (int row = 0; row < order; ++row) {
    const std::size_t idx =
        EispackMatrixView::rowMajorIndex(row, col, leadingDim);
    normSq += (eigenvectorReal[idx] * eigenvectorReal[idx]) +
              (eigenvectorImag[idx] * eigenvectorImag[idx]);
  }
  return std::sqrt(normSq);
}

[[nodiscard]] static Complex normalizedEigenvectorEntry(const double real,
                                                        const double imag,
                                                        const double norm) {
  if (norm > MATRIX_TOLERANCE) {
    return {real / norm, imag / norm};
  }
  return {real, imag};
}

[[nodiscard]] static ComplexEigen4 assembleComplexEigen4(
    const std::array<double, K_COMPLEX_EIGEN4_EIGENVALUE_STORAGE>&
        eigenvalueReal,
    const std::array<double, K_COMPLEX_EIGEN4_EIGENVALUE_STORAGE>&
        eigenvalueImag,
    const std::array<double, K_COMPLEX_EIGEN4_MATRIX_STORAGE>& eigenvectorReal,
    const std::array<double, K_COMPLEX_EIGEN4_MATRIX_STORAGE>&
        eigenvectorImag) {
  ComplexEigen4 result;
  for (int col = 0; col < K_COMPLEX_EIGEN4_SIZE; ++col) {
    result.eigenvalues[static_cast<std::size_t>(col)] =
        Complex(eigenvalueReal[static_cast<std::size_t>(col)],
                eigenvalueImag[static_cast<std::size_t>(col)]);
    const double norm =
        eigenvectorColumnNorm(K_COMPLEX_EIGEN4_SIZE, col, K_COMPLEX_EIGEN4_SIZE,
                              eigenvectorReal, eigenvectorImag);
    for (int row = 0; row < K_COMPLEX_EIGEN4_SIZE; ++row) {
      const std::size_t idx =
          EispackMatrixView::rowMajorIndex(row, col, K_COMPLEX_EIGEN4_SIZE);
      result.eigenvectors(static_cast<std::size_t>(row),
                          static_cast<std::size_t>(col)) =
          normalizedEigenvectorEntry(eigenvectorReal[idx], eigenvectorImag[idx],
                                     norm);
    }
  }
  return result;
}

static void splitMatrix4x4ToRealImag(
    const Matrix4x4& matrix,
    std::array<double, K_COMPLEX_EIGEN4_MATRIX_STORAGE>& matrixReal,
    std::array<double, K_COMPLEX_EIGEN4_MATRIX_STORAGE>& matrixImag) {
  for (std::size_t row = 0; row < Matrix4x4::K_ROWS; ++row) {
    for (std::size_t col = 0; col < Matrix4x4::K_COLS; ++col) {
      const Complex& value = matrix(row, col);
      const std::size_t idx = row + (col * Matrix4x4::K_ROWS);
      matrixReal[idx] = std::real(value);
      matrixImag[idx] = std::imag(value);
    }
  }
}

[[nodiscard]] std::optional<ComplexEigen4>
decomposeComplexEigen4(const Matrix4x4& matrix) {
  constexpr int order = K_COMPLEX_EIGEN4_SIZE;
  constexpr int leadingDim = order;
  constexpr int rowLow = 0;
  constexpr int rowHigh = order - 1;

  std::array<double, K_COMPLEX_EIGEN4_MATRIX_STORAGE> matrixReal{};
  std::array<double, K_COMPLEX_EIGEN4_MATRIX_STORAGE> matrixImag{};
  splitMatrix4x4ToRealImag(matrix, matrixReal, matrixImag);

  std::array<double, 4> householderReal{};
  std::array<double, 4> householderImag{};
  std::array<double, K_COMPLEX_EIGEN4_EIGENVALUE_STORAGE> eigenvalueReal{};
  std::array<double, K_COMPLEX_EIGEN4_EIGENVALUE_STORAGE> eigenvalueImag{};
  std::array<double, K_COMPLEX_EIGEN4_MATRIX_STORAGE> eigenvectorReal{};
  std::array<double, K_COMPLEX_EIGEN4_MATRIX_STORAGE> eigenvectorImag{};

  complexEigenReduceToHessenberg(leadingDim, order, rowLow, rowHigh, matrixReal,
                                 matrixImag, householderReal, householderImag);
  const int convergenceStatus = complexEigenQrSolve(
      leadingDim, order, rowLow, rowHigh, householderReal, householderImag,
      matrixReal, matrixImag, eigenvalueReal, eigenvalueImag, eigenvectorReal,
      eigenvectorImag);
  if (convergenceStatus != 0) {
    return std::nullopt;
  }
  return assembleComplexEigen4(eigenvalueReal, eigenvalueImag, eigenvectorReal,
                               eigenvectorImag);
}

[[nodiscard]] std::optional<ComplexEigen>
decomposeComplexEigen1x1(const DynamicMatrix& matrix) {
  ComplexEigen result;
  result.eigenvalues.push_back(matrix(0, 0));
  result.eigenvectors = DynamicMatrix(1);
  result.eigenvectors(0, 0) = 1.0;
  return result;
}

[[nodiscard]] std::optional<ComplexEigen2>
decomposeComplexEigen2(const Matrix2x2& matrix) {
  const Complex a = matrix(0, 0);
  const Complex b = matrix(0, 1);
  const Complex c = matrix(1, 0);
  const Complex d = matrix(1, 1);
  const Complex trace = a + d;
  const Complex determinant = a * d - b * c;
  const Complex discriminant = std::sqrt(trace * trace - 4.0 * determinant);
  const Complex lambda0 = (trace + discriminant) * 0.5;
  const Complex lambda1 = (trace - discriminant) * 0.5;
  if (!isFiniteComplex(lambda0) || !isFiniteComplex(lambda1)) {
    return std::nullopt;
  }

  if (std::abs(b) <= MATRIX_TOLERANCE && std::abs(c) <= MATRIX_TOLERANCE) {
    if (!isFiniteComplex(a) || !isFiniteComplex(d)) {
      return std::nullopt;
    }
    ComplexEigen2 result;
    result.eigenvalues = {a, d};
    result.eigenvectors(0, 0) = 1.0;
    result.eigenvectors(1, 0) = 0.0;
    result.eigenvectors(0, 1) = 0.0;
    result.eigenvectors(1, 1) = 1.0;
    return result;
  }

  auto eigenvectorFor = [&](const Complex& lambda) -> SmallVector<Complex> {
    SmallVector<Complex> vector(2, Complex{0.0, 0.0});
    if (std::abs(b) > MATRIX_TOLERANCE) {
      vector[0] = b;
      vector[1] = lambda - a;
    } else {
      vector[0] = lambda - d;
      vector[1] = c;
    }
    normalizeInPlace(vector);
    return vector;
  };

  const SmallVector<Complex> vector0 = eigenvectorFor(lambda0);
  const SmallVector<Complex> vector1 = eigenvectorFor(lambda1);

  ComplexEigen2 result;
  result.eigenvalues = {lambda0, lambda1};
  result.eigenvectors(0, 0) = vector0[0];
  result.eigenvectors(1, 0) = vector0[1];
  result.eigenvectors(0, 1) = vector1[0];
  result.eigenvectors(1, 1) = vector1[1];
  return result;
}

[[nodiscard]] ComplexEigen fromComplexEigen(const ComplexEigen2& eigen2) {
  ComplexEigen result;
  result.eigenvalues.assign(eigen2.eigenvalues.begin(),
                            eigen2.eigenvalues.end());
  result.eigenvectors = DynamicMatrix(eigen2.eigenvectors);
  return result;
}

[[nodiscard]] ComplexEigen fromComplexEigen(const ComplexEigen4& eigen4) {
  ComplexEigen result;
  result.eigenvalues.assign(eigen4.eigenvalues.begin(),
                            eigen4.eigenvalues.end());
  result.eigenvectors = DynamicMatrix(eigen4.eigenvectors);
  return result;
}

[[nodiscard]] std::optional<ComplexEigen>
decomposeComplexEigenDynamic(const DynamicMatrix& matrix) {
  const std::int64_t dim = matrix.rows();
  if (dim != matrix.cols()) {
    return std::nullopt;
  }
  if (dim > static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
    return std::nullopt;
  }
  assert(dim >= 3 && dim != 4);
  const int order = static_cast<int>(dim);
  const int leadingDim = order;
  const int rowLow = 0;
  const int rowHigh = order - 1;

  const std::size_t matrixStorage =
      (static_cast<std::size_t>(leadingDim) * static_cast<std::size_t>(order)) +
      static_cast<std::size_t>(order) + 1U;
  SmallVector<double> matrixReal(matrixStorage);
  SmallVector<double> matrixImag(matrixStorage);
  for (int row = 0; row < order; ++row) {
    for (int col = 0; col < order; ++col) {
      const Complex value = matrix(row, col);
      const std::size_t idx =
          EispackMatrixView::rowMajorIndex(row, col, leadingDim);
      matrixReal[idx] = std::real(value);
      matrixImag[idx] = std::imag(value);
    }
  }

  SmallVector<double> householderReal(static_cast<std::size_t>(order));
  SmallVector<double> householderImag(static_cast<std::size_t>(order));
  SmallVector<double> eigenvalueReal(static_cast<std::size_t>(order) + 1U);
  SmallVector<double> eigenvalueImag(static_cast<std::size_t>(order) + 1U);
  SmallVector<double> eigenvectorReal(matrixStorage);
  SmallVector<double> eigenvectorImag(matrixStorage);

  complexEigenReduceToHessenberg(leadingDim, order, rowLow, rowHigh, matrixReal,
                                 matrixImag, householderReal, householderImag);
  const int convergenceStatus = complexEigenQrSolve(
      leadingDim, order, rowLow, rowHigh, householderReal, householderImag,
      matrixReal, matrixImag, eigenvalueReal, eigenvalueImag, eigenvectorReal,
      eigenvectorImag);
  if (convergenceStatus != 0) {
    return std::nullopt;
  }

  ComplexEigen result;
  result.eigenvalues.reserve(static_cast<std::size_t>(order));
  result.eigenvectors = DynamicMatrix(dim);
  for (int col = 0; col < order; ++col) {
    result.eigenvalues.emplace_back(
        eigenvalueReal[static_cast<std::size_t>(col)],
        eigenvalueImag[static_cast<std::size_t>(col)]);
    const double norm = eigenvectorColumnNorm(order, col, leadingDim,
                                              eigenvectorReal, eigenvectorImag);
    for (int row = 0; row < order; ++row) {
      const std::size_t idx =
          EispackMatrixView::rowMajorIndex(row, col, leadingDim);
      result.eigenvectors(row, col) = normalizedEigenvectorEntry(
          eigenvectorReal[idx], eigenvectorImag[idx], norm);
    }
  }
  return result;
}
} // namespace mlir::qco

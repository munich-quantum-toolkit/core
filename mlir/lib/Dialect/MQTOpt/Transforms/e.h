//
// Analytical Eigen-decomposition for 4x4 and 3x3 matrices
//
//
//   Robin Straebler - George Terzakis
//
//	 University of Portsmouth 2016
//

#ifndef __EIGENDECOMPOSE_H__
#define __EIGENDECOMPOSE_H__

#include "Helpers.h"

#include <algorithm>
#include <iostream>
#include <math.h>
#include <ostream>
#include <utility>
#include <vector>

// The 4x4 implies that the namespace contains algorithms
// for the eigen decomposition of 3x3 or 4x4 matrices.
//
namespace mqt::ir::opt {

namespace PolySolvers {

// see http://mathworld.wolfram.com/QuadraticFormula.html
template <typename P> inline std::vector<P> SolveQuadratic(P a, P b, P c) {
  P delta = b * b - 4 * a * c;
  P inv_2a, sqrt_delta;
  P x1, x0;

  if (delta < 0)
    return std::vector<P>();
  if (a == 0) {
    // solve first order system
    x1 = 0;
    if (b != 0) {
      x0 = -c / b;
      return std::vector<P>({x0});
    }

    x0 = 0;
    return std::vector<P>();
  }

  inv_2a = 0.5 / a;
  if (delta == 0) {
    x0 = -b * inv_2a;
    x1 = x0;
    return std::vector<P>({x0});
  }

  sqrt_delta = sqrt(delta);
  x0 = (-b + sqrt_delta) * inv_2a;
  x1 = (-b - sqrt_delta) * inv_2a;
  return std::vector<P>({x0, x1});
}

/* see http://mathworld.wolfram.com/CubicEquation.html */
template <typename P> inline std::vector<P> SolveCubic(P a, P b, P c, P d) {
  P inv_a, b_a, b_a2, c_a, d_a;
  P Q, R, Q3, D, b_a_3;
  P AD, BD;

  P x0, x1, x2;

  if (a == 0) {
    /* solve second order system */
    if (b == 0) {
      /* solve first order system */
      if (c == 0)
        return std::vector<P>();

      x0 = -d / c;
      return std::vector<P>({x0});
    }

    x2 = 0;
    return SolveQuadratic<P>(b, c, d);
  }

  /* calculate the normalized form x^3 + a2 * x^2 + a1 * x + a0 = 0 */
  inv_a = 1.0 / a;
  b_a = inv_a * b;
  b_a2 = b_a * b_a;
  c_a = inv_a * c;
  d_a = inv_a * d;

  /* solve the cubic equation */
  Q = (3 * c_a - b_a2) / 9;
  R = (9 * b_a * c_a - 27 * d_a - 2 * b_a * b_a2) / 54;
  Q3 = Q * Q * Q;
  D = Q3 + R * R;
  b_a_3 = (1.0 / 3.0) * b_a;

  if (Q == 0) {
    if (R == 0) {
      x0 = x1 = x2 = -b_a_3;
      return std::vector<P>({x0, x1, x2});
    } else {
      x0 = pow(2 * R, 1 / 3.0) - b_a_3;
      return std::vector<P>({x0});
    }
  }

  if (D <= 0) {
    /* three real roots */
    P theta = acos(R / sqrt(-Q3));
    P sqrt_Q = sqrt(-Q);
    x0 = 2 * sqrt_Q * cos(theta / 3.0) - b_a_3;
    x1 = 2 * sqrt_Q * cos((theta + 2 * M_PI) / 3.0) - b_a_3;
    x2 = 2 * sqrt_Q * cos((theta + 4 * M_PI) / 3.0) - b_a_3;

    return std::vector<P>({x0, x1, x2});
  }

  /* D > 0, only one real root */
  AD = pow(fabs(R) + sqrt(D), 1.0 / 3.0) * (R > 0 ? 1 : (R < 0 ? -1 : 0));
  BD = (AD == 0) ? 0 : -Q / AD;

  /* calculate the sole real root */
  x0 = AD + BD - b_a_3;

  return std::vector<P>({x0});
}

/* see http://mathworld.wolfram.com/QuarticEquation.html */
template <typename P>
inline std::vector<P> SolveQuartic(P a, P b, P c, P d, P e) {
  P inv_a, b2, bc, b3, b_4;
  P r0;
  int nb_real_roots;
  P R2, R_2, R, inv_R;
  P D2, E2;

  P x0 = 0, x1 = 0, x2 = 0, x3 = 0;

  if (a == 0) {
    x3 = 0;
    return SolveCubic<P>(b, c, d, e);
  }

  /* normalize coefficients */
  inv_a = 1.0 / a;
  b *= inv_a;
  c *= inv_a;
  d *= inv_a;
  e *= inv_a;
  b2 = b * b;
  bc = b * c;
  b3 = b2 * b;

  /* solve resultant cubic */
  auto solution3 =
      SolveCubic<P>(1, -c, d * b - 4 * e, 4 * c * e - d * d - b2 * e);
  int n = (solution3.size() == 0) ? 0 : solution3.size();

  if (n == 0)
    return solution3;

  r0 = solution3[0];
  /* calculate R^2 */
  R2 = 0.25 * b2 - c + r0;
  if (R2 < 0)
    return std::vector<P>();

  R = sqrt(R2);
  inv_R = 1.0 / R;

  nb_real_roots = 0;

  /* calculate D^2 and E^2 */
  if (R < 10E-12) {
    P temp = r0 * r0 - 4 * e;
    if (temp < 0) {
      D2 = E2 = -1;
    } else {
      P sqrt_temp = sqrt(temp);
      D2 = 0.75 * b2 - 2 * c + 2 * sqrt_temp;
      E2 = D2 - 4 * sqrt_temp;
    }
  } else {
    P u = 0.75 * b2 - 2 * c - R2, v = 0.25 * inv_R * (4 * bc - 8 * d - b3);
    D2 = u + v;
    E2 = u - v;
  }

  b_4 = 0.25 * b;
  R_2 = 0.5 * R;
  if (D2 >= 0) {
    P D = sqrt(D2);
    P D_2 = 0.5 * D;
    nb_real_roots = 2;
    x0 = R_2 + D_2 - b_4;
    x1 = x0 - D;
  }

  /* calculate E^2 */
  if (E2 >= 0) {
    P E = sqrt(E2);
    P E_2 = 0.5 * E;
    if (nb_real_roots == 0) {
      x0 = -R_2 + E_2 - b_4;
      x1 = x0 - E;
      nb_real_roots = 2;
    } else {
      x2 = -R_2 + E_2 - b_4;
      x3 = x2 - E;
      nb_real_roots = 4;
    }
  }
  switch (nb_real_roots) {
  // case 0:
  //     return new Tuple<int, double[]>(0, null);
  //     break; // covered by the "default" case
  case 1:
    return std::vector<P>({x0});
    // break;
  case 2:
    return std::vector<P>({x0, x1});
    // break;
  case 3:
    return std::vector<P>({x0, x1, x2});
    // break;
  case 4:
    return std::vector<P>({x0, x1, x2, x3});
    // break;
  default:
    return std::vector<P>(); // just to shut the compiler up....
                             // break;
  }
}

} // end namespace PolySolvers

/// <summary>
/// This function performs two steps of Gauss-Jordan elmination in a 3x3 matrix.
/// NOTE: The function assumes that the matrix provided is RANK-2 and therefore
/// the solution
//        will be the null space (vector). This way we obtain an eigen vector in
//        two steps.
/// </summary>
/// <param name="std::vector<P> flat_mat"> is the coefficient matrix in flat
/// form</param> <returns>std::vector<P> solutions </returns>
template <typename P> std::vector<P> GaussJordan3x3(std::vector<P> flat_mat) {
  // For the first step of the Gauss-Jordan Pivoting, we need the max of the
  // coefficient from list0Pivot
  //
  auto absmax1it = std::max_element(
      flat_mat.begin(), flat_mat.end(),
      [](const int& a, const int& b) { return fabs(a) < fabs(b); });

  P max1 = *absmax1it;

  // Get the index of max (in absolute value)
  int index1 = std::distance(flat_mat.begin(), absmax1it);

  // If max1 = 0, we have a zero matrix and we need to return an empty list
  if (max1 == 0)
    return std::vector<P>();

  // For the second step of the Gauss-Jordan Pivoting, we will require the
  //  maximum of the elements of the resulting matrix.
  P max2; // The value of the absolute max in the next stage of the elimination
          // (only need 2 stages for 3x3 matrices)
  int index2; // the index of the absolute maximum in the next stage of the
              // elimination
  std::vector<P> flat_mat1; // and the next matrix

  // Variable which contain the coordinates of a vector
  P x1 = 0, x2 = 0, x3 = 0;

  P* pX[3]; // using a pointer array to store permutations of x1, x2, x3
  P listFinalCoef[2];

  // Taking cases according to where the maximum lies
  if (index1 == 0) {
    // List of the value after the first step of the Gauss-Jordan Pivoting
    flat_mat1.push_back(flat_mat[4] - ((flat_mat[1] * flat_mat[3]) / max1));
    flat_mat1.push_back(flat_mat[5] - ((flat_mat[2] * flat_mat[3]) / max1));
    flat_mat1.push_back(flat_mat[7] - ((flat_mat[1] * flat_mat[6]) / max1));
    flat_mat1.push_back(flat_mat[8] - ((flat_mat[2] * flat_mat[6]) / max1));

    // Get maximum and minimum elements (in order to get the maximum in absolute
    // value)
    auto absmax2it = std::max_element(
        flat_mat1.begin(), flat_mat1.end(),
        [](const int& a, const int& b) { return fabs(a) < fabs(b); });
    max2 = *absmax2it;

    // NOTE: Now max2it points to the absolute maximum !
    index2 = std::distance(flat_mat1.begin(), absmax2it);

    pX[0] = &x1;
    pX[1] = &x2;
    pX[2] = &x3;

    listFinalCoef[0] = flat_mat[1];
    listFinalCoef[1] = flat_mat[2];

  } else if (index1 == 1) {
    flat_mat1.push_back(flat_mat[3] - ((flat_mat[0] * flat_mat[4]) / max1));
    flat_mat1.push_back(flat_mat[5] - ((flat_mat[2] * flat_mat[4]) / max1));
    flat_mat1.push_back(flat_mat[6] - ((flat_mat[0] * flat_mat[7]) / max1));
    flat_mat1.push_back(flat_mat[8] - ((flat_mat[2] * flat_mat[7]) / max1));

    // Get maximum and minimum elements (in order to get the maximum in absolute
    // value)
    auto absmax2it = std::max_element(
        flat_mat1.begin(), flat_mat1.end(),
        [](const int& a, const int& b) { return fabs(a) < fabs(b); });
    max2 = *absmax2it;

    // NOTE: Now max2it points to the absolute maximum !
    index2 = std::distance(flat_mat1.begin(), absmax2it);

    pX[0] = &x2;
    pX[1] = &x1;
    pX[2] = &x3;

    listFinalCoef[0] = flat_mat[0];
    listFinalCoef[1] = flat_mat[2];

  } else if (index1 == 2) {
    flat_mat1.push_back(flat_mat[3] - ((flat_mat[0] * flat_mat[5]) / max1));
    flat_mat1.push_back(flat_mat[4] - ((flat_mat[1] * flat_mat[5]) / max1));
    flat_mat1.push_back(flat_mat[6] - ((flat_mat[0] * flat_mat[8]) / max1));
    flat_mat1.push_back(flat_mat[7] - ((flat_mat[1] * flat_mat[8]) / max1));

    // Get maximum and minimum elements (in order to get the maximum in absolute
    // value)
    auto absmax2it = std::max_element(
        flat_mat1.begin(), flat_mat1.end(),
        [](const int& a, const int& b) { return fabs(a) < fabs(b); });
    max2 = *absmax2it;

    // NOTE: Now max2it points to the absolute maximum !
    index2 = std::distance(flat_mat1.begin(), absmax2it);

    pX[0] = &x3;
    pX[1] = &x1;
    pX[2] = &x2;

    listFinalCoef[0] = flat_mat[0];
    listFinalCoef[1] = flat_mat[1];

  } else if (index1 == 3) {
    flat_mat1.push_back(flat_mat[1] - ((flat_mat[4] * flat_mat[0]) / max1));
    flat_mat1.push_back(flat_mat[2] - ((flat_mat[0] * flat_mat[5]) / max1));
    flat_mat1.push_back(flat_mat[7] - ((flat_mat[4] * flat_mat[6]) / max1));
    flat_mat1.push_back(flat_mat[8] - ((flat_mat[5] * flat_mat[6]) / max1));

    auto max2it = std::max_element(flat_mat1.begin(), flat_mat1.end());
    auto min2it = std::min_element(flat_mat1.begin(), flat_mat1.end());
    max2 = *max2it;

    // Get maximum and minimum elements (in order to get the maximum in absolute
    // value)
    auto absmax2it = std::max_element(
        flat_mat1.begin(), flat_mat1.end(),
        [](const int& a, const int& b) { return fabs(a) < fabs(b); });
    max2 = *absmax2it;

    // NOTE: Now max2it points to the absolute maximum !
    index2 = std::distance(flat_mat1.begin(), absmax2it);

    pX[0] = &x1;
    pX[1] = &x2;
    pX[2] = &x3;

    listFinalCoef[0] = flat_mat[4];
    listFinalCoef[1] = flat_mat[5];

  } else if (index1 == 4) {
    flat_mat1.push_back(flat_mat[0] - ((flat_mat[1] * flat_mat[3]) / max1));
    flat_mat1.push_back(flat_mat[2] - ((flat_mat[5] * flat_mat[1]) / max1));
    flat_mat1.push_back(flat_mat[6] - ((flat_mat[3] * flat_mat[7]) / max1));
    flat_mat1.push_back(flat_mat[8] - ((flat_mat[5] * flat_mat[7]) / max1));

    // Get maximum and minimum elements (in order to get the maximum in absolute
    // value)
    auto absmax2it = std::max_element(
        flat_mat1.begin(), flat_mat1.end(),
        [](const int& a, const int& b) { return fabs(a) < fabs(b); });
    max2 = *absmax2it;

    // NOTE: Now max2it points to the absolute maximum !
    index2 = std::distance(flat_mat1.begin(), absmax2it);

    pX[0] = &x2;
    pX[1] = &x1;
    pX[2] = &x3;

    listFinalCoef[0] = flat_mat[3];
    listFinalCoef[1] = flat_mat[5];

  } else if (index1 == 5) {
    flat_mat1.push_back(flat_mat[0] - ((flat_mat[2] * flat_mat[3]) / max1));
    flat_mat1.push_back(flat_mat[1] - ((flat_mat[2] * flat_mat[4]) / max1));
    flat_mat1.push_back(flat_mat[6] - ((flat_mat[8] * flat_mat[3]) / max1));
    flat_mat1.push_back(flat_mat[7] - ((flat_mat[8] * flat_mat[4]) / max1));

    // Get maximum and minimum elements (in order to get the maximum in absolute
    // value)
    auto absmax2it = std::max_element(
        flat_mat1.begin(), flat_mat1.end(),
        [](const int& a, const int& b) { return fabs(a) < fabs(b); });
    max2 = *absmax2it;

    // NOTE: Now max2it points to the absolute maximum !
    index2 = std::distance(flat_mat1.begin(), absmax2it);

    pX[0] = &x3;
    pX[1] = &x1;
    pX[2] = &x2;

    listFinalCoef[0] = flat_mat[3];
    listFinalCoef[1] = flat_mat[4];

  } else if (index1 == 6) {
    flat_mat1.push_back(flat_mat[1] - ((flat_mat[0] * flat_mat[7]) / max1));
    flat_mat1.push_back(flat_mat[2] - ((flat_mat[0] * flat_mat[8]) / max1));
    flat_mat1.push_back(flat_mat[4] - ((flat_mat[3] * flat_mat[7]) / max1));
    flat_mat1.push_back(flat_mat[5] - ((flat_mat[3] * flat_mat[8]) / max1));

    // Get maximum and minimum elements (in order to get the maximum in absolute
    // value)
    auto absmax2it = std::max_element(
        flat_mat1.begin(), flat_mat1.end(),
        [](const int& a, const int& b) { return fabs(a) < fabs(b); });
    max2 = *absmax2it;

    // NOTE: Now max2it points to the absolute maximum !
    index2 = std::distance(flat_mat1.begin(), absmax2it);

    pX[0] = &x1;
    pX[1] = &x2;
    pX[2] = &x3;

    listFinalCoef[0] = flat_mat[7];
    listFinalCoef[1] = flat_mat[8];

  } else if (index1 == 7) {
    flat_mat1.push_back(flat_mat[0] - ((flat_mat[1] * flat_mat[6]) / max1));
    flat_mat1.push_back(flat_mat[2] - ((flat_mat[1] * flat_mat[8]) / max1));
    flat_mat1.push_back(flat_mat[3] - ((flat_mat[4] * flat_mat[6]) / max1));
    flat_mat1.push_back(flat_mat[5] - ((flat_mat[4] * flat_mat[8]) / max1));

    // Get maximum and minimum elements (in order to get the maximum in absolute
    // value)
    auto absmax2it = std::max_element(
        flat_mat1.begin(), flat_mat1.end(),
        [](const int& a, const int& b) { return fabs(a) < fabs(b); });
    max2 = *absmax2it;

    // NOTE: Now max2it points to the absolute maximum !
    index2 = std::distance(flat_mat1.begin(), absmax2it);

    pX[0] = &x2;
    pX[1] = &x1;
    pX[2] = &x3;

    listFinalCoef[0] = flat_mat[6];
    listFinalCoef[1] = flat_mat[8];

  } else if (index1 == 8) {
    flat_mat1.push_back(flat_mat[0] - ((flat_mat[2] * flat_mat[6]) / max1));
    flat_mat1.push_back(flat_mat[1] - ((flat_mat[2] * flat_mat[7]) / max1));
    flat_mat1.push_back(flat_mat[3] - ((flat_mat[5] * flat_mat[6]) / max1));
    flat_mat1.push_back(flat_mat[4] - ((flat_mat[5] * flat_mat[7]) / max1));

    // Get maximum and minimum elements (in order to get the maximum in absolute
    // value)
    auto absmax2it = std::max_element(
        flat_mat1.begin(), flat_mat1.end(),
        [](const int& a, const int& b) { return fabs(a) < fabs(b); });
    max2 = *absmax2it;

    // NOTE: Now max2it points to the absolute maximum !
    index2 = std::distance(flat_mat1.begin(), absmax2it);

    pX[0] = &x3;
    pX[1] = &x1;
    pX[2] = &x2;

    listFinalCoef[0] = flat_mat[6];
    listFinalCoef[1] = flat_mat[7];
  }

  if (index2 == 0) {
    *(pX[2]) = 1;
    *(pX[1]) = -flat_mat1[1] / max2;
  } else if (index2 == 1) {
    *(pX[2]) = -flat_mat1[0] / max2;
    *(pX[1]) = 1;
  } else if (index2 == 2) {
    *(pX[2]) = 1;
    *(pX[1]) = -flat_mat1[3] / max2;
  } else if (index2 == 3) {
    *(pX[2]) = -flat_mat1[2] / max2;
    *(pX[1]) = 1;
  }

  *(pX[0]) = -((listFinalCoef[0] * *(pX[1])) / max1) -
             ((listFinalCoef[1] * *(pX[2])) / max1);

  // normalize
  P norm = sqrt(x1 * x1 + x2 * x2 + x3 * x3);

  return std::vector<P>({x1 / norm, x2 / norm, x3 / norm});
}

///
/// This function eliminates the coefficients of the column "col" based on an
/// element at (row, col)
///
///
template <typename P>
std::vector<P> GaussJordanFirstStep(std::vector<P> flat_mat, int row, int col) {

  // the result
  std::vector<P> result;

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {

      if (i != row && j != col) {
        result.push_back(flat_mat[i * 4 + j] -
                         ((flat_mat[row * 4 + j] * flat_mat[i * 4 + col]) /
                          flat_mat[row * 4 + col]));
      }
    }
  }

  return result;
}

///
/// The Gauss-Jordan elimination for Rank-3 4x4 matrices
/// The steps are fixed and lead to a solution up to arbitrary scale.
/// This is how we solve for the eigenvectors of the a 4x4 marix.
///
/// The input vector is a flat version of the 4x4 matrix
///
template <typename P> std::vector<P> GaussJordan4x4(std::vector<P> flat_mat) {
  // Working out the the index of the max coefficient (absolute value).
  auto absmax1it = std::max_element(
      flat_mat.begin(), flat_mat.end(),
      [](const int& a, const int& b) { return fabs(a) < fabs(b); });

  P max1 = *absmax1it;
  int index1 = std::distance(flat_mat.begin(), absmax1it);

  // Return empty list if the maximum is zero (zero matrix)
  if (max1 == 0)
    return std::vector<P>();

  // Create new variable which are use after.
  std::vector<P> flat_mat1;

  P x1 = 0, x2 = 0, x3 = 0, x4 = 0;
  // Cache for the result of 3x3 Gauss Jordan elimination
  std::vector<P> resultGaussJordan3x3;

  // go...
  if (index1 == 0) {
    /// We use the GaussJordan3x3 solver after the first step, which is the
    /// elimination of coeffcieints along column 0.
    flat_mat1 = GaussJordanFirstStep(flat_mat, 0, 0);
    resultGaussJordan3x3 = GaussJordan3x3(flat_mat1);
    x2 = resultGaussJordan3x3[0];
    x3 = resultGaussJordan3x3[1];
    x4 = resultGaussJordan3x3[2];
    x1 = -(1 / flat_mat[0]) *
         (flat_mat[1] * x2 + flat_mat[2] * x3 + flat_mat[3] * x4);
  } else if (index1 == 1) {
    /// We use the GaussJordan3x3 solver after the first step, which is the
    /// elimination of coeffcieints along column 1.
    flat_mat1 = GaussJordanFirstStep(flat_mat, 0, 1);
    resultGaussJordan3x3 = GaussJordan3x3(flat_mat1);
    x1 = resultGaussJordan3x3[0];
    x3 = resultGaussJordan3x3[1];
    x4 = resultGaussJordan3x3[2];
    x2 = -(1 / flat_mat[1]) *
         (flat_mat[0] * x1 + flat_mat[2] * x3 + flat_mat[3] * x4);
  } else if (index1 == 2) {
    flat_mat1 = GaussJordanFirstStep(flat_mat, 0, 2);
    resultGaussJordan3x3 = GaussJordan3x3(flat_mat1);
    x1 = resultGaussJordan3x3[0];
    x2 = resultGaussJordan3x3[1];
    x4 = resultGaussJordan3x3[2];
    x3 = -(1 / flat_mat[2]) *
         (flat_mat[0] * x1 + flat_mat[1] * x2 + flat_mat[3] * x4);
  } else if (index1 == 3) {
    flat_mat1 = GaussJordanFirstStep(flat_mat, 0, 3);
    resultGaussJordan3x3 = GaussJordan3x3(flat_mat1);
    x1 = resultGaussJordan3x3[0];
    x2 = resultGaussJordan3x3[1];
    x3 = resultGaussJordan3x3[2];
    x4 = -(1 / flat_mat[3]) *
         (flat_mat[0] * x1 + flat_mat[1] * x2 + flat_mat[2] * x3);
  } else if (index1 == 4) {
    flat_mat1 = GaussJordanFirstStep(flat_mat, 1, 0);
    resultGaussJordan3x3 = GaussJordan3x3(flat_mat1);
    x2 = resultGaussJordan3x3[0];
    x3 = resultGaussJordan3x3[1];
    x4 = resultGaussJordan3x3[2];
    x1 = -(1 / flat_mat[4]) *
         (flat_mat[5] * x2 + flat_mat[6] * x3 + flat_mat[7] * x4);
  } else if (index1 == 5) {
    flat_mat1 = GaussJordanFirstStep(flat_mat, 1, 1);
    resultGaussJordan3x3 = GaussJordan3x3(flat_mat1);
    x1 = resultGaussJordan3x3[0];
    x3 = resultGaussJordan3x3[1];
    x4 = resultGaussJordan3x3[2];
    x2 = -(1 / flat_mat[5]) *
         (flat_mat[4] * x1 + flat_mat[6] * x3 + flat_mat[7] * x4);
  } else if (index1 == 6) {
    flat_mat1 = GaussJordanFirstStep(flat_mat, 1, 2);
    resultGaussJordan3x3 = GaussJordan3x3(flat_mat1);
    x1 = resultGaussJordan3x3[0];
    x2 = resultGaussJordan3x3[1];
    x4 = resultGaussJordan3x3[2];
    x3 = -(1 / flat_mat[6]) *
         (flat_mat[4] * x1 + flat_mat[5] * x2 + flat_mat[7] * x4);
  } else if (index1 == 7) {
    flat_mat1 = GaussJordanFirstStep(flat_mat, 1, 3);
    resultGaussJordan3x3 = GaussJordan3x3(flat_mat1);
    x1 = resultGaussJordan3x3[0];
    x2 = resultGaussJordan3x3[1];
    x3 = resultGaussJordan3x3[2];
    x4 = -(1 / flat_mat[7]) *
         (flat_mat[4] * x1 + flat_mat[5] * x2 + flat_mat[6] * x3);
  } else if (index1 == 8) {
    flat_mat1 = GaussJordanFirstStep(flat_mat, 2, 0);
    resultGaussJordan3x3 = GaussJordan3x3(flat_mat1);
    x2 = resultGaussJordan3x3[0];
    x3 = resultGaussJordan3x3[1];
    x4 = resultGaussJordan3x3[2];
    x1 = -(1 / flat_mat[8]) *
         (flat_mat[9] * x2 + flat_mat[10] * x3 + flat_mat[11] * x4);
  } else if (index1 == 9) {
    flat_mat1 = GaussJordanFirstStep(flat_mat, 2, 1);
    resultGaussJordan3x3 = GaussJordan3x3(flat_mat1);
    x1 = resultGaussJordan3x3[0];
    x3 = resultGaussJordan3x3[1];
    x4 = resultGaussJordan3x3[2];
    x2 = -(1 / flat_mat[9]) *
         (flat_mat[8] * x1 + flat_mat[10] * x3 + flat_mat[11] * x4);
  } else if (index1 == 10) {
    flat_mat1 = GaussJordanFirstStep(flat_mat, 2, 2);
    resultGaussJordan3x3 = GaussJordan3x3(flat_mat1);
    x1 = resultGaussJordan3x3[0];
    x2 = resultGaussJordan3x3[1];
    x4 = resultGaussJordan3x3[2];
    x3 = -(1 / flat_mat[10]) *
         (flat_mat[8] * x1 + flat_mat[9] * x2 + flat_mat[11] * x4);
  } else if (index1 == 11) {
    flat_mat1 = GaussJordanFirstStep(flat_mat, 2, 3);
    resultGaussJordan3x3 = GaussJordan3x3(flat_mat1);
    x1 = resultGaussJordan3x3[0];
    x2 = resultGaussJordan3x3[1];
    x3 = resultGaussJordan3x3[2];
    x4 = -(1 / flat_mat[11]) *
         (flat_mat[8] * x1 + flat_mat[9] * x2 + flat_mat[10] * x3);
  } else if (index1 == 12) {
    flat_mat1 = GaussJordanFirstStep(flat_mat, 3, 0);
    resultGaussJordan3x3 = GaussJordan3x3(flat_mat1);
    x2 = resultGaussJordan3x3[0];
    x3 = resultGaussJordan3x3[1];
    x4 = resultGaussJordan3x3[2];
    x1 = -(1 / flat_mat[12]) *
         (flat_mat[13] * x2 + flat_mat[14] * x3 + flat_mat[15] * x4);

  } else if (index1 == 13) {
    flat_mat1 = GaussJordanFirstStep(flat_mat, 3, 1);
    resultGaussJordan3x3 = GaussJordan3x3(flat_mat1);
    x1 = resultGaussJordan3x3[0];
    x3 = resultGaussJordan3x3[1];
    x4 = resultGaussJordan3x3[2];
    x2 = -(1 / flat_mat[13]) *
         (flat_mat[12] * x1 + flat_mat[14] * x3 + flat_mat[15] * x4);
  } else if (index1 == 14) {
    flat_mat1 = GaussJordanFirstStep(flat_mat, 3, 2);
    resultGaussJordan3x3 = GaussJordan3x3(flat_mat1);
    x1 = resultGaussJordan3x3[0];
    x2 = resultGaussJordan3x3[1];
    x4 = resultGaussJordan3x3[2];
    x3 = -(1 / flat_mat[14]) *
         (flat_mat[12] * x1 + flat_mat[13] * x2 + flat_mat[15] * x4);
  } else if (index1 == 15) {
    flat_mat1 = GaussJordanFirstStep(flat_mat, 3, 3);
    resultGaussJordan3x3 = GaussJordan3x3(flat_mat1);
    x1 = resultGaussJordan3x3[0];
    x2 = resultGaussJordan3x3[1];
    x3 = resultGaussJordan3x3[2];
    x4 = -(1 / flat_mat[15]) *
         (flat_mat[12] * x1 + flat_mat[13] * x2 + flat_mat[14] * x3);
  }

  // normalize the solution
  P norm = sqrt(x1 * x1 + x2 * x2 + x3 * x3 + x4 * x4);

  return std::vector<P>({x1 / norm, x2 / norm, x3 / norm, x4 / norm});
}

///
/// Return the list of the eigenvalue of a 4x4 matrix.
/// Parameter is he argumen is a 1D array representing the 4x4 matrix
/// T
///
template <typename P> std::vector<P> EigenValues4x4(P* array) {

  P a11 = array[0 * 4 + 0], a12 = array[0 * 4 + 1], a13 = array[0 * 4 + 2],
    a14 = array[0 * 4 + 3];
  P a21 = array[1 * 4 + 0], a22 = array[1 * 4 + 1], a23 = array[1 * 4 + 2],
    a24 = array[1 * 4 + 3];
  P a31 = array[2 * 4 + 0], a32 = array[2 * 4 + 1], a33 = array[2 * 4 + 2],
    a34 = array[2 * 4 + 3];
  P a41 = array[3 * 4 + 0], a42 = array[3 * 4 + 1], a43 = array[3 * 4 + 2],
    a44 = array[3 * 4 + 3];

  // 1. Obtaining the coefficients of the characteristics polynomial of A11 -
  // λ*I where A11 is the (1, 1) submatrix of A: A11 = [a22-λ a23 a24;
  //        a32 a33-λ a34;
  //        a42 a43 a44 - λ]

  // The coefficients of the cubic polynomial det(A11-λI) are:
  P C3 = -1, C2 = a22 + a33 + a44;
  P C1 = a42 * a24 + a32 * a23 + a34 * a43 - a22 * a33 - a22 * a44 - a33 * a44;
  P C0 = a22 * a33 * a44 + a23 * a42 * a34 + a24 * a32 * a43 - a22 * a34 * a43 -
         a23 * a32 * a44 - a24 * a42 * a33;

  // 2. Now multiplying with a11 to get a quartic polynomial with coeffcients
  // W4, W3, W2, W1, W0 as follows:
  P W4 = -C3, W3 = a11 * C3 - C2, W2 = a11 * C2 - C1, W1 = a11 * C1 - C0,
    W0 = a11 * C0;

  // 4. Now we obtain the coefficients of the 3 quadratics (from the algebraic
  // complements A12, A13, A14) as follows: a. (A-λI)12 (-)
  P Q1_2 = -a12 * a21,
    Q1_1 = -a12 * (-a21 * (a33 + a44) + a23 * a31 + a24 * a41),
    Q1_0 =
        -a12 * (a24 * (a31 * a43 - a41 * a33) - a23 * (a31 * a44 - a41 * a34) +
                a21 * (a33 * a44 -
                       a43 * a34)); // good all being well...
                                    // multiplying with -a12
                                    // P1_2 *= -a12; P1_1 *= -a12; P1_0 *= -a12;

  // b. (A-λ)13 (+)
  P Q2_2 = a13 * -a31,
    Q2_1 = a13 * ((a31 * a44 - a41 * a34 + a22 * a31) - a21 * a32),
    Q2_0 =
        a13 * (a24 * (a31 * a42 - a41 * a32) + a22 * (a41 * a34 - a31 * a44) +
               a21 * (a32 * a44 - a42 * a34));

  // c. (A-λ)14 (-)
  P Q3_2 = -a14 * a41,
    Q3_1 = -a14 * ((a31 * a43 - a41 * a33 - a22 * a41) + a21 * a42),
    Q3_0 =
        -a14 * (a23 * (a31 * a42 - a41 * a32) - a22 * (a31 * a43 - a41 * a33) +
                a21 * (a32 * a43 - a42 * a33));

  // 5. Final coefficients
  P A0 = Q3_0 + Q2_0 + Q1_0 + W0, A1 = Q3_1 + Q2_1 + Q1_1 + W1,
    A2 = Q3_2 + Q2_2 + Q1_2 + W2, A3 = W3, A4 = W4;

  std::vector<P> solution = PolySolvers::SolveQuartic(A4, A3, A2, A1, A0);

  // Put the solutions in ascending order
  std::sort(solution.begin(), solution.end());

  return solution;
}

///
/// Return the list of the eigenvalues of a 3x3 matrix.
/// Parameter is he argumen is a 1D array representing the 4x4 matrix
/// T
///
template <typename P> std::vector<P> EigenValues3x3(const P* array) {

  P coef1 = -1;
  P coef2 = (array[0 * 3 + 0] + array[1 * 3 + 1] + array[2 * 3 + 2]);
  P coef3 = (array[2 * 3 + 0] * array[0 * 3 + 2]) +
            (array[1 * 3 + 0] * array[0 * 3 + 1]) +
            (array[1 * 3 + 2] * array[2 * 3 + 1]) -
            (array[0 * 3 + 0] * array[1 * 3 + 1]) -
            (array[0 * 3 + 0] * array[2 * 3 + 2]) -
            (array[1 * 3 + 1] * array[2 * 3 + 2]);

  P coef4 = (array[0 * 3 + 0] * array[1 * 3 + 1] * array[2 * 3 + 2]) +
            (array[0 * 3 + 1] * array[2 * 3 + 0] * array[1 * 3 + 2]) +
            (array[0 * 3 + 2] * array[1 * 3 + 0] * array[2 * 3 + 1]) -
            (array[0 * 3 + 0] * array[1 * 3 + 2] * array[2 * 3 + 1]) -
            (array[0 * 3 + 1] * array[1 * 3 + 0] * array[2 * 3 + 2]) -
            (array[0 * 3 + 2] * array[2 * 3 + 0] * array[1 * 3 + 1]);

  std::vector<P> solution = PolySolvers::SolveCubic(coef1, coef2, coef3, coef4);

  std::sort(solution.begin(), solution.end());

  // for (int i = 0; i < solution.size(); i++)
  //   std::cout << "eigenvalue "<<i<<" : " << solution[i] << std::endl;

  return solution;
}

// Get the eigenvector of the largest eigenvalue of a 3x3 matrix.
// NOTE: This function will return the zero vector even if no eigenvalues exist
template <typename P> std::vector<P> PrincipalEigenvector4x4(P* M) {
  // First obtain the eigenvalues of M
  auto eigenvalues = EigenValues4x4(M);
  if (eigenvalues.size() == 0)
    return std::vector<P>({0, 0, 0, 0});
  // auto absmaxit = std::max_element( eigenvalues.begin(),
  //			   eigenvalues.end() ,
  //			   [](const int& a, const int& b) { return fabs(a) <
  // fabs(b);}
  //			  );
  // P lambda = *absmaxit;
  P lambda = eigenvalues[eigenvalues.size() -
                         1]; // returning the largest eigenvalue (instead or
                             // trhe largest in absolute value)
  if (lambda == 0)
    return std::vector<P>({0, 0, 0, 0});
  // Now obtaining a vector containing the M-lambda * eye(3)
  std::vector<P> A;
  A.push_back(M[0] - lambda);
  A.push_back(M[1]);
  A.push_back(M[2]);
  A.push_back(M[3]);
  A.push_back(M[4]);
  A.push_back(M[5] - lambda);
  A.push_back(M[6]);
  A.push_back(M[7]);
  A.push_back(M[8]);
  A.push_back(M[9]);
  A.push_back(M[10] - lambda);
  A.push_back(M[11]);
  A.push_back(M[12]);
  A.push_back(M[13]);
  A.push_back(M[14]);
  A.push_back(M[15] - lambda);

  // Now get the eigenvector using the Gauss-jordan steps
  return GaussJordan4x4(A);
}

// Get the eigenvector of the largest eigenvalue of a 4x4 matrix.
// NOTE: This function will return the zero vector even if no eigenvalues exist
template <typename P> std::vector<P> PrincipalEigenvector3x3(P* M) {
  // First obtain the eigenvalues of M
  auto eigenvalues = EigenValues3x3(M);
  if (eigenvalues.size() == 0)
    return std::vector<P>({0, 0, 0});
  // auto maxit = std::max_element( eigenvalues.begin(),
  //			   eigenvalues.end() ,
  //			   [](const int& a, const int& b) { return fabs(a) <
  // fabs(b); }
  //			 );

  // P lambda = *maxit;
  P lambda = eigenvalues[eigenvalues.size() - 1]; // get the largest eigenvalue
  // if there are only zero eigenvalues, return the zero vector
  if (lambda == 0)
    return std::vector<P>({0, 0, 0}); // extra check

  // Now obtaining a vector containing the M-lambda * eye(3)
  std::vector<P> A;
  A.push_back(M[0] - lambda);
  A.push_back(M[1]);
  A.push_back(M[2]);
  A.push_back(M[3]);
  A.push_back(M[4] - lambda);
  A.push_back(M[5]);
  A.push_back(M[6]);
  A.push_back(M[7]);
  A.push_back(M[8] - lambda);

  // Now get the eigenvector using the Gauss-jordan steps
  return GaussJordan3x3(A);
}

// 3x3 Matrix eigen decomposition
template <typename P>
std::pair<std::vector<P>, std::vector<std::vector<P>>> EigenDecompose3x3(P* M) {
  // First obtain the eigenvalues of M
  auto eigenvalues = EigenValues3x3(M);
  // vector of eigenvectors
  std::vector<std::vector<P>> eigenvectors;
  // return an empty eigenvalue list and a single eigenvector withg zeros in it
  if (eigenvalues.size() == 0)
    return std::pair<std::vector<P>, std::vector<std::vector<P>>>(eigenvalues,
                                                                  eigenvectors);

  std::vector<P> A({M[0], M[1], M[2], M[3], M[4], M[5], M[6], M[7], M[8]});

  for (int i = 0; i < eigenvalues.size(); i++) {
    // Now subtract the eigenvalue from the diagonal
    A[0] -= eigenvalues[i];
    A[4] -= eigenvalues[i];
    A[8] -= eigenvalues[i];

    // compute and add the eigen vector to the list
    eigenvectors.push_back(GaussJordan3x3(A));

    // add the eigenvalue back to the diagonal in order to undo the change in
    // the elements of A
    A[0] += eigenvalues[i];
    A[4] += eigenvalues[i];
    A[8] += eigenvalues[i];
  }

  // return the decomposition (in ascending eigenvalue order)
  return std::pair<std::vector<P>, std::vector<std::vector<P>>>(eigenvalues,
                                                                eigenvectors);
}

// 4x4 Matrix eigen decomposition
template <typename P>
std::pair<std::vector<P>, std::vector<std::vector<P>>> EigenDecompose4x4(P* M) {
  // First obtain the eigenvalues of M
  auto eigenvalues = EigenValues4x4(M);
  // vector of eigenvectors
  std::vector<std::vector<P>> eigenvectors;

  // return an empty eigenvalue list and a single eigenvector withg zeros in it
  if (eigenvalues.size() == 0)
    return std::pair<std::vector<P>, std::vector<std::vector<P>>>(eigenvalues,
                                                                  eigenvectors);

  std::vector<P> A({M[0], M[1], M[2], M[3], M[4], M[5], M[6], M[7], M[8], M[9],
                    M[10], M[11], M[12], M[13], M[14], M[15]});

  for (int i = 0; i < eigenvalues.size(); i++) {
    // Now subtract the eigenvalue from the diagonal
    A[0] -= eigenvalues[i];
    A[5] -= eigenvalues[i];
    A[10] -= eigenvalues[i];
    A[15] -= eigenvalues[i];

    // compute and add the eigen vector to the list
    eigenvectors.push_back(GaussJordan4x4(A));

    // add the eigenvalue back to the diagonal in order to undo the change in
    // the elements of A
    A[0] += eigenvalues[i];
    A[5] += eigenvalues[i];
    A[10] += eigenvalues[i];
    A[15] += eigenvalues[i];
  }

  // return the decomposition (in ascending eigenvalue order)
  return std::pair<std::vector<P>, std::vector<std::vector<P>>>(eigenvalues,
                                                                eigenvectors);
}

std::pair<rmatrix4x4, rdiagonal4x4> self_adjoint_evd(rmatrix4x4 A) {
  rmatrix4x4 rvectors; // TODO: vectors result in nan because max2 is zero
  rdiagonal4x4 rvalues;
  auto [values, vectors] = EigenDecompose4x4(A.data());
  for (int i = 0; i < values.size(); ++i) {
    for (int j = 0; j < vectors[i].size(); ++j) {
      rvectors[j * 4 + i] = vectors[j][i];
    }
    rvalues[i] = values[i];
  }
  return {rvectors, rvalues};
}
} // namespace mqt::ir::opt

#endif

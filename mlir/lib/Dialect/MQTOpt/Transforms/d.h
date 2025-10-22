#pragma once

#include "Helpers.h"

#include <armadillo>

namespace mqt::ir::opt {

auto self_adjoint_evd(rmatrix4x4 A) {
  arma::Mat<fp> a(4, 4, arma::fill::scalar_holder<fp>(0.0));
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      a.at(j, i) = A[j * 4 + i];
    }
  }
  arma::Mat<fp> vecs;
  arma::Col<fp> vals;
  arma::eig_sym(vals, vecs, a, "std");
  rmatrix4x4 rvecs;
  rdiagonal4x4 rvals;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      rvecs[j * 4 + i] = vecs.at(j, i);
    }
    rvals[i] = vals.at(i);
  }
  std::cerr << "========\n" << vecs << "========\n" << std::endl;
  std::cerr << "========\n" << vals << "========\n" << std::endl;
  return std::make_pair(rvecs, rvals);
}
} // namespace mqt::ir::opt

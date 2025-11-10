#pragma once

#include "Helpers.h"
#include "eigen3/Eigen/Eigen"

namespace mqt::ir::opt {

auto self_adjoint_evd(rmatrix4x4 A) {
  Eigen::Matrix4d a;
  Eigen::SelfAdjointEigenSolver<decltype(a)> s;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      a(j, i) = A[j * 4 + i];
    }
  }
  std::cerr << "=EigIN==\n" << a << "\n========\n" << std::endl;
  s.compute(a); // TODO: computeDirect is faster
  auto vecs = s.eigenvectors();
  auto vals = s.eigenvalues();
  rmatrix4x4 rvecs;
  rdiagonal4x4 rvals;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      rvecs[j * 4 + i] = vecs(j, i);
    }
    rvals[i] = vals(i);
  }
  std::cerr << "=Eigen==\n" << vecs << "\n========\n" << std::endl;
  std::cerr << "=Eigen==\n" << vals << "\n========\n" << std::endl;
  return std::make_pair(rvecs, rvals);
}
} // namespace mqt::ir::opt

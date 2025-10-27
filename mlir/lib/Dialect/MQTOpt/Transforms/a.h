#pragma once

#include "Helpers.h"

namespace mqt::ir::opt {

// Function to compute the norm (Frobenius norm) of a matrix
double norm(const std::array<double, 16>& A) {
    double sum = 0.0;
    for (size_t i = 0; i < 16; ++i) {
        sum += A[i] * A[i];
    }
    return sqrt(sum);
}

// Function to perform a Jacobi rotation
void jacobi_rotate(std::array<double, 16>& A, std::array<double, 16>& V, int p, int q) {
    // Compute the Jacobi rotation matrix
    double theta = 0.5 * atan2(2 * A[p * 4 + q], A[q * 4 + q] - A[p * 4 + p]);
    double c = cos(theta);
    double s = sin(theta);

    // Apply the rotation to matrix A
    for (int i = 0; i < 4; ++i) {
        double ap = A[i * 4 + p];
        double aq = A[i * 4 + q];
        A[i * 4 + p] = c * ap - s * aq;
        A[i * 4 + q] = s * ap + c * aq;
    }

    // Apply the rotation to matrix V (eigenvectors)
    for (int i = 0; i < 4; ++i) {
        double vi_p = V[i * 4 + p];
        double vi_q = V[i * 4 + q];
        V[i * 4 + p] = c * vi_p - s * vi_q;
        V[i * 4 + q] = s * vi_p + c * vi_q;
    }
}

// Function to perform the Jacobi method for eigenvalue decomposition
void jacobi_eigen_decomposition(std::array<double, 16>& A, std::array<double, 16>& V, std::array<double, 4>& eigenvalues, double tolerance = 1e-9, int max_iterations = 1000) {
    // Initialize the eigenvector matrix V to identity
    V.fill(0.0);
    for (int i = 0; i < 4; ++i) {
        V[i * 4 + i] = 1.0;
    }

    int iterations = 0;
    double off_diagonal_norm = norm(A);

    // Jacobi rotation iterations
    while (off_diagonal_norm > tolerance && iterations < max_iterations) {
        // Find the largest off-diagonal element
        double max_off_diag = 0.0;
        int p = 0, q = 0;
        for (int i = 0; i < 3; ++i) {
            for (int j = i + 1; j < 4; ++j) {
                double abs_val = fabs(A[i * 4 + j]);
                if (abs_val > max_off_diag) {
                    max_off_diag = abs_val;
                    p = i;
                    q = j;
                }
            }
        }

        // Perform a Jacobi rotation if necessary
        if (max_off_diag > tolerance) {
            jacobi_rotate(A, V, p, q);
        }

        // Update the off-diagonal norm
        off_diagonal_norm = norm(A);
        ++iterations;
    }

    // Extract the eigenvalues from the diagonal of A
    for (int i = 0; i < 4; ++i) {
        eigenvalues[i] = A[i * 4 + i];
    }
}
}

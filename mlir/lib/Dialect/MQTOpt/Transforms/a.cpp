#include <array>
#include <complex>
#include <iomanip>
#include <iostream>
#include <vector>

using fp = long double;
using qfp = std::complex<fp>;
using diagonal4x4 = std::array<qfp, 4>;
using vector2d = std::vector<qfp>;
using matrix2x2 = std::array<qfp, 4>;
using matrix4x4 = std::array<qfp, 16>;

using logical = bool;
using integer = int;
using doublecomplex = qfp;

matrix4x4 zgemm2(matrix4x4 a, matrix4x4 b) {
  matrix4x4 c__;
  // a_dim1 = *lda;
  // a_offset = 1 + a_dim1;
  // a -= a_offset;
  // b_dim1 = *ldb;
  // b_offset = 1 + b_dim1;
  // b -= b_offset;
  // c_dim1 = *ldc;
  // c_offset = 1 + c_dim1;
  // c__ -= c_offset;

  int i__1 = 4;
  int i__3{};
  qfp z__1;
  qfp z__2;
  qfp temp;
  qfp alpha {1.0, 0.0};
  for (auto j = 0; j < i__1; ++j) {
    for (auto i__ = 0; i__ < 4; ++i__) {
      auto i__3 = i__ + j * 4;
      c__[i__3].real(0.), c__[i__3].imag(0.);
    }
    int i__2 = 4;
    for (auto l = 0; l < i__2; ++l) {
      i__3 = l + j * 4;
      if (b[i__3].real() != 0. || b[i__3].imag() != 0.) {
        i__3 = l + j * 4;
        z__1.real(alpha.real() * b[i__3].real() -
                  alpha.imag() * b[i__3].imag()),
            z__1.imag(alpha.real() * b[i__3].imag() +
                      alpha.imag() * b[i__3].real());
        temp.real(z__1.real()), temp.imag(z__1.imag());
        i__3 = 4;
        for (auto i__ = 0; i__ < i__3; ++i__) {
          auto i__4 = i__ + j * 4;
          auto i__5 = i__ + j * 4;
          auto i__6 = i__ + l * 4;
          z__2.real(temp.real() * a[i__6].real() -
                    temp.imag() * a[i__6].imag()),
              z__2.imag(temp.real() * a[i__6].imag() +
                        temp.imag() * a[i__6].real());
          z__1.real(c__[i__5].real() + z__2.real()),
              z__1.imag(c__[i__5].imag() + z__2.imag());
          c__[i__4].real(z__1.real()), c__[i__4].imag(z__1.imag());
          /* L70: */
        }
      }
      /* L80: */
    }
    /* L90: */
  }
  return c__;
}

int zgemm_(char* transa, char* transb, integer* m, integer* n, integer* k,
           doublecomplex* alpha, matrix4x4 a, integer* lda,
           matrix4x4 b, integer* ldb, doublecomplex* beta,
           matrix4x4& c__, integer* ldc);
matrix4x4 zgemm(matrix4x4 a, matrix4x4 b) {
  qfp alpha{1.0, 0.0};
  qfp beta{1.0, 0.0};
  int dimension = 4;
  matrix4x4 result{};
  // zgemm_("n", "n", &dimension, &dimension, &dimension, &alpha, a.data(),
  //        &dimension, b.data(), &dimension, &beta, result.data(), &dimension);
  zgemm_("N", "N", &dimension, &dimension, &dimension, &alpha, a,
         &dimension, b, &dimension, &beta, result, &dimension);
  return result;
}

void d_cnjg(doublecomplex* r, doublecomplex* z) { *r = std::conj(*z); }

/* Subroutine */ int zgemm_(char* transa, char* transb, integer* m, integer* n,
                            integer* k, doublecomplex* alpha, matrix4x4 a,
                            integer* lda, matrix4x4 b, integer* ldb,
                            doublecomplex* beta, matrix4x4& c__,
                            integer* ldc) {
  /* System generated locals */
  integer a_dim1, a_offset, b_dim1, b_offset, c_dim1, c_offset, i__1, i__2,
      i__3, i__4, i__5, i__6;
  doublecomplex z__1, z__2, z__3, z__4;

  /* Local variables */
  static integer i__, j, l, info;
  static logical nota, notb;
  static doublecomplex temp;
  static logical conja, conjb;
  static integer nrowa, nrowb;

  /*
      Purpose
      =======

      ZGEMM  performs one of the matrix-matrix operations

         C := alpha*op( A )*op( B ) + beta*C,

      where  op( X ) is one of

         op( X ) = X   or   op( X ) = X'   or   op( X ) = conjg( X' ),

      alpha and beta are scalars, and A, B and C are matrices, with op( A )
      an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.

      Arguments
      ==========

      TRANSA - CHARACTER*1.
               On entry, TRANSA specifies the form of op( A ) to be used in
               the matrix multiplication as follows:

                  TRANSA = 'N' or 'n',  op( A ) = A.

                  TRANSA = 'T' or 't',  op( A ) = A'.

                  TRANSA = 'C' or 'c',  op( A ) = conjg( A' ).

               Unchanged on exit.

      TRANSB - CHARACTER*1.
               On entry, TRANSB specifies the form of op( B ) to be used in
               the matrix multiplication as follows:

                  TRANSB = 'N' or 'n',  op( B ) = B.

                  TRANSB = 'T' or 't',  op( B ) = B'.

                  TRANSB = 'C' or 'c',  op( B ) = conjg( B' ).

               Unchanged on exit.

      M      - INTEGER.
               On entry,  M  specifies  the number  of rows  of the  matrix
               op( A )  and of the  matrix  C.  M  must  be at least  zero.
               Unchanged on exit.

      N      - INTEGER.
               On entry,  N  specifies the number  of columns of the matrix
               op( B ) and the number of columns of the matrix C. N must be
               at least zero.
               Unchanged on exit.

      K      - INTEGER.
               On entry,  K  specifies  the number of columns of the matrix
               op( A ) and the number of rows of the matrix op( B ). K must
               be at least  zero.
               Unchanged on exit.

      ALPHA  - COMPLEX*16      .
               On entry, ALPHA specifies the scalar alpha.
               Unchanged on exit.

      A      - COMPLEX*16       array of DIMENSION ( LDA, ka ), where ka is
               k  when  TRANSA = 'N' or 'n',  and is  m  otherwise.
               Before entry with  TRANSA = 'N' or 'n',  the leading  m by k
               part of the array  A  must contain the matrix  A,  otherwise
               the leading  k by m  part of the array  A  must contain  the
               matrix A.
               Unchanged on exit.

      LDA    - INTEGER.
               On entry, LDA specifies the first dimension of A as declared
               in the calling (sub) program. When  TRANSA = 'N' or 'n' then
               LDA must be at least  max( 1, m ), otherwise  LDA must be at
               least  max( 1, k ).
               Unchanged on exit.

      B      - COMPLEX*16       array of DIMENSION ( LDB, kb ), where kb is
               n  when  TRANSB = 'N' or 'n',  and is  k  otherwise.
               Before entry with  TRANSB = 'N' or 'n',  the leading  k by n
               part of the array  B  must contain the matrix  B,  otherwise
               the leading  n by k  part of the array  B  must contain  the
               matrix B.
               Unchanged on exit.

      LDB    - INTEGER.
               On entry, LDB specifies the first dimension of B as declared
               in the calling (sub) program. When  TRANSB = 'N' or 'n' then
               LDB must be at least  max( 1, k ), otherwise  LDB must be at
               least  max( 1, n ).
               Unchanged on exit.

      BETA   - COMPLEX*16      .
               On entry,  BETA  specifies the scalar  beta.  When  BETA  is
               supplied as zero then C need not be set on input.
               Unchanged on exit.

      C      - COMPLEX*16       array of DIMENSION ( LDC, n ).
               Before entry, the leading  m by n  part of the array  C must
               contain the matrix  C,  except when  beta  is zero, in which
               case C need not be set on entry.
               On exit, the array  C  is overwritten by the  m by n  matrix
               ( alpha*op( A )*op( B ) + beta*C ).

      LDC    - INTEGER.
               On entry, LDC specifies the first dimension of C as declared
               in  the  calling  (sub)  program.   LDC  must  be  at  least
               max( 1, m ).
               Unchanged on exit.

      Further Details
      ===============

      Level 3 Blas routine.

      -- Written on 8-February-1989.
         Jack Dongarra, Argonne National Laboratory.
         Iain Duff, AERE Harwell.
         Jeremy Du Croz, Numerical Algorithms Group Ltd.
         Sven Hammarling, Numerical Algorithms Group Ltd.

      =====================================================================


         Set  NOTA  and  NOTB  as  true if  A  and  B  respectively are not
         conjugated or transposed, set  CONJA and CONJB  as true if  A  and
         B  respectively are to be  transposed but  not conjugated  and set
         NROWA and  NROWB  as the number of rows and  columns  of  A
         and the number of rows of  B  respectively.
  */

  auto lsame_ = [](char* a, char* b) {
    return std::string{a} == std::string{b};
  };

  /* Parameter adjustments */
  a_dim1 = *lda;
  a_offset = 1 + a_dim1;
  // a -= a_offset;
  b_dim1 = *ldb;
  b_offset = 1 + b_dim1;
  // b -= b_offset;
  c_dim1 = *ldc;
  c_offset = 1 + c_dim1;
  // c__ -= c_offset;

  /* Function Body */
  nota = lsame_(transa, "N");
  notb = lsame_(transb, "N");
  conja = lsame_(transa, "C");
  conjb = lsame_(transb, "C");
  if (nota) {
    nrowa = *m;
  } else {
    nrowa = *k;
  }
  if (notb) {
    nrowb = *k;
  } else {
    nrowb = *n;
  }

  /*     Test the input parameters. */

  info = 0;
  if (!nota && !conja && !lsame_(transa, "T")) {
    info = 1;
  } else if (!notb && !conjb && !lsame_(transb, "T")) {
    info = 2;
  } else if (*m < 0) {
    info = 3;
  } else if (*n < 0) {
    info = 4;
  } else if (*k < 0) {
    info = 5;
  } else if (*lda < std::max(1, nrowa)) {
    info = 8;
  } else if (*ldb < std::max(1, nrowb)) {
    info = 10;
  } else if (*ldc < std::max(1, *m)) {
    info = 13;
  }
  if (info != 0) {
    // xerbla_("ZGEMM ", &info);
    return 0;
  }

  /*     Quick return if possible. */

  if (*m == 0 || *n == 0 ||
      (alpha->real() == 0. && alpha->imag() == 0. || *k == 0) &&
          (beta->real() == 1. && beta->imag() == 0.)) {
    return 0;
  }

  /*     And when  alpha.eq.zero. */

  if (alpha->real() == 0. && alpha->imag() == 0.) {
    if (beta->real() == 0. && beta->imag() == 0.) {
      i__1 = *n;
      for (j = 1; j <= i__1; ++j) {
        i__2 = *m;
        for (i__ = 1; i__ <= i__2; ++i__) {
          i__3 = i__ + j * c_dim1;
          c__[i__3].real(0.), c__[i__3].imag(0.);
          /* L10: */
        }
        /* L20: */
      }
    } else {
      i__1 = *n;
      for (j = 1; j <= i__1; ++j) {
        i__2 = *m;
        for (i__ = 1; i__ <= i__2; ++i__) {
          i__3 = i__ + j * c_dim1;
          i__4 = i__ + j * c_dim1;
          z__1.real(beta->real() * c__[i__4].real() -
                    beta->imag() * c__[i__4].imag()),
              z__1.imag(beta->real() * c__[i__4].imag() +
                        beta->imag() * c__[i__4].real());
          c__[i__3].real(z__1.real()), c__[i__3].imag(z__1.imag());
          /* L30: */
        }
        /* L40: */
      }
    }
    return 0;
  }

  /*     Start the operations. */

  if (notb) {
    if (nota) {

      /*           Form  C := alpha*A*B + beta*C. */

      i__1 = *n;
      for (j = 1; j <= i__1; ++j) {
        if (beta->real() == 0. && beta->imag() == 0.) {
          i__2 = *m;
          for (i__ = 1; i__ <= i__2; ++i__) {
            i__3 = i__ + j * c_dim1;
            c__[i__3 - c_offset].real(0.), c__[i__3 - c_offset].imag(0.);
            /* L50: */
          }
        } else if (beta->real() != 1. || beta->imag() != 0.) {
          i__2 = *m;
          for (i__ = 1; i__ <= i__2; ++i__) {
            i__3 = i__ + j * c_dim1;
            i__4 = i__ + j * c_dim1;
            z__1.real(beta->real() * c__[i__4 - c_offset].real() -
                      beta->imag() * c__[i__4 - c_offset].imag()),
                z__1.imag(beta->real() * c__[i__4 - c_offset].imag() +
                          beta->imag() * c__[i__4 - c_offset].real());
            c__[i__3 - c_offset].real(z__1.real()), c__[i__3 - c_offset].imag(z__1.imag());
            /* L60: */
          }
        }
        i__2 = *k;
        for (l = 1; l <= i__2; ++l) {
          i__3 = l + j * b_dim1;
          if (b[i__3 - b_offset].real() != 0. || b[i__3 - b_offset].imag() != 0.) {
            i__3 = l + j * b_dim1;
            z__1.real(alpha->real() * b[i__3 - b_offset].real() -
                      alpha->imag() * b[i__3 - b_offset].imag()),
                z__1.imag(alpha->real() * b[i__3 - b_offset].imag() +
                          alpha->imag() * b[i__3 - b_offset].real());
            temp.real(z__1.real()), temp.imag(z__1.imag());
            i__3 = *m;
            for (i__ = 1; i__ <= i__3; ++i__) {
              i__4 = i__ + j * c_dim1;
              i__5 = i__ + j * c_dim1;
              i__6 = i__ + l * a_dim1;
              z__2.real(temp.real() * a[i__6 - a_offset].real() -
                        temp.imag() * a[i__6 - a_offset].imag()),
                  z__2.imag(temp.real() * a[i__6 - a_offset].imag() +
                            temp.imag() * a[i__6 - a_offset].real());
              z__1.real(c__[i__5 - c_offset].real() + z__2.real()),
                  z__1.imag(c__[i__5 - c_offset].imag() + z__2.imag());
              c__[i__4 - c_offset].real(z__1.real()), c__[i__4 - c_offset].imag(z__1.imag());
              /* L70: */
            }
          }
          /* L80: */
        }
        /* L90: */
      }
    } else if (conja) {

      /*           Form  C := alpha*conjg( A' )*B + beta*C. */

      i__1 = *n;
      for (j = 1; j <= i__1; ++j) {
        i__2 = *m;
        for (i__ = 1; i__ <= i__2; ++i__) {
          temp.real(0.), temp.imag(0.);
          i__3 = *k;
          for (l = 1; l <= i__3; ++l) {
            d_cnjg(&z__3, &a[l + i__ * a_dim1]);
            i__4 = l + j * b_dim1;
            z__2.real(z__3.real() * b[i__4].real() -
                      z__3.imag() * b[i__4].imag()),
                z__2.imag(z__3.real() * b[i__4].imag() +
                          z__3.imag() * b[i__4].real());
            z__1.real(temp.real() + z__2.real()),
                z__1.imag(temp.imag() + z__2.imag());
            temp.real(z__1.real()), temp.imag(z__1.imag());
            /* L100: */
          }
          if (beta->real() == 0. && beta->imag() == 0.) {
            i__3 = i__ + j * c_dim1;
            z__1.real(alpha->real() * temp.real() -
                      alpha->imag() * temp.imag()),
                z__1.imag(alpha->real() * temp.imag() +
                          alpha->imag() * temp.real());
            c__[i__3 - c_offset].real(z__1.real()), c__[i__3 - c_offset].imag(z__1.imag());
          } else {
            i__3 = i__ + j * c_dim1;
            z__2.real(alpha->real() * temp.real() -
                      alpha->imag() * temp.imag()),
                z__2.imag(alpha->real() * temp.imag() +
                          alpha->imag() * temp.real());
            i__4 = i__ + j * c_dim1;
            z__3.real(beta->real() * c__[i__4 - c_offset].real() -
                      beta->imag() * c__[i__4 - c_offset].imag()),
                z__3.imag(beta->real() * c__[i__4 - c_offset].imag() +
                          beta->imag() * c__[i__4 - c_offset].real());
            z__1.real(z__2.real() + z__3.real()),
                z__1.imag(z__2.imag() + z__3.imag());
            c__[i__3 - c_offset].real(z__1.real()), c__[i__3 - c_offset].imag(z__1.imag());
          }
          /* L110: */
        }
        /* L120: */
      }
    } else {

      /*           Form  C := alpha*A'*B + beta*C */

      i__1 = *n;
      for (j = 1; j <= i__1; ++j) {
        i__2 = *m;
        for (i__ = 1; i__ <= i__2; ++i__) {
          temp.real(0.), temp.imag(0.);
          i__3 = *k;
          for (l = 1; l <= i__3; ++l) {
            i__4 = l + i__ * a_dim1;
            i__5 = l + j * b_dim1;
            z__2.real(a[i__4].real() * b[i__5].real() -
                      a[i__4].imag() * b[i__5].imag()),
                z__2.imag(a[i__4].real() * b[i__5].imag() +
                          a[i__4].imag() * b[i__5].real());
            z__1.real(temp.real() + z__2.real()),
                z__1.imag(temp.imag() + z__2.imag());
            temp.real(z__1.real()), temp.imag(z__1.imag());
            /* L130: */
          }
          if (beta->real() == 0. && beta->imag() == 0.) {
            i__3 = i__ + j * c_dim1;
            z__1.real(alpha->real() * temp.real() -
                      alpha->imag() * temp.imag()),
                z__1.imag(alpha->real() * temp.imag() +
                          alpha->imag() * temp.real());
            c__[i__3].real(z__1.real()), c__[i__3].imag(z__1.imag());
          } else {
            i__3 = i__ + j * c_dim1;
            z__2.real(alpha->real() * temp.real() -
                      alpha->imag() * temp.imag()),
                z__2.imag(alpha->real() * temp.imag() +
                          alpha->imag() * temp.real());
            i__4 = i__ + j * c_dim1;
            z__3.real(beta->real() * c__[i__4].real() -
                      beta->imag() * c__[i__4].imag()),
                z__3.imag(beta->real() * c__[i__4].imag() +
                          beta->imag() * c__[i__4].real());
            z__1.real(z__2.real() + z__3.real()),
                z__1.imag(z__2.imag() + z__3.imag());
            c__[i__3].real(z__1.real()), c__[i__3].imag(z__1.imag());
          }
          /* L140: */
        }
        /* L150: */
      }
    }
  } else if (nota) {
    if (conjb) {

      /*           Form  C := alpha*A*conjg( B' ) + beta*C. */

      i__1 = *n;
      for (j = 1; j <= i__1; ++j) {
        if (beta->real() == 0. && beta->imag() == 0.) {
          i__2 = *m;
          for (i__ = 1; i__ <= i__2; ++i__) {
            i__3 = i__ + j * c_dim1;
            c__[i__3].real(0.), c__[i__3].imag(0.);
            /* L160: */
          }
        } else if (beta->real() != 1. || beta->imag() != 0.) {
          i__2 = *m;
          for (i__ = 1; i__ <= i__2; ++i__) {
            i__3 = i__ + j * c_dim1;
            i__4 = i__ + j * c_dim1;
            z__1.real(beta->real() * c__[i__4].real() -
                      beta->imag() * c__[i__4].imag()),
                z__1.imag(beta->real() * c__[i__4].imag() +
                          beta->imag() * c__[i__4].real());
            c__[i__3].real(z__1.real()), c__[i__3].imag(z__1.imag());
            /* L170: */
          }
        }
        i__2 = *k;
        for (l = 1; l <= i__2; ++l) {
          i__3 = j + l * b_dim1;
          if (b[i__3].real() != 0. || b[i__3].imag() != 0.) {
            d_cnjg(&z__2, &b[j + l * b_dim1]);
            z__1.real(alpha->real() * z__2.real() -
                      alpha->imag() * z__2.imag()),
                z__1.imag(alpha->real() * z__2.imag() +
                          alpha->imag() * z__2.real());
            temp.real(z__1.real()), temp.imag(z__1.imag());
            i__3 = *m;
            for (i__ = 1; i__ <= i__3; ++i__) {
              i__4 = i__ + j * c_dim1;
              i__5 = i__ + j * c_dim1;
              i__6 = i__ + l * a_dim1;
              z__2.real(temp.real() * a[i__6].real() -
                        temp.imag() * a[i__6].imag()),
                  z__2.imag(temp.real() * a[i__6].imag() +
                            temp.imag() * a[i__6].real());
              z__1.real(c__[i__5].real() + z__2.real()),
                  z__1.imag(c__[i__5].imag() + z__2.imag());
              c__[i__4].real(z__1.real()), c__[i__4].imag(z__1.imag());
              /* L180: */
            }
          }
          /* L190: */
        }
        /* L200: */
      }
    } else {

      /*           Form  C := alpha*A*B'          + beta*C */

      i__1 = *n;
      for (j = 1; j <= i__1; ++j) {
        if (beta->real() == 0. && beta->imag() == 0.) {
          i__2 = *m;
          for (i__ = 1; i__ <= i__2; ++i__) {
            i__3 = i__ + j * c_dim1;
            c__[i__3].real(0.), c__[i__3].imag(0.);
            /* L210: */
          }
        } else if (beta->real() != 1. || beta->imag() != 0.) {
          i__2 = *m;
          for (i__ = 1; i__ <= i__2; ++i__) {
            i__3 = i__ + j * c_dim1;
            i__4 = i__ + j * c_dim1;
            z__1.real(beta->real() * c__[i__4].real() -
                      beta->imag() * c__[i__4].imag()),
                z__1.imag(beta->real() * c__[i__4].imag() +
                          beta->imag() * c__[i__4].real());
            c__[i__3].real(z__1.real()), c__[i__3].imag(z__1.imag());
            /* L220: */
          }
        }
        i__2 = *k;
        for (l = 1; l <= i__2; ++l) {
          i__3 = j + l * b_dim1;
          if (b[i__3].real() != 0. || b[i__3].imag() != 0.) {
            i__3 = j + l * b_dim1;
            z__1.real(alpha->real() * b[i__3].real() -
                      alpha->imag() * b[i__3].imag()),
                z__1.imag(alpha->real() * b[i__3].imag() +
                          alpha->imag() * b[i__3].real());
            temp.real(z__1.real()), temp.imag(z__1.imag());
            i__3 = *m;
            for (i__ = 1; i__ <= i__3; ++i__) {
              i__4 = i__ + j * c_dim1;
              i__5 = i__ + j * c_dim1;
              i__6 = i__ + l * a_dim1;
              z__2.real(temp.real() * a[i__6].real() -
                        temp.imag() * a[i__6].imag()),
                  z__2.imag(temp.real() * a[i__6].imag() +
                            temp.imag() * a[i__6].real());
              z__1.real(c__[i__5].real() + z__2.real()),
                  z__1.imag(c__[i__5].imag() + z__2.imag());
              c__[i__4].real(z__1.real()), c__[i__4].imag(z__1.imag());
              /* L230: */
            }
          }
          /* L240: */
        }
        /* L250: */
      }
    }
  } else if (conja) {
    if (conjb) {

      /*           Form  C := alpha*conjg( A' )*conjg( B' ) + beta*C. */

      i__1 = *n;
      for (j = 1; j <= i__1; ++j) {
        i__2 = *m;
        for (i__ = 1; i__ <= i__2; ++i__) {
          temp.real(0.), temp.imag(0.);
          i__3 = *k;
          for (l = 1; l <= i__3; ++l) {
            d_cnjg(&z__3, &a[l + i__ * a_dim1]);
            d_cnjg(&z__4, &b[j + l * b_dim1]);
            z__2.real(z__3.real() * z__4.real() - z__3.imag() * z__4.imag()),
                z__2.imag(z__3.real() * z__4.imag() +
                          z__3.imag() * z__4.real());
            z__1.real(temp.real() + z__2.real()),
                z__1.imag(temp.imag() + z__2.imag());
            temp.real(z__1.real()), temp.imag(z__1.imag());
            /* L260: */
          }
          if (beta->real() == 0. && beta->imag() == 0.) {
            i__3 = i__ + j * c_dim1;
            z__1.real(alpha->real() * temp.real() -
                      alpha->imag() * temp.imag()),
                z__1.imag(alpha->real() * temp.imag() +
                          alpha->imag() * temp.real());
            c__[i__3].real(z__1.real()), c__[i__3].imag(z__1.imag());
          } else {
            i__3 = i__ + j * c_dim1;
            z__2.real(alpha->real() * temp.real() -
                      alpha->imag() * temp.imag()),
                z__2.imag(alpha->real() * temp.imag() +
                          alpha->imag() * temp.real());
            i__4 = i__ + j * c_dim1;
            z__3.real(beta->real() * c__[i__4].real() -
                      beta->imag() * c__[i__4].imag()),
                z__3.imag(beta->real() * c__[i__4].imag() +
                          beta->imag() * c__[i__4].real());
            z__1.real(z__2.real() + z__3.real()),
                z__1.imag(z__2.imag() + z__3.imag());
            c__[i__3].real(z__1.real()), c__[i__3].imag(z__1.imag());
          }
          /* L270: */
        }
        /* L280: */
      }
    } else {

      /*           Form  C := alpha*conjg( A' )*B' + beta*C */

      i__1 = *n;
      for (j = 1; j <= i__1; ++j) {
        i__2 = *m;
        for (i__ = 1; i__ <= i__2; ++i__) {
          temp.real(0.), temp.imag(0.);
          i__3 = *k;
          for (l = 1; l <= i__3; ++l) {
            d_cnjg(&z__3, &a[l + i__ * a_dim1]);
            i__4 = j + l * b_dim1;
            z__2.real(z__3.real() * b[i__4].real() -
                      z__3.imag() * b[i__4].imag()),
                z__2.imag(z__3.real() * b[i__4].imag() +
                          z__3.imag() * b[i__4].real());
            z__1.real(temp.real() + z__2.real()),
                z__1.imag(temp.imag() + z__2.imag());
            temp.real(z__1.real()), temp.imag(z__1.imag());
            /* L290: */
          }
          if (beta->real() == 0. && beta->imag() == 0.) {
            i__3 = i__ + j * c_dim1;
            z__1.real(alpha->real() * temp.real() -
                      alpha->imag() * temp.imag()),
                z__1.imag(alpha->real() * temp.imag() +
                          alpha->imag() * temp.real());
            c__[i__3].real(z__1.real()), c__[i__3].imag(z__1.imag());
          } else {
            i__3 = i__ + j * c_dim1;
            z__2.real(alpha->real() * temp.real() -
                      alpha->imag() * temp.imag()),
                z__2.imag(alpha->real() * temp.imag() +
                          alpha->imag() * temp.real());
            i__4 = i__ + j * c_dim1;
            z__3.real(beta->real() * c__[i__4].real() -
                      beta->imag() * c__[i__4].imag()),
                z__3.imag(beta->real() * c__[i__4].imag() +
                          beta->imag() * c__[i__4].real());
            z__1.real(z__2.real() + z__3.real()),
                z__1.imag(z__2.imag() + z__3.imag());
            c__[i__3].real(z__1.real()), c__[i__3].imag(z__1.imag());
          }
          /* L300: */
        }
        /* L310: */
      }
    }
  } else {
    if (conjb) {

      /*           Form  C := alpha*A'*conjg( B' ) + beta*C */

      i__1 = *n;
      for (j = 1; j <= i__1; ++j) {
        i__2 = *m;
        for (i__ = 1; i__ <= i__2; ++i__) {
          temp.real(0.), temp.imag(0.);
          i__3 = *k;
          for (l = 1; l <= i__3; ++l) {
            i__4 = l + i__ * a_dim1;
            d_cnjg(&z__3, &b[j + l * b_dim1]);
            z__2.real(a[i__4].real() * z__3.real() -
                      a[i__4].imag() * z__3.imag()),
                z__2.imag(a[i__4].real() * z__3.imag() +
                          a[i__4].imag() * z__3.real());
            z__1.real(temp.real() + z__2.real()),
                z__1.imag(temp.imag() + z__2.imag());
            temp.real(z__1.real()), temp.imag(z__1.imag());
            /* L320: */
          }
          if (beta->real() == 0. && beta->imag() == 0.) {
            i__3 = i__ + j * c_dim1;
            z__1.real(alpha->real() * temp.real() -
                      alpha->imag() * temp.imag()),
                z__1.imag(alpha->real() * temp.imag() +
                          alpha->imag() * temp.real());
            c__[i__3].real(z__1.real()), c__[i__3].imag(z__1.imag());
          } else {
            i__3 = i__ + j * c_dim1;
            z__2.real(alpha->real() * temp.real() -
                      alpha->imag() * temp.imag()),
                z__2.imag(alpha->real() * temp.imag() +
                          alpha->imag() * temp.real());
            i__4 = i__ + j * c_dim1;
            z__3.real(beta->real() * c__[i__4].real() -
                      beta->imag() * c__[i__4].imag()),
                z__3.imag(beta->real() * c__[i__4].imag() +
                          beta->imag() * c__[i__4].real());
            z__1.real(z__2.real() + z__3.real()),
                z__1.imag(z__2.imag() + z__3.imag());
            c__[i__3].real(z__1.real()), c__[i__3].imag(z__1.imag());
          }
          /* L330: */
        }
        /* L340: */
      }
    } else {

      /*           Form  C := alpha*A'*B' + beta*C */

      i__1 = *n;
      for (j = 1; j <= i__1; ++j) {
        i__2 = *m;
        for (i__ = 1; i__ <= i__2; ++i__) {
          temp.real(0.), temp.imag(0.);
          i__3 = *k;
          for (l = 1; l <= i__3; ++l) {
            i__4 = l + i__ * a_dim1;
            i__5 = j + l * b_dim1;
            z__2.real(a[i__4].real() * b[i__5].real() -
                      a[i__4].imag() * b[i__5].imag()),
                z__2.imag(a[i__4].real() * b[i__5].imag() +
                          a[i__4].imag() * b[i__5].real());
            z__1.real(temp.real() + z__2.real()),
                z__1.imag(temp.imag() + z__2.imag());
            temp.real(z__1.real()), temp.imag(z__1.imag());
            /* L350: */
          }
          if (beta->real() == 0. && beta->imag() == 0.) {
            i__3 = i__ + j * c_dim1;
            z__1.real(alpha->real() * temp.real() -
                      alpha->imag() * temp.imag()),
                z__1.imag(alpha->real() * temp.imag() +
                          alpha->imag() * temp.real());
            c__[i__3].real(z__1.real()), c__[i__3].imag(z__1.imag());
          } else {
            i__3 = i__ + j * c_dim1;
            z__2.real(alpha->real() * temp.real() -
                      alpha->imag() * temp.imag()),
                z__2.imag(alpha->real() * temp.imag() +
                          alpha->imag() * temp.real());
            i__4 = i__ + j * c_dim1;
            z__3.real(beta->real() * c__[i__4].real() -
                      beta->imag() * c__[i__4].imag()),
                z__3.imag(beta->real() * c__[i__4].imag() +
                          beta->imag() * c__[i__4].real());
            z__1.real(z__2.real() + z__3.real()),
                z__1.imag(z__2.imag() + z__3.imag());
            c__[i__3].real(z__1.real()), c__[i__3].imag(z__1.imag());
          }
          /* L360: */
        }
        /* L370: */
      }
    }
  }

  return 0;

  /*     End of ZGEMM . */

} /* zgemm_ */

template <std::size_t N> void print(std::array<std::complex<fp>, N> matrix) {
  int i{};
  for (auto&& a : matrix) {
    std::cerr << std::setprecision(50) << a.real() << '+' << a.imag() << "i, ";
    if (++i % 4 == 0) {
      std::cerr << '\n';
    }
  }
  std::cerr << '\n';
}

auto mul(qfp a, qfp b) {
  return qfp((a.real() * b.real() - a.imag() * b.imag()),
             (a.real() * b.imag() + a.imag() * b.real()));
}

// Function to perform SYRK (Symmetric Rank-K update) on 4x4 matrices stored in
// std::array
matrix4x4 syrk(bool upper, fp alpha, const matrix4x4& A, fp beta) {
  matrix4x4 C;
  // Iterate over the matrix rows and columns
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = (upper ? i : 0); j < 4; ++j) {
      // Compute the dot product A[i, :] * A[j, :] (real and imaginary
      // separately)
      qfp sum{0.0, 0.0}; // Initialize sum as a complex number (0.0 + 0.0i)

      for (size_t k = 0; k < 4; ++k) {
        sum += A[i * 4 + k] * std::conj(A[j * 4 + k]); // A[i, :] * A[j, :]
      }

      // Apply the SYRK update: C(i, j) = alpha * sum + beta * C(i, j)
      C[i * 4 + j] = alpha * sum + beta * C[i * 4 + j];

      // If updating the lower triangle, mirror the values from the upper
      // triangle
      if (!upper && i != j) {
        C[j * 4 + i] = C[i * 4 + j]; // Maintain symmetry
      }
    }
  }
  return C;
}

template <typename T, std::size_t N>
[[nodiscard]] inline auto multiply(const std::array<T, N>& lhs,
                                   const std::array<T, N>& rhs) {
  // return matrixMultiplyWithKahan(lhs, rhs);
  std::array<T, N> result{};
  const int n = std::sqrt(N);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        std::cout << std::setprecision(17) << lhs[i * n + k] << " * "
                  << rhs[k * n + j] << " = "
                  << mul(lhs[i * n + k], rhs[k * n + j]) << '\n';
        result[i * n + j] += mul(lhs[i * n + k], rhs[k * n + j]);
        std::cout << std::setprecision(50) << "->" << result[i * n + j] << '\n';
      }
      std::cout << "\n===\n";
    }
  }
  return result;
}

static matrix4x4 transpose(const matrix4x4& matrix) {
  matrix4x4 result;
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j < 4; ++j) {
      result[j * 4 + i] = matrix[i * 4 + j];
    }
  }
  return result;
}

int main() {
  matrix4x4 a = {qfp(0.3535533905932738, +0.35355339059327373),
                 qfp(-0.35355339059327373, +0.3535533905932738),
                 qfp(-0.35355339059327373, +0.3535533905932738),
                 qfp(0.3535533905932738, +0.35355339059327373),
                 qfp(0.35355339059327373, -0.3535533905932738),
                 qfp(0.3535533905932738, +0.35355339059327373),
                 qfp(-0.3535533905932738, -0.35355339059327373),
                 qfp(-0.35355339059327373, +0.3535533905932738),
                 qfp(0.35355339059327373, -0.3535533905932738),
                 qfp(-0.3535533905932738, -0.35355339059327373),
                 qfp(0.3535533905932738, +0.35355339059327373),
                 qfp(-0.35355339059327373, +0.3535533905932738),
                 qfp(0.3535533905932738, +0.35355339059327373),
                 qfp(0.35355339059327373, -0.3535533905932738),
                 qfp(0.35355339059327373, -0.3535533905932738),
                 qfp(0.3535533905932738, +0.35355339059327373)};

  // print(multiply(transpose(a), transpose(transpose(a))));
  // print(syrk(false, 1.0, transpose(a), 0.0));
  print(zgemm(transpose(a), a));
}

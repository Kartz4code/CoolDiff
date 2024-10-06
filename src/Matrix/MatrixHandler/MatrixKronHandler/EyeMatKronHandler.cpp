/**
 * @file src/Matrix/MatrixHandler/MatrixKronHandler/EyeMatKronHandler.cpp
 *
 * @copyright 2023-2024 Karthik Murali Madhavan Rathai
 */
/*
 * This file is part of CoolDiff library.
 *
 * You can redistribute it and/or modify it under the terms of the GNU
 * General Public License version 3 as published by the Free Software
 * Foundation.
 *
 * Licensees holding a valid commercial license may use this software
 * in accordance with the commercial license agreement provided in
 * conjunction with the software.  The terms and conditions of any such
 * commercial license agreement shall govern, supersede, and render
 * ineffective any application of the GPLv3 license to this software,
 * notwithstanding of any reference thereto in the software or
 * associated repository.
 */

#include "EyeMatKronHandler.hpp"
#include "Matrix.hpp"
#include "MatrixEyeOps.hpp"

// When left matrix is special matrix of identity type
void KronEyeLHS(const Matrix<Type> *lhs, const Matrix<Type> *rhs,
                Matrix<Type> *&result) {
  /* Matrix-Matrix numerical Kronocker product */
  // Rows and columns of result matrix and if result is nullptr, then create a
  // new resource
  const size_t lr{lhs->getNumRows()};
  const size_t lc{lhs->getNumColumns()};
  const size_t rr{rhs->getNumRows()};
  const size_t rc{rhs->getNumColumns()};

  MatrixPool((lr*rr), (lc*rc), result);

  const auto lhs_idx = Range<size_t>(0, lr * lc);
  const auto rhs_idx = Range<size_t>(0, rr * rc);
  std::for_each(EXECUTION_PAR lhs_idx.begin(), lhs_idx.end(),
                [&](const size_t n1) {
                  const size_t j{n1 % lc};
                  const size_t i{(n1 - j) / lc};

                  // If i == j, then val is 1, else zero (LHS identity creation)
                  Type val = ((i == j) ? (Type)(1) : (Type)(0));

                  // If val is not zero
                  if ((Type)(0) != val) {
                    std::for_each(EXECUTION_PAR rhs_idx.begin(), rhs_idx.end(),
                                  [&](const size_t n2) {
                                    const size_t m{n2 % rc};
                                    const size_t l{(n2 - m) / rc};
                                    (*result)(i * rr + l, j * rc + m) =
                                        (*rhs)(l, m) * val;
                                  });
                  }
                });
}

// When right matrix is special matrix of identity type
void KronEyeRHS(const Matrix<Type> *lhs, const Matrix<Type> *rhs,
                Matrix<Type> *&result) {
  /* Matrix-Matrix numerical Kronocker product */
  // Rows and columns of result matrix and if result is nullptr, then create a
  // new resource
  const size_t lr{lhs->getNumRows()};
  const size_t lc{lhs->getNumColumns()};
  const size_t rr{rhs->getNumRows()};
  const size_t rc{rhs->getNumColumns()};

  MatrixPool((lr*rr), (lc*rc), result);

  const auto lhs_idx = Range<size_t>(0, lr * lc);
  const auto rhs_idx = Range<size_t>(0, rr * rc);
  std::for_each(EXECUTION_PAR lhs_idx.begin(), lhs_idx.end(),
                [&](const size_t n1) {
                  const size_t j{n1 % lc};
                  const size_t i{(n1 - j) / lc};

                  // Value of LHS matrix at (i,j) index
                  Type val = (*lhs)(i, j);

                  // If val is not zero
                  if ((Type)(0) != val) {
                    std::for_each(EXECUTION_PAR rhs_idx.begin(), rhs_idx.end(),
                                  [&](const size_t n2) {
                                    const size_t m{n2 % rc};
                                    const size_t l{(n2 - m) / rc};
                                    (*result)(i * rr + l, j * rc + m) =
                                        ((l == m) ? val : (Type)(0));
                                  });
                  }
                });
}

void EyeMatKronHandler::handle(const Matrix<Type> *lhs, const Matrix<Type> *rhs,
                               Matrix<Type> *&result) {
#if defined(NAIVE_IMPL)
  /* Eye matrix special check */
  if (auto *it = EyeMatKron(lhs, rhs); nullptr != it) {
    if (it == lhs) {
      KronEyeRHS(lhs, rhs, result);
    } else if (it == rhs) {
      KronEyeLHS(lhs, rhs, result);
    } else {
      result = const_cast<Matrix<Type> *>(it);
    }
    return;
  }

  /* Eye matrix numerical check */
  else if (auto *it = EyeMatKronNum(lhs, rhs); nullptr != it) {
    if (it == lhs) {
      KronEyeRHS(lhs, rhs, result);
    } else if (it == rhs) {
      KronEyeLHS(lhs, rhs, result);
    } else {
      result = const_cast<Matrix<Type> *>(it);
    }
    return;
  }
#endif

  // Chain of responsibility
  MatrixHandler::handle(lhs, rhs, result);
}
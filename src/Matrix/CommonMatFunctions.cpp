/**
 * @file src/Matrix/CommonMatFunctions.cpp
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

#include "CommonMatFunctions.hpp"

#include "CommonFunctions.hpp"

// Is the matrix zero
bool IsZeroMatrix(const Matrix<Type> &m) {
  auto *it = m.getMatrixPtr();
  const size_t n = m.getNumElem();
  return std::all_of(EXECUTION_PAR it, it + n,
                     [](Type i) { return (i == (Type)(0)); });
}

// Is the matrix identity
bool IsEyeMatrix(const Matrix<Type> &m) {
  // If matrix is rectangular, return false
  if (false == IsSquareMatrix(m)) {
    return false;
  }
  const size_t rows = m.getNumRows();
  for (size_t i{}; i < rows; ++i) {
    if (m(i, i) != (Type)(1)) {
      return false;
    }
  }
  return true;
}

// Is the matrix ones
bool IsOnesMatrix(const Matrix<Type> &m) {
  auto *it = m.getMatrixPtr();
  const size_t n = m.getNumElem();
  return std::all_of(EXECUTION_PAR it, it + n,
                     [](Type i) { return (i == (Type)(1)); });
}

// Is the matrix square
bool IsSquareMatrix(const Matrix<Type> &m) {
  return (m.getNumColumns() == m.getNumRows());
}

// Is the matrix diagonal?
bool IsDiagMatrix(const Matrix<Type> &m) {
  // If matrix is rectangular, return false
  if (false == IsSquareMatrix(m)) {
    return false;
  }
  const size_t rows = m.getNumRows();
  const size_t cols = m.getNumColumns();
  for (size_t i{}; i < rows; ++i) {
    for (size_t j{}; j < cols; ++j) {
      if ((i != j) && (m(i, j) != (Type)(0))) {
        return false;
      }
    }
  }
  return true;
}

// Is the row matrix ?
bool IsRowMatrix(const Matrix<Type> &m) { return (m.getNumRows() == 1); }

// Is the column matrix ?
bool IsColMatrix(const Matrix<Type> &m) { return (m.getNumColumns() == 1); }

// Find type of matrix
size_t FindMatType(const Matrix<Type> &m) {
  size_t result{};
  // Zero matrix check
  if (true == IsZeroMatrix(m)) {
    result |= MatrixSpl::ZEROS;
  }
  // One matrix check
  if (true == IsOnesMatrix(m)) {
    result |= MatrixSpl::ONES;
  }
  // Identity matrix check
  if (true == IsEyeMatrix(m)) {
    result |= MatrixSpl::EYE;
  }
  // Diagonal matrix check
  if (true == IsDiagMatrix(m)) {
    result |= MatrixSpl::DIAG;
  }
  // Row matrix check
  if (true == IsRowMatrix(m)) {
    result |= MatrixSpl::ROW_MAT;
  }
  // Column matrix check
  if (true == IsRowMatrix(m)) {
    result |= MatrixSpl::COL_MAT;
  }
  return result;
}

// Matrix evaluation
Matrix<Type> &Eval(Matrix<Expression> &mexp) {
  // Create new matrix
  Matrix<Type> &res =
      CreateMatrix<Type>(mexp.getNumRows(), mexp.getNumColumns());
  // Reset graph/tree
  mexp.resetImpl();
  res = *(mexp.eval());
  // Return evaluation value
  return res;
}

// Matrix derivative evaluation
Matrix<Type> &DevalF(Matrix<Expression> &mexp, const Variable &x) {
  // Create new matrix
  Matrix<Type> &dres =
      CreateMatrix<Type>(mexp.getNumRows(), mexp.getNumColumns());
  // Reset graph/tree
  mexp.resetImpl();
  dres = *(mexp.devalF(x));
  // Return evaluation value
  return dres;
}
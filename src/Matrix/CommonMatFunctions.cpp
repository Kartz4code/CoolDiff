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
#include "Matrix.hpp"

/*
// Is the matrix ones
bool IsOnesMatrix(const Matrix<Type> &m) {
  // If special matrix and of corresponding type, return
  if(m.getMatType() == MatrixSpl::ONES) {
    return true;
  }

  // Get Matrix pointer and number of elements
  auto *it = m.getMatrixPtr();
  const size_t n = m.getNumElem();

  // Check all elements for one
  return std::all_of(EXECUTION_PAR 
                     it, it + n,
                     [](Type i) { return (i == (Type)(1)); });
}


// Is the matrix diagonal?
bool IsDiagMatrix(const Matrix<Type> &m) {
  // If special matrix and of corresponding type, return
  if(m.getMatType() == MatrixSpl::DIAG) {
    return true;
  }

  // If matrix is rectangular, return false
  if (false == IsSquareMatrix(m)) {
    return false;
  }

  if(true == IsZeroMatrix(m)) {
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
bool IsRowMatrix(const Matrix<Type> &m) { 
  // If special matrix and of corresponding type, return
  if(m.getMatType() == MatrixSpl::ROW_MAT) {
    return true;
  }

  // If special matrix and of corresponding type, return
  if(m.getMatType() == MatrixSpl::ZEROS) {
    return true;
  }
  return (m.getNumRows() == 1); 
}

// Is the column matrix ?
bool IsColMatrix(const Matrix<Type> &m) { 
  // If special matrix and of corresponding type, return
  if(m.getMatType() == MatrixSpl::COL_MAT) {
    return true;
  }

  return (m.getNumColumns() == 1); 
}

// Find type of matrix
size_t FindMatType(const Matrix<Type> &m) {
  size_t result{0};
  // Zero matrix check
  if (true == IsZeroMatrix(m)) {
    result |= MatrixSpl::ZEROS;
  }
  // One matrix check
  else if (true == IsOnesMatrix(m)) {
    result |= MatrixSpl::ONES;
  }
  // Identity matrix check
  else if (true == IsEyeMatrix(m)) {
    result |= MatrixSpl::EYE;
  }
  // Diagonal matrix check
  else if (true == IsDiagMatrix(m)) {
    result |= MatrixSpl::DIAG;
  }
  // Row matrix check
  else if (true == IsRowMatrix(m)) {
    result |= MatrixSpl::ROW_MAT;
  }
  // Column matrix check
  else if (true == IsColMatrix(m)) {
    result |= MatrixSpl::COL_MAT;
  } 
  // If none of special type
  else {
  result = -1;
  }  
  return result;
}
*/

// Matrix evaluation
Matrix<Type> &Eval(Matrix<Expression> &Mexp) {
  // Reset graph/tree
  Mexp.resetImpl();
  // Return evaluation value
  return *(Mexp.eval());
}

// Matrix-Matrix derivative evaluation
Matrix<Type>& DevalF(Matrix<Expression>& Mexp, Matrix<Variable>& X) {
  // Reset graph/tree
  Mexp.resetImpl();
  // Return evaluation value
  return *(Mexp.devalF(X));
}
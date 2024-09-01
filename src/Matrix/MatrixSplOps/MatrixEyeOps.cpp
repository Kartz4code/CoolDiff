/**
 * @file src/Matrix/MatrixSplOps/MatrixEyeOps.cpp
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

#include "MatrixEyeOps.hpp"
#include "CommonMatFunctions.hpp"

// Is the matrix square
bool IsSquareMatrix(Matrix<Type>* m) {
  // Null pointer check
  NULL_CHECK(m, "Matrix (m) is a nullptr");
  // Check for square matrix
  return (m->getNumColumns() == m->getNumRows());
}
// Is the matrix identity
bool IsEyeMatrix(Matrix<Type>* m) {
  // Null pointer check
  NULL_CHECK(m, "Matrix (m) is a nullptr");

  // If matrix not square, return false
  if (false == IsSquareMatrix(m)) {
    return false;
  }

  // Eye special matrix check 
  if(m->getMatType() == MatrixSpl::EYE) {
    return true;
  } 
  // else if m is some special matrix
  else if(m->getMatType() != -1) {
    return false;
  }

  // Rows and columns
  const size_t rows = m->getNumRows();
  const size_t cols = m->getNumColumns();

  // Diagonal elements check (1's)
  Vector<size_t> elemD(rows);
  std::iota(elemD.begin(), elemD.end(), 0);
  if(auto it = std::find_if(EXECUTION_PAR elemD.begin(), elemD.end(), 
               [&m](const size_t i) { return ((*m)(i, i) != (Type)(1)); }); 
          it != elemD.end()) {
    return false;
  }

  // Non-diagonal elements check (0's) 
  Vector<size_t> elemM(rows*cols);
  std::iota(elemM.begin(), elemM.end(), 0);
  if(auto it = std::find_if(EXECUTION_PAR elemM.begin(), elemM.end(), 
               [&m,rows,cols](const size_t n) {
                const size_t j = n%cols;
                const size_t i = (n-j)/cols;
                return ((i != j) && ((*m)(i,j) != (Type)(0))); }); 
          it != elemM.end()) {
    return false;
  }

  // If none of the above conditions are satisfied, return true
  return true;
}


// Eye matrix addition checks
Matrix<Type>* EyeMatAdd(Matrix<Type>* lhs, Matrix<Type>* rhs) {
  // Null pointer check
  NULL_CHECK(lhs, "LHS Matrix (lhs) is a nullptr");
  NULL_CHECK(rhs, "RHS Matrix (rhs) is a nullptr");

  // Left matrix rows and column numbers
  const size_t lr = lhs->getNumRows();
  const size_t rc = rhs->getNumColumns();

  // If both lhs and rhs matrices are zero matrices
  if(lhs->getMatType() == MatrixSpl::EYE &&
     rhs->getMatType() == MatrixSpl::EYE) {
      // Technically, rhs can also be returned since both lhs, rhs are eye
      return lhs;
  } else if(lhs->getMatType() == MatrixSpl::EYE) {
    return rhs;
  } else if(rhs->getMatType() == MatrixSpl::EYE) {
    return lhs;
  } else {
    return nullptr;
  }
}

// Eye matrix multiplication checks
Matrix<Type>* EyeMatMul(Matrix<Type>* lhs, Matrix<Type>* rhs) {
  // Null pointer check
  NULL_CHECK(lhs, "LHS Matrix (lhs) is a nullptr");
  NULL_CHECK(rhs, "RHS Matrix (rhs) is a nullptr");
    
  // Left matrix rows and column numbers
  const size_t lr = lhs->getNumRows();
  const size_t rc = rhs->getNumColumns();

  // If both lhs and rhs matrices are zero matrices
  if(lhs->getMatType() == MatrixSpl::EYE && 
     rhs->getMatType() == MatrixSpl::EYE) {
      // Technically, rhs can also be returned since both lhs, rhs are zero
      return lhs;
  }
  // If lhs is a zero matrix
  else if(lhs->getMatType() == MatrixSpl::EYE) {
    return rhs;
  // If rhs is a zero matrix
  } else if(rhs->getMatType() == MatrixSpl::EYE) {
    return lhs;
  } 
  // If neither, then return nullptr
  else {
      return nullptr;
  }
}

// Eye matrix addition numerical checks
Matrix<Type>* EyeMatAddNum(Matrix<Type>* lhs, Matrix<Type>* rhs) {
  // Null pointer check
  NULL_CHECK(lhs, "LHS Matrix (lhs) is a nullptr");
  NULL_CHECK(rhs, "RHS Matrix (rhs) is a nullptr");

  // Boolean check
  const bool lhs_bool = IsEyeMatrix(lhs);
  const bool rhs_bool = IsEyeMatrix(rhs);
  if(true == lhs_bool || true == rhs_bool) {
    return lhs;
  } else {
    return nullptr;
  }

}

// Eye matrix multiplication numerical check
Matrix<Type>* EyeMatMulNum(Matrix<Type>* lhs, Matrix<Type>* rhs) {
  // Null pointer check
  NULL_CHECK(lhs, "LHS Matrix (lhs) is a nullptr");
  NULL_CHECK(rhs, "RHS Matrix (rhs) is a nullptr");

  // Get rows and columns 
  const size_t lrows = lhs->getNumRows();
  const size_t rcols = rhs->getNumColumns();

  // Boolean check
  const bool lhs_bool = IsEyeMatrix(lhs);
  const bool rhs_bool = IsEyeMatrix(rhs);
  if(lhs_bool == true && rhs_bool == true) {
    return CreateMatrixPtr<Type>(lrows, rcols, MatrixSpl::EYE); 
  } else if(lhs_bool == true) {
    return rhs;
  } else if (rhs_bool == true) {
    return lhs;
  } else {
    return nullptr;
  }
}

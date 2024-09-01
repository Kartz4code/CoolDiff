/**
 * @file src/Matrix/MatrixSplOps/MatrixZeroOps.cpp
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

#include "MatrixZeroOps.hpp"
#include "CommonMatFunctions.hpp"

// Is the matrix zero 
bool IsZeroMatrix(Matrix<Type>* m) {
  // Null pointer check
  NULL_CHECK(m, "Matrix (m) is a nullptr");

  // Zero special matrix check 
  if(m->getMatType() == MatrixSpl::ZEROS) {
    return true;
  } 
  // else if m is some special matrix
  else if(m->getMatType() != -1) {
    return false;
  }

  // Get Matrix pointer and number of elements
  auto *it = m->getMatrixPtr();
  const size_t n = m->getNumElem();

  // Check all elements for zero
  return std::all_of(EXECUTION_PAR 
                     it, it + n,
                     [](Type i) { return (i == (Type)(0)); 
                     });
}

// Zero matrix addition checks
Matrix<Type>* ZeroMatAdd(Matrix<Type>* lhs, Matrix<Type>* rhs) {
  // Null pointer check
  NULL_CHECK(lhs, "LHS Matrix (lhs) is a nullptr");
  NULL_CHECK(rhs, "RHS Matrix (rhs) is a nullptr");

  // Rows and columns of result matrix    
  const size_t nrows{lhs->getNumRows()};
  const size_t ncols{rhs->getNumColumns()};

  // If both lhs and rhs matrices are zero matrices
  if(lhs->getMatType() == MatrixSpl::ZEROS && 
     rhs->getMatType() == MatrixSpl::ZEROS) {
    return CreateMatrixPtr<Type>(nrows, ncols, MatrixSpl::ZEROS); 
  }
  // If lhs is a zero matrix
  else if(lhs->getMatType() == MatrixSpl::ZEROS) {
    return rhs;
  // If rhs is a zero matrix
  } else if(rhs->getMatType() == MatrixSpl::ZEROS) {
    return lhs;
  } else {
    return nullptr;
  }
}
// Zero matrix multiplication checks
Matrix<Type>* ZeroMatMul(Matrix<Type>* lhs, Matrix<Type>* rhs) {
  // Null pointer check
  NULL_CHECK(lhs, "LHS Matrix (lhs) is a nullptr");
  NULL_CHECK(rhs, "RHS Matrix (rhs) is a nullptr");

  // Left matrix rows and right matrix columns
  const size_t lr = lhs->getNumRows();
  const size_t rc = rhs->getNumColumns();

  // If both lhs and rhs matrices are zero matrices
  if(lhs->getMatType() == MatrixSpl::ZEROS || 
     rhs->getMatType() == MatrixSpl::ZEROS) {    
      return CreateMatrixPtr<Type>(lr, rc, MatrixSpl::ZEROS);
  }
  // If neither, then return nullptr
  else {
    return nullptr;
  }
}

// Zero matrix addition numerical check
Matrix<Type>* ZeroMatAddNum(Matrix<Type>* lhs, Matrix<Type>* rhs) {
  // Null pointer check
  NULL_CHECK(lhs, "LHS Matrix (lhs) is a nullptr");
  NULL_CHECK(rhs, "RHS Matrix (rhs) is a nullptr");

  // Boolean check
  const bool lhs_bool = IsZeroMatrix(lhs);
  const bool rhs_bool = IsZeroMatrix(rhs);

  // Rows and columns of result matrix    
  const size_t nrows{lhs->getNumRows()};
  const size_t ncols{rhs->getNumColumns()};

  if(lhs_bool == true && rhs_bool == true) {
    return CreateMatrixPtr<Type>(nrows, ncols, MatrixSpl::ZEROS); 
  } else if(lhs_bool == true) {
    return rhs;
  } else if (rhs_bool == true) {
    return lhs;
  } else {
    return nullptr;
  }
}
// Zero matrix multiplication numerical checks
Matrix<Type>* ZeroMatMulNum(Matrix<Type>* lhs, Matrix<Type>* rhs) {
  // Null pointer check
  NULL_CHECK(lhs, "LHS Matrix (lhs) is a nullptr");
  NULL_CHECK(rhs, "RHS Matrix (rhs) is a nullptr");

  // Left matrix rows and column numbers
  const size_t lr = lhs->getNumRows();
  const size_t rc = rhs->getNumColumns();

  // Boolean check
  const bool lhs_bool = IsZeroMatrix(lhs);
  const bool rhs_bool = IsZeroMatrix(rhs); 
  if(lhs_bool == true || rhs_bool == true) {
    return CreateMatrixPtr<Type>(lr, rc, MatrixSpl::ZEROS);
  } else {
    return nullptr;
  }
}
